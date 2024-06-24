import torch
import torch.nn as nn
import torch.nn.init as init
import math
from einops import rearrange
import torch.nn.functional as F
import torch
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.nn as nn
import torch.nn.functional as F



class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class KBAFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk ** 2
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        uf = torch.nn.functional.unfold(x, kernel_size=selfk, padding=selfk // 2)

        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        x = attk @ uf.unsqueeze(-1)  #
        del attk, uf
        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = x.transpose(-1, -2).reshape(B, selfc, H, W)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)

        dselfb = att.transpose(-2, -1) @ dbias
        datt = dbias @ selfb.transpose(-2, -1)

        attk = att @ selfw
        uf = F.unfold(x, kernel_size=selfk, padding=selfk // 2)
        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)

        dattk = dx @ uf.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf = attk.transpose(-2, -1) @ dx
        del attk, uf

        dattk = dattk.view(B, H * W, -1)
        datt += dattk @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk

        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw
class KBBlock_s(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False):
        super(KBBlock_s, self).__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc
        self.w_custom = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b_custom = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w_custom, self.b_custom)

#         self.norm1 = LayerNorm2d(c)
#         self.norm2 = LayerNorm2d(c)
#         self.norm = torch.nn.BatchNorm2d(c, 0.001, 0.03)
#         self.relu = torch.nn.SiLU(inplace=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias=True),
            )

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.sg = SimpleGate()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp):
        x = inp

#         x = self.norm1(x)
        sca = self.sca(x)
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.KBA(uf, att, self.k, self.g, self.b_custom, self.w_custom) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
#         x = self.relu(self.norm(x))
        y = inp + x * self.beta
#         return inp + x
        # FFN
#         x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma


class TransAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TransAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(dim*2, dim , kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)
    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        # out2 = out
        # out = self.dropout(self.act(self.conv1(out)))
        # out = self.dropout(self.conv2(out)) + out2
        return out


class MFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, act=True, gc=2, nset=32, k=3):
        super(MFF, self).__init__()
        self.act = act
        self.gc = gc
#         self.norm = torch.nn.BatchNorm2d(dim, 0.001, 0.03)
#         self.relu = torch.nn.SiLU(inplace=True)
        hidden_features = int(dim * ffn_expansion_factor)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=hidden_features, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      groups=hidden_features, bias=bias),
        )

        c = hidden_features
        self.k, self.c = k, c
        self.nset = nset

        self.g = c // gc
        self.w_custom = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b_custom = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w_custom, self.b_custom)
        interc = min(dim,32)
        # print(c, interc)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )
        self.conv211 = nn.Conv2d(in_channels=dim, out_channels=self.nset, kernel_size=1)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.ga1 = nn.Parameter(torch.zeros((1, hidden_features, 1, 1)) + 1e-2, requires_grad=True)

    def forward(self, inp):
        x = inp
        sca = self.sca(x)
        x1 = self.dwconv(x)

        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv1(x)
        x2 = self.KBA(uf, att, self.k, self.g, self.b_custom, self.w_custom) * self.ga1 + uf

        x = F.gelu(x1) * x2 if self.act else x1 * x2
        x = x * sca
        x = self.project_out(x)
        return  inp + x

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)


class KBBlock_l(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(KBBlock_l, self).__init__()

#         self.norm1 = LayerNorm2d(dim)
#         self.norm2 = LayerNorm2d(dim)

#         self.attn = MFF(dim, ffn_expansion_factor, bias)
        self.ffn = TransAttention(dim, num_heads, bias)

    def forward(self, x):
#         x = x + self.attn(self.norm1(x))
#         x = x + self.ffn(self.norm2(x))
#         x = x + self.attn(x)
        x = x + self.ffn(x)
        return x
