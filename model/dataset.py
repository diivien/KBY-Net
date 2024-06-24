
import torch
import numpy as np
import random
import cv2
import os
import math
from PIL import Image
FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, main_dir, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size

        # Read labels
        cache = self.load_label(main_dir)
        clean_imgs, labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.filenames = list(cache.keys())  # update
        self.clean_filenames = list(clean_imgs)  # update

        self.n = len(shapes)  # number of samples
        self.indices = range(self.n)
        # Albumentations (optional, only used if package is installed)
        self.albumentations = Albumentations()
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):
        index = self.indices[index]

        params = self.params
        # Load image
        image,clean_image, shape = self.load_image(index)

        h, w = image.shape[:2]

        # Resize
        image,clean_image, ratio, pad = resize(image,clean_image, self.input_size, self.augment)

        shapes = shape, ((h / shape[0], w / shape[1]), pad)  # for COCO mAP rescaling

        label = self.labels[index].copy()
        if label.size:
            label[:, 1:] = wh2xypad(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
        if self.augment:
            image,clean_image, label = random_perspective(image,clean_image, label, params)
        nl = len(label)  # number of labels
        if nl:
            label[:, 1:5] = xy2whpad(label[:, 1:5], image.shape[1], image.shape[0])

        if self.augment:
            # Albumentations
            image,clean_image, label = self.albumentations(image,clean_image, label)
            nl = len(label)  # update after albumentations
            # HSV color-space
            augment_hsv(image,clean_image, params)
            # Flip up-down
            if random.random() < params['flip_ud']:
                image = np.flipud(image)
                clean_image = np.flipud(clean_image)
                if nl:
                    label[:, 2] = 1 - label[:, 2]
            # Flip left-right
            if random.random() < params['flip_lr']:
                image = np.fliplr(image)
                clean_image = np.fliplr(clean_image)
                if nl:
                    label[:, 1] = 1 - label[:, 1]

        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)
        
        clean_sample = clean_image.transpose((2, 0, 1))[::-1]
        clean_sample = np.ascontiguousarray(clean_sample)
        return torch.from_numpy(sample)/255,torch.from_numpy(clean_sample)/255, target, shapes

    def load_image(self, i):
        image = cv2.imread(self.filenames[i])
        clean_image = cv2.imread(self.clean_filenames[i])
        
        h, w = image.shape[:2]
        r = self.input_size / max(h, w)
        if r != 1:
            choice = resample()
            image = cv2.resize(image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=choice if self.augment else cv2.INTER_LINEAR)
            clean_image = cv2.resize(clean_image,
                               dsize=(int(w * r), int(h * r)),
                               interpolation=choice if self.augment else cv2.INTER_LINEAR)
        return image,clean_image, (h, w)

    @staticmethod
    def load_voc_label(main_dir):
        filenames = os.listdir(os.path.join(main_dir, "fog"))
#         path = f'{filenames[0]}.cache'
#         if os.path.exists(path):
#             return torch.load(path)
        x = {}
        for filename in filenames:
            try:
                # verify images
                with open(os.path.join(main_dir, "fog", filename), 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                base_name = filename.split('.jpg')[0]
                img_loc_clean = os.path.join(main_dir, "clean", base_name +".jpg")
                annotation_loc = os.path.join(main_dir, "annotations", base_name +".txt")

                annotations = []
                with open(annotation_loc, 'r') as file:

                    for line in file:
                        annotation = line.split()

                # Map the remaining classes to 0 to 4
                        class_label = int(annotation[0])
                        x_center, y_center, width, height = map(float, annotation[1:])
                        x_center =x_center /shape[0]
                        y_center =y_center/shape[1]
                        width = width /shape[0]
                        height = height/shape[1]

                        annotations.append([class_label, x_center, y_center, width, height])
                annotations = np.array(annotations)
                if filename:
                    x[os.path.join(main_dir, "fog", filename)] = [img_loc_clean, annotations, shape]
            except FileNotFoundError:
                pass
#         torch.save(x, path)
        return x
    @staticmethod
    def load_label(main_dir):
        class_mapping = {
            "person":0,
            "car":1,
            "bicycle":2,
            "motorcycle":3,
            "bus":4
        }
        filenames = os.listdir(os.path.join(main_dir, "rainy"))
#         path = f'{filenames[0]}.cache'
#         if os.path.exists(path):
#             return torch.load(path)
        x = {}
        for filename in filenames:
            try:
                # verify images
                with open(os.path.join(main_dir, "rainy", filename), 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                base_name = filename.split('_rain')[0]
                img_loc_clean = os.path.join(main_dir, "clean", base_name +".png")
                annotation_loc = os.path.join(main_dir, "annotations", base_name +".txt")

                annotations = []
                with open(annotation_loc, 'r') as file:

                    for line in file:
                        annotation = line.split()
                        if annotation[0] in ["rider","truck","train"]:
                            continue

                # Map the remaining classes to 0 to 4
                        class_label = class_mapping[annotation[0]]
                        left, top, width, height = map(float, annotation[1:])
                        x_center = (left + (width / 2)) /shape[0]
                        y_center = (top + (height / 2))/shape[1]
                        width = width /shape[0]
                        height = height/shape[1]

                        annotations.append([class_label, x_center, y_center, width, height])
                annotations = np.array(annotations)
                if filename:
                    x[os.path.join(main_dir, "rainy", filename)] = [img_loc_clean, annotations, shape]
            except FileNotFoundError:
                pass
#         torch.save(x, path)
        return x



def wh2xypad(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2whpad(x, w=640, h=640):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image,clean_image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    h1, s1, v1 = cv2.split(cv2.cvtColor(clean_image, cv2.COLOR_BGR2HSV))

    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = np.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = np.clip(x * r[2], 0, 255).astype('uint8')

    x1 = np.arange(0, 256, dtype=r.dtype)
    lut_h1 = ((x * r[0]) % 180).astype('uint8')
    lut_s1 = np.clip(x * r[1], 0, 255).astype('uint8')
    lut_v1 = np.clip(x * r[2], 0, 255).astype('uint8')
    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed
    
    im_hsv1 = cv2.merge((cv2.LUT(h1, lut_h1), cv2.LUT(s1, lut_s1), cv2.LUT(v1, lut_v1)))
    cv2.cvtColor(im_hsv1, cv2.COLOR_HSV2BGR, dst=clean_image)  # no return needed


def resize(image,clean_image, input_size, augment):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(input_size / shape[0], input_size / shape[1])
    if not augment:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2
    choice = resample()
    if shape[::-1] != pad:  # resize
        image = cv2.resize(image,
                           dsize=pad,
                           interpolation=choice if augment else cv2.INTER_LINEAR)
        clean_image = cv2.resize(clean_image,
                   dsize=pad,
                   interpolation=choice if augment else cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    clean_image = cv2.copyMakeBorder(clean_image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border

    return image,clean_image, (r, r), (w, h)


def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


def random_perspective(samples,clean_samples, targets, params, border=(0, 0)):
    h = samples.shape[0] + border[0] * 2
    w = samples.shape[1] + border[1] * 2

    # Center
    center = np.eye(3)
    center[0, 2] = -samples.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -samples.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = np.eye(3)

    # Rotation and Scale
    rotate = np.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = np.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation
    translate = np.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():  # image changed
        samples = cv2.warpAffine(samples, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))
        clean_samples = cv2.warpAffine(clean_samples, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))
    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

        # filter candidates
        indices = candidates(box1=targets[:, 1:5].T * s, box2=new.T)
        targets = targets[indices]
        targets[:, 1:5] = new[indices]

    return samples,clean_samples, targets



class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as album

            transforms = [album.Blur(p=0.01),
                          album.CLAHE(p=0.01),
                          album.ToGray(p=0.01),
                          album.MedianBlur(p=0.01)]
            self.transform = album.Compose(transforms,
                                           album.BboxParams('yolo', ['class_labels']),
                                           additional_targets={'clean_image': 'image'})

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image,clean_image, label):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=label[:, 1:],
                               class_labels=label[:, 0],
                              clean_image=clean_image)
            image = x['image']
            clean_image = x['clean_image']

            label = np.array([[c, *b] for c, b in zip(x['class_labels'], x['bboxes'])])
        return image,clean_image, label
    
def collateFunction(batch):
    images, images_clean, targets, shapes = zip(*batch)
    for i, item in enumerate(targets):
        item[:, 0] = i  # add target image index
    return torch.stack(images), torch.stack(images_clean), torch.cat(targets),shapes