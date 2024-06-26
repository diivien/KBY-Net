import gradio as gr
import torch
import cv2
import numpy as np
from model.utils import scale, non_max_suppression
from model.kbynet import YOLO, DarkNet, DarkFPN, CSP, Residual, Head, SPP, Conv, KBDecoder, DFL
from model.blocks import KBAFunction, KBBlock_l, KBBlock_s, MFF, TransAttention, SimpleGate
from model.kbynet import YOLO2, DarkNet2, DarkFPN2

# Load your models
model = torch.load('6-4.pt', map_location='cuda')
model = model['model'].half().eval().cuda()

clean_model = torch.load('clean_best.pt', map_location='cuda')
clean_model = clean_model['model'].half().eval().cuda()

# Define class names and colors
class_names = ['person', 'car', 'bicycle', 'motorcycle', 'bus']
colors = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]

# Define input size
input_size = 640

# Keep the load_image, resize, and preprocess_image functions as they are
def load_image(input):
    if isinstance(input, np.ndarray):
        image = input
    else:
        image = cv2.imread(input)
    h, w = image.shape[:2]
    r = input_size / max(h, w)
    image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    return image, (h, w)

def resize(image, input_size):
    shape = image.shape[:2]  # current shape [height, width]
    r = min(input_size / shape[0], input_size / shape[1])
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2
    if shape[::-1] != pad:
        image = cv2.resize(image, dsize=pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image, (r, r), (w, h)

def preprocess_image(input):
    image, shape = load_image(input)
    image, ratio, pad = resize(image, input_size)
    shapes = shape, ((shape[0] / shape[0], shape[1] / shape[1]), pad)
    sample = image.transpose((2, 0, 1))[::-1]
    sample = np.ascontiguousarray(sample)
    return torch.from_numpy(sample).float().div(255.0).unsqueeze(0).half(), shapes
def draw_labels(boxes, colors, class_ids, classes, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = classes[int(class_ids[i])]
        color = colors[int(class_ids[i])]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - text_height - 8), (x1 + text_width + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img

def process_detections(det_outputs, input_tensor, shapes):
    boxes = []
    class_ids = []
    for output in det_outputs:
        detections = output.clone()
        scale(detections[:, :4], input_tensor.shape[2:], shapes[0])
        
        if output.shape[0] > 0:
            for box in detections.cpu().numpy():
                x1, y1, x2, y2, class_conf, class_pred = box
                if class_pred < len(class_names):
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(int(class_pred))
    return boxes, class_ids

def inference(image):
    # Preprocess image
    input_tensor, shapes = preprocess_image(image)
    
    # Run inference on both models
    with torch.no_grad():
        outputs = model(input_tensor.cuda())
        clean_outputs = clean_model(input_tensor.cuda())
    
    # Process outputs
    restoration = outputs['Restoration'].float().cpu().numpy()[0]
    det_outputs = non_max_suppression(outputs['Detection'][1], 0.5, 0.7)
    clean_det_outputs = non_max_suppression(clean_outputs[1], 0.5, 0.7)
    
    # Convert restoration output to uint8 image
    h, w = image.shape[:2]
    pad_h, pad_w = shapes[1][1]
    pad_h, pad_w = int(pad_h), int(pad_w)
    restored_img = restoration[:, pad_w:input_size-pad_w, pad_h:input_size-pad_h]
    restored_img = cv2.resize(restored_img.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    restored_img = np.clip(restored_img, 0, 1)
    restored_img = (restored_img * 255).astype(np.uint8)
    
    # Process detections for both models
    boxes, class_ids = process_detections(det_outputs, input_tensor, shapes)
    clean_boxes, clean_class_ids = process_detections(clean_det_outputs, input_tensor, shapes)
    
    # Draw labels on the restored image for main model
    main_result_img = draw_labels(boxes, colors, class_ids, class_names, restored_img.copy())
    
    # Draw labels on the original image for clean model
    clean_result_img = draw_labels(clean_boxes, colors, clean_class_ids, class_names, image.copy())
    
    # Convert back to RGB for display
    main_result_img = cv2.cvtColor(main_result_img, cv2.COLOR_BGR2RGB)
    clean_result_img = cv2.cvtColor(clean_result_img, cv2.COLOR_BGR2RGB)
    
    return main_result_img, clean_result_img

# Create Gradio interface
iface = gr.Interface(
    fn=inference,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(label="Main Model (with restoration)"),
        gr.Image(label="Clean Model (original image)")
    ],
    title="KBY-Net vs YOLOv8* Comparison",
    description="Upload an image to perform object detection and image restoration. Shows detections from both main and clean models."
)

# Launch the app
iface.launch()