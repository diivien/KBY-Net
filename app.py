import gradio as gr
import torch
import cv2
import numpy as np
from model.utils import scale, non_max_suppression
from model.kbynet import YOLO, DarkNet, DarkFPN, CSP, Residual, Head, SPP, Conv, KBDecoder, DFL
from model.blocks import KBAFunction, KBBlock_l, KBBlock_s, MFF, TransAttention, SimpleGate
from model.kbynet import YOLO2, DarkNet2, DarkFPN2
import os
import random
class Uncertainty:
    def __init__(self):
        pass

confidence = 0.5
iou = 0.5
# Load your models
model = torch.load('kitti 6-4.pt', map_location='cuda')
model = model['model'].half().eval().cuda()

clean_model = torch.load('kitticlean.pt', map_location='cuda')
clean_model = clean_model['model'].half().eval().cuda()

# Define class names and colors
# class_names = ['person', 'car', 'bicycle', 'motorcycle', 'bus']
class_names = ['car', 'van', 'truck', 'pedestrian', 'sitting', 'cyclist'] 

colors = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan

]
# image_folders = ['rain_test','citydata/test/rainy','rain_train','rain_kitti_split/test/rainy']
# image_folders = ['citydata/test/rainy','rain_kitti_split/test/rainy', 'citydata/train/rainy']

image_folders = ['test_rainy']

# Define input size
input_size = 640
def get_random_image():
    # Randomly choose a folder
    if not image_folders:
        return None
    
    chosen_folder = random.choice(image_folders)
    
    # Get all images from the chosen folder
    images = [f for f in os.listdir(chosen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    if images:
        # Randomly choose an image from the chosen folder
        random_image = random.choice(images)
        return os.path.join(chosen_folder, random_image)
    else:
        return None
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
def draw_labels(boxes, colors, class_ids, classes, confidences, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{classes[int(class_ids[i])]} {confidences[i]:.2f}"
        color = colors[int(class_ids[i])]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - text_height - 8), (x1 + text_width + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img

def process_detections(det_outputs, input_tensor, shapes):
    boxes = []
    class_ids = []
    confidences = []
    for output in det_outputs:
        detections = output.clone()
        scale(detections[:, :4], input_tensor.shape[2:], shapes[0])
        
        if output.shape[0] > 0:
            for box in detections.cpu().numpy():
                x1, y1, x2, y2, class_conf, class_pred = box
                if class_pred < len(class_names):
                    boxes.append([x1, y1, x2, y2])
                    class_ids.append(int(class_pred))
                    confidences.append(class_conf)
    return boxes, class_ids, confidences

def inference(image, confidence_threshold, random_image_path=None):
    if random_image_path:
        image = cv2.imread(random_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    input_tensor, shapes = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(input_tensor.cuda())
        clean_outputs = clean_model(input_tensor.cuda())
    
    restoration = outputs['Restoration'].float().cpu().numpy()[0]
    det_outputs = non_max_suppression(outputs['Detection'][1], confidence_threshold, iou)
    clean_det_outputs = non_max_suppression(clean_outputs[1], confidence_threshold, iou)
    
    h, w = image.shape[:2]
    pad_h, pad_w = shapes[1][1]
    pad_h, pad_w = int(pad_h), int(pad_w)
    restored_img = restoration[:, pad_w:input_size-pad_w, pad_h:input_size-pad_h]
    restored_img = cv2.resize(restored_img.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_LINEAR)
    restored_img = np.clip(restored_img, 0, 1)
    restored_img = (restored_img * 255).astype(np.uint8)
    
    boxes, class_ids, confidences = process_detections(det_outputs, input_tensor, shapes)
    clean_boxes, clean_class_ids, clean_confidences = process_detections(clean_det_outputs, input_tensor, shapes)
    
    main_result_img = draw_labels(boxes, colors, class_ids, class_names, confidences, restored_img.copy())
    clean_result_img = draw_labels(clean_boxes, colors, clean_class_ids, class_names, clean_confidences, cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR))
    
    main_result_img = cv2.cvtColor(main_result_img, cv2.COLOR_BGR2RGB)
    clean_result_img = cv2.cvtColor(clean_result_img, cv2.COLOR_BGR2RGB)
    
    return main_result_img, clean_result_img

custom_css = """
footer {visibility: hidden}
.gr-button {
    background-color: #4CAF50 !important;
    border: none !important;
}
.gr-form {
    flex-grow: 1;
    padding: 20px;
}
/* Adjust the size of the input image container */
.input-image .image-container {
    max-height: 300px !important;
}
.input-image img {
    max-height: 300px !important;
    width: auto !important;
    object-fit: contain !important;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css) as iface:
    gr.Markdown(
        """
        # KBY-Net vs YOLOv8* Comparison
        Upload an image or use the 'I'm Feeling Lucky' button to perform object detection and image restoration. 
        Shows detections from both main and clean models.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            input_image = gr.Image(type="numpy", label="Input Image", elem_classes="input-image")
        with gr.Column(scale=1, min_width=100):
            confidence_slider = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="Confidence Threshold")
            lucky_button = gr.Button("I'm Feeling Lucky", size="sm")
    
    with gr.Row():
        output_image1 = gr.Image(label="Main Model (with restoration)")
        output_image2 = gr.Image(label="Clean Model (original image)")
    
    def update_results(image, confidence_threshold):
        if image is None:
            return None, None
        return inference(image, confidence_threshold)
    
    def lucky_inference(confidence_threshold):
        random_image_path = get_random_image()
        if random_image_path:
            img = cv2.imread(random_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            main_result, clean_result = inference(img, confidence_threshold, random_image_path)
            return img, main_result, clean_result
        else:
            return None, None, None
    
    input_image.change(update_results, inputs=[input_image, confidence_slider], outputs=[output_image1, output_image2])
    confidence_slider.release(update_results, inputs=[input_image, confidence_slider], outputs=[output_image1, output_image2])
    lucky_button.click(lucky_inference, inputs=[confidence_slider], outputs=[input_image, output_image1, output_image2])

# Launch the app
iface.launch()