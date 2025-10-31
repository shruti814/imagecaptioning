import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load models
seg_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
seg_model.eval()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def image_segmentation(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    with torch.no_grad():
        predictions = seg_model([img_tensor])[0]
    
    # Draw masks
    masks = predictions['masks']
    boxes = predictions['boxes']
    labels = predictions['labels']
    scores = predictions['scores']
    
    np_image = np.array(image)
    for i in range(len(masks)):
        if scores[i] > 0.8:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            np_image[mask > 128] = np_image[mask > 128] * 0.5 + color * 0.5
            box = boxes[i].cpu().numpy().astype(int)
            cv2.rectangle(np_image, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)
    return Image.fromarray(np_image)

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = caption_model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def process_image(image_path):
    segmented_image = image_segmentation(image_path)
    caption = generate_caption(image_path)
    return segmented_image, caption
