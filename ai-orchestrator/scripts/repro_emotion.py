
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
from torchvision import models
from huggingface_hub import hf_hub_download
import torch.nn.functional as F

# Config
FACE_EMOTION_MODEL_ID = "20-team-daeng-ddang-ai/dog-emotion-classification"
IMAGE_URL = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=2874&auto=format&fit=crop" # Generic dog image

def test_model():
    print("1. Loading Processor...")
    try:
        processor = AutoImageProcessor.from_pretrained(FACE_EMOTION_MODEL_ID)
    except Exception:
        print(f"Fallback to google/efficientnet-b0")
        processor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
        processor.do_normalize = True
        processor.image_mean = [0.485, 0.456, 0.406]
        processor.image_std = [0.229, 0.224, 0.225]
        processor.size = {"height": 224, "width": 224}

    print(f"Processor normalized: {processor.do_normalize}")
    print(f"Processor mean: {processor.image_mean}")
    print(f"Processor std: {processor.image_std}")

    print("2. Loading Model...")
    classifier = models.efficientnet_b0(pretrained=False)
    num_features = classifier.classifier[1].in_features
    classifier.classifier[1] = torch.nn.Linear(num_features, 4)
    
    weight_path = hf_hub_download(repo_id=FACE_EMOTION_MODEL_ID, filename="model.safetensors")
    from safetensors.torch import load_file
    state_dict = load_file(weight_path)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    
    print("3. Inference...")
    # Download image manually or use a local one. Using a random noise or simple array for shape check if no url.
    # Actually let's just make a dummy image (random) to check mapping consistency if possible?
    # No, we need a real image to see if it predicts Happy/Sad correctly. 
    # Let's use the actual frame extraction logic on a temp file? 
    # Or just use a known "Happy" dog image url.
    
    import requests
    img = Image.open(requests.get(IMAGE_URL, stream=True).raw).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    input_tensor = inputs['pixel_values']
    
    with torch.no_grad():
        logits = classifier(input_tensor)
        probs = F.softmax(logits, dim=-1)[0]
        
    print(f"Logits: {logits}")
    print(f"Probs: {probs}")
    
    id2label = {0: "happy", 1: "sad", 2: "angry", 3: "relaxed"}
    for i, p in enumerate(probs):
        print(f"{id2label[i]}: {p:.4f}")

if __name__ == "__main__":
    test_model()
