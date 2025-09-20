import os, random
import torch
from torchvision import transforms
from PIL import Image

# Ripeness mapping
ripeness_to_days = {
    "Unripe": 7.0,
    "PartiallyRipe": 5.0,
    "Ripe": 3.0,
    "Overripe": 1.0,
    "Rotten": 0.0
}

# Preprocessing (same as validation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_days(model, image, device):
    """Predict days to rotten for a single image"""
    tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    return output.item()

def predict_random_image(model, device, root_dir="./banana_data/Banana Ripeness Classification Dataset"):
    """Pick a random dataset image and return prediction + ground truth"""
    split = random.choice(["train", "test", "valid"])
    class_dir = os.path.join(root_dir, split)
    class_name = random.choice(os.listdir(class_dir))
    img_dir = os.path.join(class_dir, class_name)
    img_name = random.choice(os.listdir(img_dir))
    img_path = os.path.join(img_dir, img_name)

    image = Image.open(img_path).convert("RGB")
    pred_days = predict_days(model, image, device)
    true_days = ripeness_to_days.get(class_name, None)

    return image, pred_days, true_days, class_name
