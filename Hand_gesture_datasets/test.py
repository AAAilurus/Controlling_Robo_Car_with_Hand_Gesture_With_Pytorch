import torch
import torchvision.transforms as transforms
from PIL import Image
import os

DATASET_DIR = 'Hand_gesture_datasets'
MODEL_SAVE_PATH = 'model/gesture_model.pth'
IMAGE_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

import torch.nn as nn

class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

dataset_classes = sorted(os.listdir(DATASET_DIR))

num_classes = len(dataset_classes)

model = GestureCNN(num_classes)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')  
    img_tensor = transform(image)                  
    img_tensor = img_tensor.unsqueeze(0)         

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        return dataset_classes[class_idx]

test_image_path = os.path.join(DATASET_DIR, 'right', 'right_15.jpg') 
prediction = predict_image(test_image_path)
print(f'Predicted Gesture: {prediction}')
