import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

MODEL_PATH = 'model/gesture_model.pth'
DATASET_DIR = 'Hand_gesture_datasets'
IMAGE_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = sorted(os.listdir(DATASET_DIR))
num_classes = len(classes)

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

model = GestureCNN(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not found. Make sure it's connected.")
    exit()

print("Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        label = classes[predicted.item()]
        confidence_val = confidence.item()

    if confidence_val > 0.70:
        display_text = f"{label} ({confidence_val:.2f})"
    else:
        display_text = "Detecting..."

    cv2.putText(frame, f"Prediction: {display_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
