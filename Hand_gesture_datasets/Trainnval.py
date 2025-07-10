
import os                             
import torch                          
import torchvision                    
import torchvision.transforms as transforms  
from torch.utils.data import DataLoader     
import torch.nn as nn                
import torch.optim as optim          


DATASET_DIR = 'Hand_gesture_datasets'         
MODEL_SAVE_PATH = 'model/gesture_model.pth'   
BATCH_SIZE = 32                                
NUM_EPOCHS = 10                                
IMAGE_SIZE = 64                                

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),   
    transforms.ToTensor(),                        
    transforms.Normalize([0.5], [0.5])             
])


dataset = torchvision.datasets.ImageFolder(root=DATASET_DIR, transform=transform) 

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

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

num_classes = len(dataset.classes)
model = GestureCNN(num_classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device) 

criterion = nn.CrossEntropyLoss()                 
optimizer = optim.Adam(model.parameters(), lr=0.001)  


for epoch in range(NUM_EPOCHS):
    model.train()        
    total_loss = 0.0     

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)              
        loss = criterion(outputs, labels)    

        optimizer.zero_grad()  
        loss.backward()      
        optimizer.step()   

        total_loss += loss.item() 

    avg_loss = total_loss / len(train_loader)  
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

   
    model.eval()  
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)       
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%\n")


os.makedirs('model', exist_ok=True)                  
torch.save(model.state_dict(), MODEL_SAVE_PATH)      
print(f"Model saved to {MODEL_SAVE_PATH}")
