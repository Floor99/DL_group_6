### Import Packages ###
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models 
import torch.nn as nn
from torcheval.metrics.functional import binary_f1_score



### Make initial model ###
class ImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 256x256x3 -> 128x128x32 (halved because of maxpooling layer)
            nn.Conv2d(3, 32, kernel_size= (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),  
            
            # 128x128x32 -> 64x64x64 (halved because of maxpooling layer)
            nn.Conv2d(32, 64, kernel_size= (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),  
            
            # 64x64x64 -> 32x32x128 (halved because of maxpooling layer)
            nn.Conv2d(64, 128, kernel_size= (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),  
            
            nn.Flatten(),
            nn.Linear(131072, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid())
    
    def forward(self, xb):
        return self.network(xb)

if __name__ == '__main__':

    model = ImageClassification()
    model.load_state_dict(torch.load("data/results/model/initial_CNN_model.pth"))
    model.eval()

    correct = 0
    total = 0

    BATCH_SIZE = 16

    mean = torch.load("data/mean_std/mean.pt")
    variance = torch.load("data/mean_std/variance.pt")
    std = torch.load("data/mean_std/std.pt")

    transformers = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    test_data = ImageFolder(root='data/test', transform=transformers)  
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    for images, labels in test_dataloader:
        labels = labels.view(-1, 1).float()
        pred_labels = model(images)
        predicted = (pred_labels >= 0.5).int()
        total += labels.size(0)
        correct += (predicted.flatten() == labels.flatten()).sum().item()
        
    accuracy = 100 * correct/ total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
    ########## F1 #############
    all_preds = []
    all_labels = []

    for images, labels in test_dataloader:
        labels = labels.view(-1, 1).float()
        pred_labels = model(images)
        predicted = (pred_labels >= 0.5).int()
        total += labels.size(0)
        correct += (predicted.flatten() == labels.flatten()).sum().item()
        all_preds.append(predicted)
        all_labels.append(labels)
        
    all_preds = torch.cat(all_preds).view(-1)
    all_labels = torch.cat(all_labels).view(-1).int()  

    f1 = binary_f1_score(all_preds, all_labels)
    print(f"Binary F1_Score: {f1}")