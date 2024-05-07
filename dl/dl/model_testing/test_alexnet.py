import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models 
import torch.nn as nn
from torcheval.metrics.functional import binary_f1_score

if __name__ == "__main__":
    BATCH_SIZE = 16
    
    mean = torch.load("data/mean_std/mean.pt")
    variance = torch.load("data/mean_std/variance.pt")
    std = torch.load("data/mean_std/std.pt")
    
    transformers = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])
    

    test_data = ImageFolder(root='data/test', transform=transformers)  
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    ########################### TEST MODEL ################################

    # Load pretrained model
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    # Number of classes and modify last layer
    num_classes = 2 

    # for param in model.parameters():
    #     param.requires_grad = False

    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 1)
    model.classifier.add_module("7", nn.Sigmoid())

    
    model.load_state_dict(torch.load("data/results/model/alex_net_model.pth"))
    model.eval()


    correct = 0
    total = 0

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