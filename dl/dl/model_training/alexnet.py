import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch
import torchvision.models as models 
import torch.nn as nn
from torcheval.metrics.functional import binary_f1_score
import copy
import torch.optim as optim



def train_model(model, criterion, optimizer, num_epochs=10):
    model = model.to(device)
    losses = []
    accuracy = []
    
    best_loss = float('inf')
    best_model_weights = None
    patience = 1
    
    for epoch in range(num_epochs):
        for phase in ['train', 'validation']:
            print(phase)
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                preds = (outputs >= 0.5).int().squeeze()
                
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            
            losses.append(epoch_loss)
            accuracy.append(epoch_acc)
            
        if phase == 'validation':
            print(losses[-1])
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                print(f"{best_loss= }")
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = 5 
            else: 
                patience -= 1
                print(f"{patience= }")
                if patience == 0:
                    break
                       
            
        model.load_state_dict(best_model_weights)
        print(f"Epoch [{epoch+1}/{num_epochs}]")

    with open('data/results/log_loss_alexnet.txt', 'w') as f: 
        for line in losses:
            f.write(f"{line}\n") 
            
    with open('data/results/accuracy_alexnet.txt', 'w') as f: 
        for line in accuracy:
            f.write(f"{line}\n")
    
           


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
    

    training_data = ImageFolder(root='data/train', transform=transformers)
    # training_data = Subset(training_data, list(range(0,500)))
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    validation_data = ImageFolder(root='data/validation', transform=transformers) 
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    dataloaders = {
        "train": train_dataloader,
        "validation": validation_dataloader
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier[6].in_features

    model.classifier[6] = nn.Linear(num_ftrs, 1)
    model.classifier.add_module("7", nn.Sigmoid())

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model=model, criterion=criterion, optimizer=optimizer, num_epochs=60)
    
    # Save the trained model
    model_path = 'data/results/model/alex_net_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")  