import torch 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dl.earlystopping import EarlyStopping


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


### Train and validate model ###

def train_model(model, patience, n_epochs):
    model = model.to(device)
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []
    
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    model.train()    
     
    for epoch in range(1, n_epochs + 1):
        ####### train the model #######
        train_loss, correct_train, total_train = 0, 0, 0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            labels = labels.view(-1, 1).float()
            pred_labels = model(images)
            loss = F.binary_cross_entropy(pred_labels, labels)
            loss.backward()
            optimizer.step() 
            
            train_loss += loss.item() * images.size(0)
            predicted = (pred_labels >= 0.5).int()
            total_train += labels.size(0)
            correct_train += (predicted.flatten() == labels.flatten()).sum().item()
        
        train_loss /= len(train_dataloader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
   
            
        ####### validate the model #######
        validation_loss, correct_val, total_val = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in validation_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.view(-1, 1).float()
                pred_labels = model(images)
                loss = F.binary_cross_entropy(pred_labels, labels)
                
                validation_loss += loss.item() * images.size(0)
                predicted = (pred_labels >= 0.5).int()
                total_val += labels.size(0)
                correct_val += (predicted.flatten() == labels.flatten()).sum().item()
                  
        validation_loss /= len(validation_dataloader)
        validation_accuracy = 100 * correct_val/ total_val
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        
        early_stopping(validation_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Print metrics
        print(f'Epoch [{epoch}/{n_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, \
            Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%')
    
    with open('data/results/train_losses_initial_cnn.txt', 'w') as f: 
        for line in train_losses:
            f.write(f"{line}\n") 
            
    with open('data/results/train_accuracies_initial_cnn.txt', 'w') as f: 
        for line in train_accuracies:
            f.write(f"{line}\n")
    
    with open('data/results/validation_losses_initial_cnn.txt', 'w') as f: 
        for line in validation_losses:
            f.write(f"{line}\n")
    
    with open('data/results/validation_accuracies_initial_cnn.txt', 'w') as f: 
        for line in validation_accuracies:
            f.write(f"{line}\n")
        
    
    return model, train_losses, validation_losses, train_accuracies, validation_accuracies



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
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
    
    training_data = ImageFolder(root='data/train', transform=transformers)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    validation_data = ImageFolder(root='data/validation', transform=transformers)  
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    test_data = ImageFolder(root='data/validation', transform=transformers)  
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    initial_CNN_model = ImageClassification()
    optimizer = optim.Adam(initial_CNN_model.parameters(), lr = 0.001)
    
    initial_CNN_model, train_losses, validation_losses, train_accuracies, validation_accuracies  = train_model(initial_CNN_model, 5, 60)
    
    # Save the trained model
    model_path = 'data/results/model/initial_CNN_model.pth'
    torch.save(initial_CNN_model.state_dict(), model_path)
    print(f"Model saved at {model_path}")  