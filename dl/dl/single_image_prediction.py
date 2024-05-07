### Import Packages ###
import torch 
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dl.make_image_data import create_df_from_all_images, drop_incorrect_images_from_metadata, remove_removals, select_chosen_species, species, removals, split_image_data

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

def preprocess_image(image_path, mean, std):
    # Load image
    image = Image.open(image_path)
    
    # Define the same transformations as used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) 
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0) 
    return image_tensor

    ### Make prediction ###

def predict(image_tensor):
    with torch.no_grad():  
        outputs = model(image_tensor)
        predicted = (outputs >= 0.5).int()
        return "not sick" if predicted.item() == 1 else "sick"

if __name__ == '__main__':

### Load trained model ###

    model = ImageClassification()
    model.load_state_dict(torch.load("data/results/model/initial_CNN_model.pth"))
    model.eval()

    ### Preprocess image ###
    mean = torch.load("data/mean_std/mean.pt")
    variance = torch.load("data/mean_std/variance.pt")
    std = torch.load("data/mean_std/std.pt")


    df = create_df_from_all_images('data/raw/color')
        
    df = select_chosen_species(df, species)

    df = drop_incorrect_images_from_metadata(df, [256,256], 3)

    df = remove_removals(df, removals)
    all_data, train, test, val = split_image_data(df)

    ### Get output ###

    # Diseased image from test set
    random_path_one = test[test['label'] == 'diseased']['source'].sample(n=1).iloc[0]
    image_tensor_one = preprocess_image(random_path_one, mean, std) 
    prediction_one = predict(image_tensor_one)
    print(f"The prediction for the image is: {prediction_one}")

    image = Image.open(random_path_one)
    plt.imshow(image)
    plt.title(f"Prediction: {prediction_one}")
    plt.show()

    # Not diseased image from test set
    random_path_two = test[test['label'] == 'healthy']['source'].sample(n=1).iloc[0]
    image_tensor_two = preprocess_image(random_path_two, mean, std) 
    prediction_two = predict(image_tensor_two)
    print(f"The prediction for the image is: {prediction_two}")

    image = Image.open(random_path_two)
    plt.imshow(image)
    plt.title(f"Prediction: {prediction_two}")
    plt.show()

