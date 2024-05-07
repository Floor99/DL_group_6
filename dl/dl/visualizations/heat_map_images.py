import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from dl.make_image_data import create_df_from_all_images, drop_incorrect_images_from_metadata, remove_removals, select_chosen_species, species, removals, split_image_data

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

class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.feature_maps = None
        self.gradients = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, class_idx=None):
        output = self.model(input_image)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        target = output[:, class_idx]
        target.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.feature_maps.size(1)):
            self.feature_maps[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.feature_maps, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy()
    
def find_last_conv_layer(model):
    # We reverse the layers and find the first convolutional layer encountered.
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return name
    return None
    
if __name__ == '__main__':
    # Load your model
    model = ImageClassification()
    model.load_state_dict(torch.load("data/results/model/initial_CNN_model.pth"))
    model.eval()

    last_conv_layer_name = find_last_conv_layer(model)
    if last_conv_layer_name is not None:
        grad_cam = GradCAM(model, last_conv_layer_name)
        # Proceed with the rest of the Grad-CAM process
    else:
        print("No convolutional layer found in the model!")
        
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
    
    df = create_df_from_all_images('data/raw/color')
    
    df = select_chosen_species(df, species)
    
    df = drop_incorrect_images_from_metadata(df, [256,256], 3)
    
    df = remove_removals(df, removals)
    all_data, train, test, val = split_image_data(df)

    # Diseased image from test set
    random_path_one = test[test['label'] == 'diseased']['source'].sample(n=1).iloc[0]
    image = Image.open(random_path_one)
    input_tensor = transformers(image).unsqueeze(0)

    # Generate heatmap
    heatmap = grad_cam.generate_heatmap(input_tensor)
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    original_img = np.array(image)
    original_img = cv2.resize(original_img, (256, 256))
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()
