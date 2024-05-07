from torch.utils.data import DataLoader
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def calculate_mean_and_variance(loader):
    mean = 0.0
    variance = 0.0
    total_images = 0

    for images, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        # Update total_images
        total_images += images.size(0)
        # Compute mean and variance here
        mean += images.float().mean(2).sum(0) 
        variance += images.float().var(2).sum(0)

    # Final mean and variance
    mean /= total_images
    variance /= total_images

    return mean, variance

if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    training_data = ImageFolder(root='data/train', transform=train_transform)
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

    print("calculating mean and variance")
    mean, variance = calculate_mean_and_variance(train_dataloader)
    std = variance.sqrt()
    print("Done calculating mean and variance")

    torch.save(mean, 'data/mean_std/mean.pt')
    torch.save(variance, 'data/mean_std/variance.pt')
    torch.save(std, 'data/mean_std/std.pt')