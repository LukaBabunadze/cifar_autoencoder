import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from utils.noise import add_noise
from models.autoencoder import Autoencoder
from config import *
import torch

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

model = Autoencoder().to(DEVICE)
model.load_state_dict(torch.load("autoencoder.pth"))  # Load saved model
model.eval()

imgs, _ = next(iter(testloader))
noisy_imgs = add_noise(imgs)
with torch.no_grad():
    outputs = model(noisy_imgs.to(DEVICE))

def show(orig, noisy, output):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(orig.permute(1, 2, 0))
    axs[1].imshow(noisy.permute(1, 2, 0))
    axs[2].imshow(output.permute(1, 2, 0).cpu())
    for ax in axs: ax.axis('off')
    plt.show()

show(imgs[0], noisy_imgs[0], outputs[0])
