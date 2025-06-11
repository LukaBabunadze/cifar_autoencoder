import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from models.autoencoder import Autoencoder
from utils.noise import add_noise
from config import *

transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    for images, _ in trainloader:
        images = images.to(DEVICE)
        noisy = add_noise(images).to(DEVICE)

        output = model(noisy)
        loss = criterion(output, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")


# Save the model after training
torch.save(model.state_dict(), "autoencoder.pth")
print("✅ Model saved as autoencoder.pth")
