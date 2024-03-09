import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), 1.0).view(-1, self.num_classes, 1, 1)
        gen_input = torch.cat([noise, gen_input.repeat(1, 1, noise.size(2), noise.size(3))], 1)
        return self.model(gen_input)

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Conv2d(1 + num_classes, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1 + num_classes, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        disc_input = torch.mul(self.label_emb(labels), 1.0).view(-1, self.num_classes, 1, 1).repeat(1, 1, images.size(2), images.size(3))
        disc_input = torch.cat([images, disc_input], 1)
        return self.model(disc_input).chunk(2, dim=1)

# Load the dataset
data_transforms = Compose([Resize((64, 64)), ToTensor(), Normalize([0.5], [0.5])])
dataset = ImageFolder('path/to/chest_xray/train', transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Initialize the generator and discriminator
latent_dim = 100
num_classes = 2
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# Define the loss functions
adversarial_loss = nn.BCELoss()
auxiliary_loss = nn.CrossEntropyLoss()

# Define the optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Generate fake images and labels
        batch_size = images.size(0)
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_labels = torch.randint(0, 2, (batch_size,), device=device)
        fake_images = generator(noise, fake_labels)

        # Train the discriminator
        disc_optimizer.zero_grad()
        real_outputs, real_aux = discriminator(images, labels)
        fake_outputs, fake_aux = discriminator(fake_images.detach(), fake_labels)
        disc_real_loss = adversarial_loss(real_outputs, torch.ones_like(real_outputs))
        disc_fake_loss = adversarial_loss(fake_outputs, torch.zeros_like(fake_outputs))
        disc_loss = disc_real_loss + disc_fake_loss
        disc_loss += auxiliary_loss(real_aux, labels) + auxiliary_loss(fake_aux, fake_labels)
        disc_loss.backward()
        disc_optimizer.step()

        # Train the generator
        gen_optimizer.zero_grad()
        fake_outputs, fake_aux = discriminator(fake_images, fake_labels)
        gen_loss = adversarial_loss(fake_outputs, torch.ones_like(fake_outputs))
        gen_loss += auxiliary_loss(fake_aux, fake_labels)
        gen_loss.backward()
        gen_optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}] Discriminator Loss: {disc_loss.item():.4f} Generator Loss: {gen_loss.item():.4f}')
