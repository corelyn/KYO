import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# =========================
# CONFIG
# =========================
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30000
NOISE_DIM = 100
LR = 0.0002
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_FOLDER = "images"
CAPTIONS_FILE = "captions.json"
OUTPUT_FOLDER = "outputs"
MODEL_FOLDER = "models"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# =========================
# DATASET
# =========================
class GraffitiDatasetWithCaptions(Dataset):
    def __init__(self, image_folder, captions_file, transform=None):
        self.transform = transform
        self.image_folder = image_folder
        self.data = []

        with open(captions_file, "r", encoding="utf-8") as f:
            captions = json.load(f)
            for item in captions:
                filename = os.path.basename(item["filepath"])
                self.data.append({
                    "filename": filename,
                    "caption": item["desc"]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = os.path.join(self.image_folder, item["filename"])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        caption = item["caption"]
        return img, caption

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = GraffitiDatasetWithCaptions(IMAGE_FOLDER, CAPTIONS_FILE, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# MODELS
# =========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x).view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# =========================
# OPTIMIZERS AND LOSS
# =========================
opt_G = torch.optim.Adam(G.parameters(), lr=LR)
opt_D = torch.optim.Adam(D.parameters(), lr=LR)
loss_fn = nn.BCELoss()

# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    for real_imgs, captions in loader:
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        # ---------------------
        # Train Discriminator
        # ---------------------
        noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
        fake_imgs = G(noise)

        real_labels = torch.ones(batch_size, 1).to(DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

        loss_real = loss_fn(D(real_imgs), real_labels)
        loss_fake = loss_fn(D(fake_imgs.detach()), fake_labels)
        loss_D = loss_real + loss_fake

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        loss_G = loss_fn(D(fake_imgs), real_labels)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

    # Save sample generated images every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            test_noise = torch.randn(16, NOISE_DIM).to(DEVICE)
            samples = G(test_noise)
            save_image(samples, os.path.join(OUTPUT_FOLDER, f"epoch_{epoch+1}.png"), nrow=4, normalize=True)

    # Save model every 50 epochs
    if (epoch + 1) % 50 == 0:
        torch.save(G.state_dict(), os.path.join(MODEL_FOLDER, f"generator_epoch_{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(MODEL_FOLDER, f"discriminator_epoch_{epoch+1}.pth"))
        print(f"Saved models at epoch {epoch+1}")

# Save final models
torch.save(G.state_dict(), os.path.join(MODEL_FOLDER, "generator_final.pth"))
torch.save(D.state_dict(), os.path.join(MODEL_FOLDER, "discriminator_final.pth"))
print("Training complete! Generated images and models saved.")
