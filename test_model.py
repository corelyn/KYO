import torch
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------
# CONFIG
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_DIM = 100
IMAGE_SIZE = 64   # Must match your GAN training
NUM_SAMPLES = 16  # How many graffiti images to generate
GENERATOR_PATH = "models/generator_final.pth"
OUTPUT_IMAGE = "generated_graffiti.png"

# ------------------------
# Generator Definition (must match training)
# ------------------------
import torch.nn as nn

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

# ------------------------
# Load trained generator
# ------------------------
G = Generator().to(DEVICE)
G.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))
G.eval()

# ------------------------
# Generate graffiti images
# ------------------------
noise = torch.randn(NUM_SAMPLES, NOISE_DIM).to(DEVICE)
with torch.no_grad():
    fake_images = G(noise)

# ------------------------
# Save grid of images
# ------------------------
save_image(fake_images, OUTPUT_IMAGE, nrow=4, normalize=True)
print(f"Generated graffiti images saved to {OUTPUT_IMAGE}")

# ------------------------
# Display one example
# ------------------------
img = fake_images[0].cpu() * 0.5 + 0.5  # unnormalize
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title("Example Generated Graffiti")
plt.show()
