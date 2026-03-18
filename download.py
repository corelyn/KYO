import os
import re
from icrawler.builtin import BingImageCrawler
from PIL import Image

folder = "pre_images"
temp_folder = "temp_download"
num_images = 50

os.makedirs(folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# ---- detect next index automatically ----
existing = os.listdir(folder)
numbers = []

for f in existing:
    m = re.match(r"(\d+)\.png", f)
    if m:
        numbers.append(int(m.group(1)))

start_index = max(numbers) + 1 if numbers else 12
print("Starting at index:", start_index)

# ---- download images ----
crawler = BingImageCrawler(storage={"root_dir": temp_folder})
crawler.crawl(keyword="graffiti", max_num=num_images)

files = sorted(os.listdir(temp_folder))

i = start_index

for f in files:
    try:
        path = os.path.join(temp_folder, f)

        img = Image.open(path).convert("RGB")
        save_path = os.path.join(folder, f"{i}.png")

        img.save(save_path, "PNG")

        print("Saved", save_path)
        i += 1
    except:
        pass
