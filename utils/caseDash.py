#!/usr/bin/env python3
"""
Combine all PNG images in the current directory horizontally.

Before combining:
- All images are resized to the same height.
- Aspect ratio is preserved.

Output:
    <current_folder>_YYYYMMDDhhmm.png

Requirements:
    pip install pillow
"""

from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps
import re

# Current directory
cwd = Path.cwd()
folder_name = cwd.name

# Exclude previously generated combined images:
# <folder_name>_YYYYMMDDhhmm.png
pattern = re.compile(
    rf"^{re.escape(folder_name)}_\d{{12}}\.png$",
    re.IGNORECASE
)

png_files = sorted(
    p for p in cwd.iterdir()
    if (
        p.is_file()
        and p.suffix.lower() == ".png"
        and not pattern.match(p.name)
    )
)

if not png_files:
    raise SystemExit("No PNG files found.")

# Load images
images = [Image.open(f).convert("RGBA") for f in png_files]

# Target height = maximum image height
target_height = max(img.height for img in images)

# Resize while keeping aspect ratio and add border
resized = []

for img in images:
    if img.height != target_height:
        scale = target_height / img.height
        new_width = int(img.width * scale)

        img = img.resize(
            (new_width, target_height),
            Image.Resampling.LANCZOS
        )

    # Add 1-pixel black border
    img = ImageOps.expand(
        img,
        border=1,
        fill="black"
    )
    resized.append(img)

# Combined image dimensions
total_width = sum(img.width for img in resized)
max_height = max(img.height for img in resized)

combined = Image.new(
    "RGBA",
    (total_width, max_height),
    (255, 255, 255, 255)
)

# Paste side-by-side
x = 0
for img in resized:
    combined.paste(img, (x, 0))
    x += img.width

# Output filename
folder_name = cwd.name
timestamp = datetime.now().strftime("%Y%m%d%H%M")

output_file = cwd / f"{folder_name}_{timestamp}.png"

combined.save(output_file)

print(f"Created: {output_file}")