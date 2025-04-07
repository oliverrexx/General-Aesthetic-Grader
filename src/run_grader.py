import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import fiftyone as fo



# Load model and preprocessor
model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model = model.to(torch.bfloat16).cuda() if torch.cuda.is_available() else model.cpu()

# Image folder
image_dir = "data/images"
image_paths = list(Path(image_dir).glob("*.[jJ][pP][gG]")) + \
              list(Path(image_dir).glob("*.[jJ][pP][eE][gG]")) + \
              list(Path(image_dir).glob("*.[pP][nN][gG]"))


samples = []

print("Scoring images...")
for path in tqdm(image_paths):
    try:
        image = Image.open(path).convert("RGB")
        pixel_values = preprocessor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(torch.bfloat16) if torch.cuda.is_available() else pixel_values

        with torch.inference_mode():
            score = model(pixel_values).logits.squeeze().float().cpu().item()

        sample = fo.Sample(filepath=str(path))
        sample["aesthetic_score"] = round(score, 2)
        samples.append(sample)

    except Exception as e:
        print(f"Failed: {path} – {e}")

# Delete existing dataset if it exists
dataset_name = "aesthetic_grader_v2_5"
if dataset_name in fo.list_datasets():
    fo.delete_dataset(dataset_name)

# Create dataset and add samples
dataset = fo.Dataset(dataset_name)
dataset.add_samples(samples)


# Launch FiftyOne App
session = fo.launch_app(dataset)
session.view = dataset.sort_by("aesthetic_score", reverse=True)

# Show score under thumbnails
dataset.app_config.media_fields = ["aesthetic_score"]
dataset.save()
session.refresh()

# Keep session running
input("Fiftone session running: Press ENTER To End")
