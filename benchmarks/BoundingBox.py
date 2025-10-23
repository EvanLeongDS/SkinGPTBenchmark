from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


import pydantic_ai._utils as _utils
from pydantic_ai._utils import number_to_datetime as _orig_ndt
import pydantic_ai.models.openai as _openai_mod
import datetime

import sys
import io
from PIL import Image, ImageDraw
from pathlib import Path

def _safe_number_to_datetime(x):
    if x is None:
        return datetime.datetime.now()
    return _orig_ndt(x)
_utils.number_to_datetime = _safe_number_to_datetime

# patch the copy that openai.py pulled in at import time
_openai_mod.number_to_datetime = _safe_number_to_datetime

# get dataset 
yolobox_images = Path("C:/Users/evanl/MultiMedEval-clean/benchmarks/data/yolobox/merged/images")
yolobox_labels = Path("C:/Users/evanl/MultiMedEval-clean/benchmarks/data/yolobox/merged/labels")
sys.path.append(yolobox_images)

# get model
MODEL_NAME = "lingshu-7b"
# Provider tells pydantic_ai how to connect to LM Studio:
# "http://127.0.0.1:1234"
provider = OpenAIProvider(
    base_url= 'http://localhost:1234/v1',  
    api_key= 'ollama',                      
)
ollama_model = OpenAIModel(
    model_name=MODEL_NAME,
    provider=provider,
)
class Box(BaseModel):
    # x1 and y1 are the top-left corner coordinates
    # x2 and y2 are the bottom-right corner coordinates
    x1: float
    y1: float
    x2: float
    y2: float
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def iou(self, other):
        # Evaluation function for bounding boxes 
        # Calculate intersection over union with another box
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        union_area = self.area() + other.area() - intersection_area

        return intersection_area / (union_area + 1e-6) # avoid division by zero

def load_all_pairs():
    pairs = []
    for img_path in yolobox_images.glob("*.jpg"):
        label_path = yolobox_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        lines = label_path.read_text().splitlines()
        if not lines:
            continue
        image = Image.open(img_path)
        box = label_to_box(lines[0])
        if box:
            pairs.append((img_path.name, image, box))
    return pairs

def label_to_box(line: str) -> Box:
    parts = line.strip().split()
    if len(parts) < 10:
        return None  # invalid format

    # Extract the 8 numeric coordinates
    coords = list(map(float, parts[:8]))
    x_coords = coords[::2]
    y_coords = coords[1::2]

    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)

    return Box(x1=x1, y1=y1, x2=x2, y2=y2)

# Run through pydantic agent 
def predict_box(agent: Agent, image: Image.Image) -> Box:
    """Helper to convert PIL→BinaryContent→model→Box."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_part = BinaryContent(data=buf.getvalue(), media_type="image/jpg")
    result = agent.run_sync([
        "Find the location of the lesion and extract the dimensions of the box.",
        img_part
    ])
    return result.output

def run():
        agent = Agent(ollama_model, 
                    output_type=Box,
                    retries = 1,
                    output_retries= 1,
                    instructions=(
                        "You are a bounding-box generator.  For every image, respond "
                        "by *calling* the output tool exactly once, like:\n"
                        "__tool__[output]({\"x1\":<num>,\"y1\":<num>,\"x2\":<num>,\"y2\":<num>})__end__\n"
                        "Do *not* include any other text."
                    ),
                    )

        all_pairs = load_all_pairs()
        ious = [] # get the ious into a big list and then take the average of the entire list 
        count = len(all_pairs)
        print(f"Count: {count}")
        for idx, (name, image, truth) in enumerate(all_pairs):
            if idx == 21:
                print(f"⏩ Skipping image #{idx} ({name})")
                count -= 1
                continue
            pred = predict_box(agent, image)
            iou = pred.iou(truth)
            print(f"{name}: IoU={iou:.4f}")
            ious.append(iou)

            count -= 1
            print(f"Count: {count}") # print the count 

        avg_iou = sum(ious) / len(ious) if ious else 0.0
        print(f"\nProcessed {len(ious)} images, mean IoU = {avg_iou:.4f}")
        return {
            "name": "BBox Evaluation",
            "model": MODEL_NAME,
            "images": len(ious),
            "mean_iou": round(avg_iou, 4)
        }
if __name__ == "__main__":
    # Prepare model & agent once:
    run()
