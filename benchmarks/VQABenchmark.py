import sys
import os
from io import BytesIO
from base64 import b64encode
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multimedeval import MultiMedEval, SetupParams, EvalParams
from multimedeval.utils import BatcherInput, BatcherOutput
from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
from PIL import Image
import torch
from typing import List
from openai import OpenAI
from tqdm import tqdm 

# Step 1: Initialize the engine from MultiMedEval
engine = MultiMedEval()

# Step 2: Point to the vqa rad directory and establish the model
setupParams = SetupParams(
    vqa_rad_dir="data/vqa_rad"
)
MODEL_NAME = "lingshu-7b"  # Model to use for VQA-Rad

# Step 3: Set up the engine (runs data loading etc.)
engine.setup(setupParams)

# make a simple prompt but good enough for output
my_prompt="""
You are a highly experienced 
Do not offer disclaimers, uncertainties, or probabilities unless absolutely necessary.
Make your answer 1-2 words, nothing more. Just give the diagnosis. 
"""

class batcherVQA:
    def __init__(self) -> None:
        self.count = 451 # VQA-RAD dataset size

    def __call__(self, prompts: List[BatcherInput]) -> List[BatcherOutput]:
        # Change batcher input size to one
        if len(prompts) > 1:
            raise ValueError("Batch size must be one.")
        
        # OpenAI part 
        client = OpenAI(
            base_url = 'http://localhost:1234/v1',
            api_key='ollama', # required, but unused
        )
        results = []
        print(f"Running {MODEL_NAME} model for VQA-Rad dataset")
        for prompt in prompts:
            # print(f"conversation: {prompt.conversation}")  find the conversation structure
            # print(f"image: {prompt.images}")  find the image structure 
            print(f"Processing sample #{self.count}")  # for debugging
            self.count -= 1
            # Image conversion to base64
            if not prompt.images:   
                raise ValueError("No images provided in the prompt.")
            image = prompt.images[0]
            if not isinstance(image, Image.Image):
                raise ValueError("Image must be a PIL Image object.")
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = b64encode(buffered.getvalue()).decode('utf-8')

            response = client.chat.completions.create(
                model= MODEL_NAME,
                messages=[
                    {"role": "system", "content": my_prompt},
                    # prompt.conversation[0],  # the first message in the conversation
                    # prompt.images[0],  # the first image in the images

                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt.conversation[0]["content"], # ask question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"  # base64 encoded image
                            }
                        }
                    ]}
                ]
            )
            print(response.choices[0].message.content)
            results.append(BatcherOutput(text=response.choices[0].message.content))
        return results
    
# Step 4: Create the medgemma batcher
#def batcher(prompts: list[BatcherInput]) -> list[BatcherOutput]:
 #  return [BatcherOutput("mel") for _ in prompts]  # dummy answer for testing

# Step 5: Run evaluation
batcher = batcherVQA()
results = engine.eval(["VQA-Rad"], batcher, EvalParams(batch_size = 1))

# Step 6: Show the results
import json
def run():
    print(json.dumps(results, indent=2))
    return json.dumps(results, indent=2)

if __name__ == "__main__":
    run()

