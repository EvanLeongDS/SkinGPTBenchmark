import random
from io import BytesIO
from base64 import b64encode
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import List
import requests
import traceback

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Establish model and judge here
MODEL_NAME = "lingshu-7b"
OLLAMA_BASE_URL = "http://localhost:1234/v1"
JUDGE = "qwen/qwen2.5-vl-7b"  # Qwen judge 
COUNT = 1  # samples to evaluate (vqa has 451)

# setup
provider = OpenAIProvider(base_url=OLLAMA_BASE_URL, api_key="ollama")
model = OpenAIModel(model_name=MODEL_NAME, provider=provider)
qwen_judge = OpenAIModel(model_name = JUDGE, provider=provider)

# Define the judge 
judge = Agent(
    qwen_judge,
    output_type=str,
    instructions=(
        "You are a hallucination detector. Given a medical question and an image-based answer, "
        "respond with either 'truthful' or 'hallucinated'. Only respond with one word."
    ),
    retries=1,
    output_retries=1,
)

# Preprocess dataset
dataset = load_dataset("flaviagiammarino/vqa-rad")["train"]

def normalize_label(label: str) -> str:
    return {
        "truthful": "truthful",
        "hallucinated": "hallucinated",
        "yes": "hallucinated",
        "no": "truthful"
    }.get(label.strip().lower(), "")

def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("utf-8")

# run data through image and question to get model answer
def generate_model_answer(image: Image.Image, question: str) -> str:
    img_base64 = image_to_base64(image)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": (
                "You are a helpful, concise, and medically accurate vision-language assistant. "
                "Answer the user's medical question based only on the provided image."
            )},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "text", "text": question}
            ]}
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{OLLAMA_BASE_URL}/chat/completions", headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()

# big function to benchmark the model
def benchmark_model(dataset, count=50):
    results = []
    for idx, example in enumerate(random.sample(list(dataset), k=count)):
        question = example["question"]
        image = example["image"]
        print(f"\nCount: {count - idx - 1}")

        try:
            model_answer = generate_model_answer(image, question)
            judgment = judge.run_sync([f"Question: {question}", f"Answer: {model_answer}"]).output
            print(f"Q: {question}\nA: {model_answer}\nJudgment: {judgment}")
            results.append({
                "question": question,
                "model_answer": model_answer,
                "judgment": normalize_label(judgment)
            })
        except Exception:
            traceback.print_exc()
            continue

    return results

# get results and print them out
def summarize(results):
    # summarize results
    y_pred = [r["judgment"] for r in results]

    report = classification_report(
        y_pred, y_pred,
        labels=["truthful", "hallucinated"],
        output_dict=True
    )
    cm = confusion_matrix(y_pred, y_pred, labels=["truthful", "hallucinated"])
    acc = accuracy_score(y_pred, y_pred)

    total_truthful = y_pred.count("truthful")
    total_hallucinated = y_pred.count("hallucinated")

    # Print summary
    print(f"\nQwen Classification Summary for {MODEL_NAME}")
    print(classification_report(y_pred, y_pred, labels=["truthful", "hallucinated"]))
    print("Confusion Matrix (self-agreement, for structure):")
    print(cm)
    print(f"\nTotal Truthful: {total_truthful}")
    print(f"Total Hallucinated: {total_hallucinated}")
    print(f"Accuracy (self-agreement): {acc:.4f}")

    return {
        "name": "HaluEval",
        "model": MODEL_NAME,
        "total_evaluated": len(results),
        "total_truthful": total_truthful,
        "total_hallucinated": total_hallucinated,
        "truthful_precision": round(report["truthful"]["precision"], 4),
        "hallucinated_precision": round(report["hallucinated"]["precision"], 4),
        "truthful_recall": round(report["truthful"]["recall"], 4),
        "hallucinated_recall": round(report["hallucinated"]["recall"], 4),
        "accuracy": round(acc, 4)
    }
# run functions
def run():
    print(f"Benchmarking {MODEL_NAME} on VQA-RAD")
    results = benchmark_model(dataset, count=COUNT) # Adjust count as needed 
    return summarize(results)
if __name__ == "__main__":
    run()
