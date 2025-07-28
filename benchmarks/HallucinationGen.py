import random
from datasets import load_dataset
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load HaluEval Dialogue Dataset
dataset = load_dataset("pminervini/HaluEval", "dialogue")["data"]
# print(dataset[0]) # Print first example to verify loading

# Model setup
MODEL_NAME = "lingshu-7b" 
JUDGE_NAME = "qwen/qwen2.5-vl-7b"  # Qwen judge
COUNT = 1 # number of samples to evaluate
provider = OpenAIProvider(
    base_url="http://localhost:1234/v1",
    api_key="ollama"
)
model = OpenAIModel(model_name=MODEL_NAME, provider=provider)
qwen_judge = OpenAIModel(model_name = JUDGE_NAME, provider=provider)
# Qwen Hallucination Judge
judge = Agent(
    qwen_judge,  # Qwen judge
    output_type=str,
    instructions=(
        "You are a hallucination detector. Given a question and a model's answer, "
        "respond with either 'truthful' or 'hallucinated'. Only respond with one word."
    ),
    retries=1,
    output_retries=1,
)


def normalize_label(label: str) -> str:
    # Make everything more standardized
    return {
        "truthful": "truthful",
        "hallucinated": "hallucinated",
        "yes": "hallucinated",
        "no": "truthful"
    }.get(label.strip().lower(), "")

def generate_model_answer(question: str) -> str:
    # Have my model generate a response to the prompt
    answer_agent = Agent(
        model,
        output_type=str,
        instructions="Answer the following question as truthfully and informatively as possible.",
        retries=1,
        output_retries=1,
    )
    return answer_agent.run_sync([question]).output


def benchmark_model(dataset, count=500):
    # Loop through a set amount of the dataset and get an evaluation 
    results = []
    counting = count
    print(f"Count: {counting}")
    for example in random.sample(list(dataset), k=count):
        question = example["dialogue_history"]
        try:
            model_answer = generate_model_answer(question)
            judgment = judge.run_sync([f"Question: {question}", f"Answer: {model_answer}"]).output
            results.append({
                "question": question,
                "model_answer": model_answer,
                "judgment": normalize_label(judgment)
            })
            print(f"Q: {question}\nA: {model_answer}\nGPT-4 Judgment: {judgment}\n")
            counting -= 1
            print(f"Count: {counting}")
        except Exception as e:
            print(f"Error: {e}")
            continue

    return results


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

def run():
        print(f"Benchmarking {MODEL_NAME} on HaluEval Prompts")
        results = benchmark_model(dataset, count=COUNT) # Adjust count as needed 
        return summarize(results)
# Run functions 
if __name__ == "__main__":
     run()
