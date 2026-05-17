import os
import math
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
print("Booting up Evaluation Layer...")

judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
judge_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

data_samples = {
    "question": ["According to the handbook, who is the manager at Valve?"],
    "answer": ["Valve has no management, and nobody 'reports' to anybody else. Even the founder/president isn't your manager."],
    "contexts": [["Valve is flat. Valve has no management, and nobody 'reports' to anybody else. We do have a founder/president, but even he isn't your manager."]],
    "ground_truth": ["Valve does not have managers. The company has a flat structure where nobody reports to anyone."]
}

dataset = Dataset.from_dict(data_samples)

print("Evaluating metrics...")
results = evaluate(
    dataset,
    metrics=[Faithfulness(), AnswerRelevancy()],
    llm=judge_llm,
    embeddings=judge_embeddings
)

print(f"\nEvaluation Results:\n{results}")

print("\nGenerating visual chart...")

#for handling nan values
f_score = results.get("faithfulness", 0.0)
ar_score = results.get("answer_relevancy", 0.0)

f_score = 0.0 if math.isnan(f_score) else float(f_score)
ar_score = 0.0 if math.isnan(ar_score) else float(ar_score)

metrics_names = ["Faithfulness", "Answer Relevancy"]
metrics_scores = [f_score, ar_score]

plt.figure(figsize=(6, 5))
bars = plt.bar(metrics_names, metrics_scores, color=['#4CAF50', '#2196F3'], width=0.4)

plt.ylim(0, 1.1)  
plt.title("Agentic RAG Performance", fontsize=14, fontweight='bold')
plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

image_path = "evaluation_chart.png"
plt.savefig(image_path, bbox_inches='tight')
print(f"Success! Chart saved as: {image_path}")