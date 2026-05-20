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
    "question": [
        "Does Valve have managers?",
        "How do I choose what to work on?",
        "What is the policy on vacations?",
        "Why do desks have wheels?",
        "What is a Cabal?",
        "How are hiring decisions made?",
        "Who can fire an employee?",
        "What is a T-shaped employee?",
        "How do peer reviews work?",
        "What is stack ranking?",
        "Are there strict office hours?",
        "What happens if I make a mistake?",
        "Who is the founder of Valve?",
        "How do I get a promotion?",
        "What is the dress code?",
        "How do teams form?",
        "What is the role of a team lead?",
        "Do I need approval to ship a feature?",
        "How much overtime is expected?",
        "What is the ultimate goal of a Valve employee?",
        "Who tells me what to do?",
        "How do I move to a new project?",
        "What is the policy on training?",
        "How is compensation determined?",
        "What if I disagree with a peer?"
    ],
    "answer": [
        "Valve has no management, and nobody reports to anybody else.",
        "You vote with your desk by rolling it to the project that interests you.",
        "Vacation time is untracked and unlimited. You are expected to take time off.",
        "Desks have wheels so employees can easily move to new teams and projects.",
        "A Cabal is a multidisciplinary project team formed organically by employees.",
        "Hiring is the most important thing in the universe at Valve. It requires unanimous agreement.",
        "Anyone can be fired, and peer feedback heavily influences employment status.",
        "Someone who is both a generalist and an absolute expert in one specific area.",
        "Employees anonymously review their peers, and this data is used for stack ranking.",
        "A system where employees are ranked by peers to determine compensation and value.",
        "There are no set hours, but people generally work normal business hours.",
        "Mistakes are considered learning opportunities as long as you learn from them.",
        "Gabe Newell is the founder and president.",
        "Promotions do not exist in the traditional sense because there is no hierarchy.",
        "There is no official dress code. You can wear whatever you want.",
        "Teams form organically when people decide to work together on a shared goal.",
        "A team lead is an information clearinghouse, not a boss.",
        "No, you are trusted to make decisions and ship features on your own.",
        "Overtime is not encouraged. If you are working crunch hours, something is wrong.",
        "To entertain customers and create value for them.",
        "Nobody. You are your own boss.",
        "You literally unplug your computer, roll your desk to the new area, and plug it in.",
        "You learn by doing and by asking your peers for help.",
        "Compensation is heavily based on peer reviews and your stack rank.",
        "You are expected to talk it out and reach a consensus without a manager stepping in."
    ],
    "contexts": [
        ["Valve is flat. Valve has no management, and nobody reports to anybody else."],
        ["You decide what to work on. We want you to vote with your wheels."],
        ["We do not track vacation time. We want you to take vacations."],
        ["Your desk has wheels so you can roll it anywhere in the company."],
        ["Cabals are multidisciplinary project teams. They form organically."],
        ["Hiring is the most important thing in the universe. Do not lower the bar."],
        ["Firing is handled by peers. If someone is not contributing, the team removes them."],
        ["We value T-shaped people. Broad skills across disciplines, deep skills in one."],
        ["Peer reviews are gathered annually to give feedback and determine ranking."],
        ["Stack ranking is how we determine compensation based on peer feedback."],
        ["We do not have official hours. Most people are here during normal daytime hours."],
        ["Nobody has ever been fired for making a mistake. Learn from it."],
        ["Gabe Newell is the president, but even he is not your manager."],
        ["Since we are flat, traditional promotions up a corporate ladder do not exist."],
        ["Wear what you want. We do not care about dress codes."],
        ["Teams form when individuals decide a project is worth their time."],
        ["Team leads are just centers for the team, they do not dictate work."],
        ["You have the power to greenlight your own projects and ship them."],
        ["Crunch time is a sign of bad planning. Go home and rest."],
        ["Our singular goal is to build things that customers want and entertain them."],
        ["You are the only person who can tell yourself what to do."],
        ["To switch teams, just roll your desk over to the new Cabal."],
        ["There is no formal training program. Find a peer and ask questions."],
        ["Pay is determined by how much value your peers think you bring to the company."],
        ["Arguments happen. Work it out like adults because there is no manager to escalate to."]
    ],
    "ground_truth": [
        "Valve is a flat company with no traditional managers.",
        "Employees self direct and choose projects that interest them.",
        "There is no formal tracking of vacation time.",
        "To allow employees to physically relocate to new teams easily.",
        "A self organized team of employees working on a specific project.",
        "It is treated as a critical task requiring high standards.",
        "Peers determine if someone should be let go due to poor performance.",
        "An individual with broad general knowledge and deep specialized expertise.",
        "Feedback is collected from peers to evaluate performance.",
        "A peer driven ranking system used for evaluating employee value.",
        "No, hours are flexible.",
        "Mistakes are accepted as part of the innovative process.",
        "Gabe Newell.",
        "Valve does not have a traditional corporate ladder for promotions.",
        "Valve does not enforce a dress code.",
        "Organically through employee interest.",
        "To facilitate communication, not to manage people.",
        "Employees are empowered to make shipping decisions.",
        "Consistent overtime is viewed as a failure in planning.",
        "Creating value and entertainment for the customer.",
        "Employees manage themselves.",
        "By physically moving your desk to the new project area.",
        "Learning is informal and peer driven.",
        "It is tied directly to peer evaluations and stack ranking.",
        "Peers must resolve conflicts directly with each other."
    ]
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
