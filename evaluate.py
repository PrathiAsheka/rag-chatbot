"""
evaluate.py — Score your RAG pipeline using RAGAS metrics.

Metrics:
  - faithfulness     : Is the answer grounded in the retrieved context?
  - answer_relevancy : Is the answer relevant to the question?
  - context_recall   : Did retrieval find the right chunks?

Usage:
  python evaluate.py
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from datasets import Dataset
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()


def run_evaluation():
    # Sample Q&A pairs — replace with questions relevant to your documents
    test_cases = [
        {
            "question": "What is the main topic of the document?",
            "ground_truth": "Replace this with the actual expected answer."
        },
        {
            "question": "What are the key findings?",
            "ground_truth": "Replace this with the actual expected answer."
        },
    ]

    # Initialize pipeline — update with your actual PDF path
    pipeline = RAGPipeline()
    pipeline.load_documents(["your_document.pdf"])

    # Collect results
    questions, answers, contexts, ground_truths = [], [], [], []

    for case in test_cases:
        result = pipeline.query(case["question"])
        questions.append(case["question"])
        answers.append(result["answer"])
        contexts.append([s["content"] for s in result["sources"]])
        ground_truths.append(case["ground_truth"])

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Run evaluation
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    print("\n📊 RAGAS Evaluation Results:")
    print(results)
    return results


if __name__ == "__main__":
    run_evaluation()
