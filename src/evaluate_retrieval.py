
import csv
from pathlib import Path

from src.semantic_search import semantic_search
from src.hybrid_search import hybrid_search
from src.reranking import reranked_search


# File paths
CSV_PATH = Path("evaluation/evaluation_questions.csv")
OUTPUT_PATH = Path("evaluation/retrieval_results.csv")

TOP_K = 5


# Find the rank of the first relevant document
def get_first_relevant_rank(results, expected_sources):

    for rank, result in enumerate(results, start=1):

        if result["source"] in expected_sources:
            return rank

    return None


# Calculate retrieval metrics
def calculate_metrics(ranks, total_questions):

    metrics = {}

    for k in [1, 3, 5]:

        hits = sum(
            1 for rank in ranks
            if rank is not None and rank <= k
        )

        metrics[f"Hit@{k}"] = hits / total_questions

    reciprocal_ranks = [
        1 / rank if rank is not None else 0
        for rank in ranks
    ]

    metrics["MRR"] = sum(reciprocal_ranks) / total_questions

    return metrics


def main():

    # Read evaluation questions
    with open(CSV_PATH, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        questions = list(reader)


    print("TOTAL QUESTIONS:", len(questions))


    # Separate answerable and unanswerable questions
    answerable_questions = [
        row for row in questions
        if row["expected_source"].strip().upper() != "NONE"
    ]

    unanswerable_count = len(questions) - len(answerable_questions)

    print("ANSWERABLE QUESTIONS:", len(answerable_questions))
    print("UNANSWERABLE QUESTIONS:", unanswerable_count)


    # Store ranks for metric calculation
    semantic_ranks = []
    hybrid_ranks = []
    reranked_ranks = []

    detailed_results = []


    # Evaluate each answerable question
    for index, row in enumerate(answerable_questions, start=1):

        question_id = row["question_id"]
        question = row["question"]

        expected_sources = {
            source.strip()
            for source in row["expected_source"].split(";")
        }

        print(f"\nEvaluating {index}/{len(answerable_questions)}: {question_id}")


        # 1. Semantic search
        semantic_results = semantic_search(question, top_k=TOP_K)

        semantic_rank = get_first_relevant_rank(
            semantic_results,
            expected_sources
        )


        # 2. Hybrid search
        hybrid_results = hybrid_search(question, top_k=TOP_K)

        hybrid_rank = get_first_relevant_rank(
            hybrid_results,
            expected_sources
        )


        # 3. Hybrid search + reranking
        reranked_results = reranked_search(question, top_k=TOP_K)

        reranked_rank = get_first_relevant_rank(
            reranked_results,
            expected_sources
        )


        # Save ranks
        semantic_ranks.append(semantic_rank)
        hybrid_ranks.append(hybrid_rank)
        reranked_ranks.append(reranked_rank)


        # Save individual question results
        detailed_results.append({
            "question_id": question_id,
            "question": question,
            "expected_source": row["expected_source"],
            "semantic_rank": semantic_rank or "Not found",
            "hybrid_rank": hybrid_rank or "Not found",
            "reranked_rank": reranked_rank or "Not found"
        })


    # Calculate metrics
    total = len(answerable_questions)

    semantic_metrics = calculate_metrics(semantic_ranks, total)
    hybrid_metrics = calculate_metrics(hybrid_ranks, total)
    reranked_metrics = calculate_metrics(reranked_ranks, total)


    # Display comparison
    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 65)

    print(f"{'Metric':<15}{'Semantic':<15}{'Hybrid':<15}{'Reranked':<15}")

    for metric in ["Hit@1", "Hit@3", "Hit@5", "MRR"]:

        print(
            f"{metric:<15}"
            f"{semantic_metrics[metric]:<15.4f}"
            f"{hybrid_metrics[metric]:<15.4f}"
            f"{reranked_metrics[metric]:<15.4f}"
        )


    # Save detailed results to CSV
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:

        fieldnames = [
            "question_id",
            "question",
            "expected_source",
            "semantic_rank",
            "hybrid_rank",
            "reranked_rank"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(detailed_results)


    print("\nDetailed results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
