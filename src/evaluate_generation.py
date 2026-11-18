
import csv
import json
from pathlib import Path

from src.reranking import reranked_search
from src.llm_generator import generate_answer


INPUT_FILE = Path("evaluation/evaluation_questions.csv")
OUTPUT_FILE = Path("evaluation/generation_results.csv")

TOP_K = 5

ABSTENTION_PHRASE = "i couldn't find enough information"


def get_retrieval_details(retrieved_chunks):

    details = []

    for rank, chunk in enumerate(retrieved_chunks, start=1):

        details.append({
            "rank": rank,
            "source": chunk["source"],
            "page": chunk["page"],
            "rerank_score": chunk.get("rerank_score"),
            "rrf_score": chunk.get("rrf_score"),
            "text": chunk["text"]
        })

    return details


def check_expected_sources(expected_source, retrieved_sources):

    expected_sources = [
        source.strip()
        for source in expected_source.split(";")
        if source.strip()
    ]

    # Unanswerable questions have no expected source document
    if not expected_sources or expected_sources[0].upper() == "NONE":

        return {
            "expected_sources_found": "",
            "missing_expected_sources": "",
            "expected_source_hit_any": "N/A",
            "expected_source_hit_all": "N/A"
        }

    found = [
        source for source in expected_sources
        if source in retrieved_sources
    ]

    missing = [
        source for source in expected_sources
        if source not in retrieved_sources
    ]

    return {
        "expected_sources_found": "; ".join(found),
        "missing_expected_sources": "; ".join(missing),
        "expected_source_hit_any": bool(found),
        "expected_source_hit_all": not missing
    }


def evaluate_generation():

    results = []
    successful_count = 0
    error_count = 0

    # Read evaluation questions
    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)
        questions = list(reader)

    print(f"\nTotal questions to evaluate: {len(questions)}")

    for index, row in enumerate(questions, start=1):

        question_id = row["question_id"]
        question = row["question"]
        expected_source = row["expected_source"]
        reference_answer = row["reference_answer"]
        question_type = row["question_type"]

        print(f"\nProcessing {index}/{len(questions)}: {question_id}")
        print(f"Question: {question}")

        retrieved_chunks = []

        try:

            # Step 1: Retrieve and rerank relevant chunks
            retrieved_chunks = reranked_search(
                question,
                top_k=TOP_K
            )

            # Step 2: Generate answer
            if retrieved_chunks:

                generated_answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            else:

                generated_answer = (
                    "I couldn't find enough information "
                    "in the provided documents."
                )

            # Step 3: Record retrieved sources
            retrieved_sources = sorted({
                chunk["source"] for chunk in retrieved_chunks
            })

            # Step 4: Check expected source retrieval
            source_check = check_expected_sources(
                expected_source,
                retrieved_sources
            )

            # Step 5: Save actual retrieved chunk details
            chunk_details = get_retrieval_details(
                retrieved_chunks
            )

            # Step 6: Basic abstention indicator
            possible_abstention = (
                ABSTENTION_PHRASE in generated_answer.lower()
            )

            results.append({

                "question_id": question_id,
                "question": question,
                "question_type": question_type,
                "expected_source": expected_source,
                "reference_answer": reference_answer,

                "retrieved_sources": "; ".join(retrieved_sources),

                "expected_sources_found":
                    source_check["expected_sources_found"],

                "missing_expected_sources":
                    source_check["missing_expected_sources"],

                "expected_source_hit_any":
                    source_check["expected_source_hit_any"],

                "expected_source_hit_all":
                    source_check["expected_source_hit_all"],

                "retrieved_chunk_details":
                    json.dumps(chunk_details, ensure_ascii=False),

                "generated_answer": generated_answer,
                "possible_abstention": possible_abstention

            })

            successful_count += 1

            print("Generated answer:")
            print(generated_answer)

            print(
                "Expected source found:",
                source_check["expected_source_hit_any"]
            )

        except Exception as error:

            print(f"Error processing {question_id}: {error}")

            error_count += 1

            retrieved_sources = sorted({
                chunk["source"] for chunk in retrieved_chunks
            })

            source_check = check_expected_sources(
                expected_source,
                retrieved_sources
            )

            chunk_details = get_retrieval_details(
                retrieved_chunks
            )

            results.append({

                "question_id": question_id,
                "question": question,
                "question_type": question_type,
                "expected_source": expected_source,
                "reference_answer": reference_answer,

                "retrieved_sources": "; ".join(retrieved_sources),

                "expected_sources_found":
                    source_check["expected_sources_found"],

                "missing_expected_sources":
                    source_check["missing_expected_sources"],

                "expected_source_hit_any":
                    source_check["expected_source_hit_any"],

                "expected_source_hit_all":
                    source_check["expected_source_hit_all"],

                "retrieved_chunk_details":
                    json.dumps(chunk_details, ensure_ascii=False),

                "generated_answer": f"ERROR: {error}",
                "possible_abstention": False

            })

    # Save all results to CSV
    fieldnames = [
        "question_id",
        "question",
        "question_type",
        "expected_source",
        "reference_answer",
        "retrieved_sources",
        "expected_sources_found",
        "missing_expected_sources",
        "expected_source_hit_any",
        "expected_source_hit_all",
        "retrieved_chunk_details",
        "generated_answer",
        "possible_abstention"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n========================================")
    print("GENERATION EVALUATION COMPLETED")
    print("========================================")

    print(f"Total questions: {len(questions)}")
    print(f"Successfully evaluated: {successful_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_generation()
