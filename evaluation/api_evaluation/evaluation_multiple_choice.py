import argparse
import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
from langchain_core.utils import convert_to_secret_str
from langchain_openai import ChatOpenAI
from loguru import logger


class MultipleChoiceEvaluator:
    def __init__(self, model):
        logger.info("Initializing model evaluator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.client = ChatOpenAI(
            model=model,
            base_url=os.getenv("LITELLM_URL"),
            api_key=convert_to_secret_str(os.getenv("LITELLM_KEY")),
            timeout=600,
            temperature=0.1,
            max_tokens=1,
        )
        logger.info("Model loaded.")
        logger.info("Client loaded.")

    def evaluate_multiple_choice(self, mc_data):
        """Evaluate model on multiple choice questions."""

        correct = 0
        total = 0

        for data_point in mc_data:
            question = data_point["question"]
            prompt = f"""Answer the following multiple choice question:
            {question}. Please only output A, B, C or D"""

            try:
                response = self.client(prompt)
                answer_text = response.content

                correct_answer = data_point["answer"].upper()
                if correct_answer == answer_text:
                    correct += 1
                total += 1
                logger.info(f"Correct answer: {correct_answer}, predicted answer: {answer_text}")

            except Exception as e:
                logger.info(f"Error processing answer: {e}")
                continue

        results = {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
        }
        logger.info("Multiple Choice Results:")
        logger.info(f"Total questions: {results['total_questions']}")
        logger.info(f"Correct answers: {results['correct_answers']}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        return results


def evaluate_all_datasets(evaluator, multiple_data):
    total_correct = 0
    total_questions = 0

    dataset_results = {}

    for dataset_name in multiple_data.keys():
        logger.info(f"Running Multiple Choice evaluation on {dataset_name}...")
        results = evaluator.evaluate_multiple_choice(multiple_data[dataset_name])

        dataset_results[dataset_name] = results

        total_correct += results["correct_answers"]
        total_questions += results["total_questions"]

    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    logger.info("=" * 50)
    logger.info("Overall Results:")
    logger.info(f"Total questions across all datasets: {total_questions}")
    logger.info(f"Total correct answers: {total_correct}")
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")

    logger.info("\nIndividual Dataset Results:")
    for dataset_name, results in dataset_results.items():
        logger.info(f"\n{dataset_name}:")
        logger.info(f"Questions: {results['total_questions']}")
        logger.info(f"Correct: {results['correct_answers']}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")

    return {
        "overall": {
            "total_questions": total_questions,
            "correct_answers": total_correct,
            "accuracy": overall_accuracy,
        },
        "datasets": dataset_results,
    }


if __name__ == "__main__":
    load_dotenv()
    logging_file = "/dss/work/toex4699/logs/evaluate_multiple_choice_haverkamp.log"
    logger.add(logging_file, format="{time} {level} {message}", level="INFO")
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model.")
    parser.add_argument("--model", type=str, required=True, help="The model path or name.")
    args = parser.parse_args()

    model_path = args.model

    logger.info(f"Evaluating model: {model_path}")
    evaluator = MultipleChoiceEvaluator(model_path)

    logger.info("Prepare Multiple Choice evaluation...")

    logger.info("Running Multiple Choice evaluation on special data...")
    special_train = "/dss/work/toex4699/datasets/specific_dataset/train_multiple_choice.jsonl"
    special_test = "/dss/work/toex4699/datasets/specific_dataset/test_multiple_choice.jsonl"
    special_val = "/dss/work/toex4699/datasets/specific_dataset/val_multiple_choice.jsonl"

    multiple_data = load_dataset(
        "json", data_files={"train": special_train, "test": special_test, "val": special_val}
    )

    all_results = evaluate_all_datasets(evaluator, multiple_data)
    logger.info("Use whole Multiple Choices...")
    mc_file = "/dss/work/toex4699/datasets/test_multiple_choice.jsonl"
    mc_file_val = "/dss/work/toex4699/datasets/val_multiple_choice.jsonl"
    mc_file_more = "/dss/work/toex4699/datasets/old_train_new_test_multiple_choices.jsonl"

    multiple_data = load_dataset(
        "json", data_files={"test": mc_file, "old_train": mc_file_more, "val": mc_file_val}
    )

    all_results = evaluate_all_datasets(evaluator, multiple_data)
    haverkamp = "/dss/work/toex4699/datasets/haverkamp_evaluation/haverkamp_multiple_choices.jsonl"

    logger.info("Evaluate Haverkamps now...")

    multiple_data = load_dataset("json", data_files={"haverkamp": haverkamp})
    all_results = evaluate_all_datasets(evaluator, multiple_data)
    logger.info("Use english Multiple Choices...")

    mc_file_english = "/dss/work/toex4699/datasets/multiple_choices_english/formated_english_multiple_choices.jsonl"
    multiple_data = load_dataset("json", data_files={"english_file": mc_file_english})

    all_results = evaluate_all_datasets(evaluator, multiple_data)
