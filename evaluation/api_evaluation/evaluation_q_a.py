import argparse
import os

import torch
from bert_score import score as bert_score
from datasets import load_dataset
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

load_dotenv()
from langchain_core.utils import convert_to_secret_str
from langchain_openai import ChatOpenAI
from loguru import logger


class ModelEvaluator:
    def __init__(self, model, test_file_path):
        logger.info("Initializing model evaluator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.client = ChatOpenAI(
            model=model,
            base_url=os.getenv("LITELLM_URL"),
            api_key=convert_to_secret_str(os.getenv("LITELLM_KEY")),
            timeout=600,
            temperature=0.1,
            max_tokens=100,
        )
        logger.info("Model loaded.")
        logger.info("Client loaded.")
        self.test_data = self._load_test_data(test_file_path)
        logger.info("Test data loaded.")
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        logger.info("Rouge scorer loaded.")

    def _load_test_data(self, file_path):
        logger.info("Loading test data...")
        data = load_dataset("json", data_files={"test": file_path})
        data["test"]
        logger.info(data["test"])

        logger.info("Test data loaded.")
        return data["test"]

    def evaluate(self, system_message=""):

        generated_texts = []
        reference_texts = []
        bleu_scores = []
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_L_scores = []
        rouge_1_precisions = []
        rouge_1_recalls = []
        rouge_2_precisions = []
        rouge_2_recalls = []
        rouge_L_precisions = []
        rouge_L_recalls = []

        bertscore_f1_scores = []
        bertscore_precision_scores = []
        bertscore_recall_scores = []

        total = 0
        for i, data_point in enumerate(self.test_data):

            reference_text = data_point["answer"]
            question = data_point["question"] + "Please answer shortly in one sentence without lists!"
            response = self.client(question)
            answer_text = response.content
            logger.info(f"Question: {question}")
            logger.info(f"Answer: {answer_text}")
            logger.info(f"Reference: {reference_text}")
            generated_texts.append(answer_text)
            reference_texts.append(reference_text)
            rouge_1_score = self.scorer.score(reference_text, answer_text)["rouge1"]
            rouge_2_score = self.scorer.score(reference_text, answer_text)["rouge2"]
            rouge_L_score = self.scorer.score(reference_text, answer_text)["rougeL"]
            rouge_1_scores.append(rouge_1_score.fmeasure)
            rouge_2_scores.append(rouge_2_score.fmeasure)
            rouge_L_scores.append(rouge_L_score.fmeasure)
            rouge_1_precisions.append(rouge_1_score.precision)
            rouge_1_recalls.append(rouge_1_score.recall)
            rouge_2_precisions.append(rouge_2_score.precision)
            rouge_2_recalls.append(rouge_2_score.recall)
            rouge_L_precisions.append(rouge_L_score.precision)
            rouge_L_recalls.append(rouge_L_score.recall)
            bleu_scores.append(sentence_bleu([reference_text.split()], answer_text.split()))
            bert_f1, bert_precision, bert_recall = bert_score(
                [answer_text], [reference_text], lang="en", rescale_with_baseline=True
            )
            bertscore_f1_scores.append(bert_f1.item())
            bertscore_precision_scores.append(bert_precision.item())
            bertscore_recall_scores.append(bert_recall.item())
            total += 1

        results = self._calculate_metrics(
            rouge_1_scores,
            rouge_2_scores,
            rouge_L_scores,
            rouge_1_precisions,
            rouge_1_recalls,
            rouge_2_precisions,
            rouge_2_recalls,
            rouge_L_precisions,
            rouge_L_recalls,
            bleu_scores,
            bertscore_f1_scores,
            bertscore_precision_scores,
            bertscore_recall_scores,
        )
        return results

    def _calculate_metrics(
        self,
        rouge_1_scores,
        rouge_2_scores,
        rouge_L_scores,
        rouge_1_precisions,
        rouge_1_recalls,
        rouge_2_precisions,
        rouge_2_recalls,
        rouge_L_precisions,
        rouge_L_recalls,
        bleu_scores,
        bertscore_f1_scores,
        bertscore_precision_scores,
        bertscore_recall_scores,
    ):
        logger.info("Calculating metrics...")
        logger.info(f"bleu scores: {bleu_scores}")
        logger.info(f"rouge1 scores: {rouge_1_scores}")
        logger.info(f"rouge2 scores: {rouge_2_scores}")
        logger.info(f"rougeL scores: {rouge_L_scores}")
        logger.info(f"rouge1 precisions: {rouge_1_precisions}")
        logger.info(f"rouge1 recalls: {rouge_1_recalls}")
        logger.info(f"rouge2 precisions: {rouge_2_precisions}")
        logger.info(f"rouge2 recalls: {rouge_2_recalls}")
        logger.info(f"rougeL precisions: {rouge_L_precisions}")
        logger.info(f"rougeL recalls: {rouge_L_recalls}")
        logger.info(f"bertscore f1 scores: {bertscore_f1_scores}")
        logger.info(f"bertscore precision scores: {bertscore_precision_scores}")
        logger.info(f"bertscore recall scores: {bertscore_recall_scores}")
        results = {
            "bleu": sum(bleu_scores) / len(bleu_scores),
            "rouge1_fmeasure": sum(rouge_1_scores) / len(rouge_1_scores),
            "rouge1_precision": sum(rouge_1_precisions) / len(rouge_1_precisions),
            "rouge1_recall": sum(rouge_1_recalls) / len(rouge_1_recalls),
            "rouge2_fmeasure": sum(rouge_2_scores) / len(rouge_2_scores),
            "rouge2_precision": sum(rouge_2_precisions) / len(rouge_2_precisions),
            "rouge2_recall": sum(rouge_2_recalls) / len(rouge_2_recalls),
            "rougeL_fmeasure": sum(rouge_L_scores) / len(rouge_L_scores),
            "rougeL_precision": sum(rouge_L_precisions) / len(rouge_L_precisions),
            "rougeL_recall": sum(rouge_L_recalls) / len(rouge_L_recalls),
            "bertscore_f1": sum(bertscore_f1_scores) / len(bertscore_f1_scores),
            "bertscore_precision": sum(bertscore_precision_scores) / len(bertscore_precision_scores),
            "bertscore_recall": sum(bertscore_recall_scores) / len(bertscore_recall_scores),
        }

        return results


if __name__ == "__main__":
    logging_file = "/logs/evaluate_q_a.log"
    logger.add(logging_file, format="{time} {level} {message}", level="INFO")
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model.")
    parser.add_argument("--model", type=str, required=True, help="The model path or name.")
    args = parser.parse_args()

    model_path = args.model

    test_file = "/datasets/test.jsonl"

    logger.info(f"Evaluating model: {model_path}")
    evaluator = ModelEvaluator(model_path, test_file)
    initial_results = evaluator.evaluate()
    logger.info("Model Results:")
    for metric, value in initial_results.items():
        logger.info(f"{metric}: {value:.4f}")
