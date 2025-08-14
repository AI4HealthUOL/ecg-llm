import json
import os
import sys
import time

import torch
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
from loguru import logger
from prometheus_evaluator import PrometheusEvaluator

load_dotenv()


class Prometheus:
    def __init__(self, test_file_path):
        logger.info("Initializing model evaluator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prometheus_evaluation = False

        self.test_data = self._load_test_data(test_file_path)
        self.prometheus_evaluator = self._init_prometheus_evaluator()
        logger.info("Model evaluator initialized.")

    def _load_test_data(self, file_path):
        logger.info("Loading test data...")
        data = load_dataset("json", data_files={"test": file_path})
        test_data = data["test"]
        specific_test_data = test_data
        self.existing_dict = {}

        for entry in test_data:
            self.existing_dict[entry["question"]] = entry
        logger.info(data["test"])

        logger.info("Test data loaded.")
        return specific_test_data

    def _init_prometheus_evaluator(self):
        api_token = os.environ.get(
            "OPEN_ROUTER_TOKEN_Uni"
        )  
        evaluator = PrometheusEvaluator(
            api_token,
            logger,
            model="deepseek-ai/DeepSeek-R1",
            score_rubric="0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
            tokencount_limit=4000,
        )
        return evaluator

    def evaluate(self):

        logger.info("Starting Prometheus ranking evaluation...")
        logger.info(f"Total test data points: {len(self.test_data)}")
        total = 0

        for i, data_point in enumerate(self.test_data):
            try:
                name = f"evaluation"
                if self.existing_dict[data_point["question"]].get(name) is None:

                    context = data_point["context"]

                    total += 1
                    time.sleep(10)
                    evaluation = self.prometheus_evaluator.evaluate(
                        question=data_point["question"],
                        response_A=data_point["answer"][0],
                        #   reference_answer=reference_text,
                        context=context,
                        response_B=data_point["answer"][1],
                        response_C=data_point["answer"][2],
                        response_D=data_point["answer"][3],
                        response_E=data_point["answer"][4],
                        response_F=data_point["answer"][5],
                        response_G=data_point["answer"][6],
                        response_H=data_point["answer"][7],
                        response_I=data_point["answer"][8],
                        response_J=data_point["answer"][9],
                        response_K=data_point["answer"][10],
                        response_L=data_point["answer"][11],
                    )

                    evaluation_dict = {}
                    evaluation_dict["ranking"] = evaluation.ranking
                    evaluation_dict["scores"] = evaluation.scores

                    evaluation_dict["is_correct"] = evaluation.correctness_responses
                    evaluation_dict["style"] = evaluation.style
                    logger.info(f"Prometheus ranking: {evaluation.ranking}")

                    self.existing_dict[data_point["question"]][name] = evaluation_dict
                    if total >= 50:
                        break
            except Exception as e:
                logger.error(f"Error during Prometheus evaluation: {e}")

                logger.info(f"Progress: {total}/{len(self.test_data)*4} questions processed")

        evaluation_list = list(self.existing_dict.values())

        return evaluation_list


if __name__ == "__main__":
    logging_file = "/dss/work/toex4699/logs/PROMETHEUS.log"
    logger.add(logging_file, format="{time} {level} {message}", level="INFO")

    test_file = "/dss/work/toex4699/training_evaluation/Prometheus_results.json"
    evaluator = Prometheus(test_file)
    results = evaluator.evaluate()

    # Save results to file
    output_file = "/dss/work/toex4699/training_evaluation/Prometheus_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nResults saved to {output_file}")
