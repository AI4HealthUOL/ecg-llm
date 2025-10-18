import argparse
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transformers
from dotenv import load_dotenv

load_dotenv()
from loguru import logger


class MultipleChoiceEvaluator:
    def __init__(self, model_path, initial_model):
        logger.info("Initializing model evaluator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            if "70" in initial_model:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,  # Use full precision since we have 48GB
                    device_map={"": 0},  # Force everything to GPU
                    max_memory={0: "40GB"},  # Use almost all GPU memory
                    use_cache=True,
                    low_cpu_mem_usage=True,
                    # Additional memory optimizations
                    offload_folder=None,  # No CPU offloading
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)

        logger.info("Model loaded.")
        self.tokenizer = AutoTokenizer.from_pretrained(initial_model)
        logger.info("Tokenizer loaded.")

    def apply_chat_template(self, system_message, user_message):
        prompt = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

        prompt_token_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        input_ids_tensor = torch.tensor(prompt_token_ids).to(self.device)
        inputs = {"input_ids": input_ids_tensor}
        return inputs

    def _form_prompt(self, system_message, user_message):
        if system_message != "":
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_message}<|eot_id|>
 <|start_header_id|>assistant<|end_header_id|>""".strip()
        else:
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
 {user_message}<|eot_id|>
 <|start_header_id|>assistant<|end_header_id|>""".strip()

    def evaluate_multiple_choice(self, mc_data, system_message=""):
        """Evaluate model on multiple choice questions."""
        self.model.eval()
        generation_config = transformers.GenerationConfig(
            max_new_tokens=50,
            temperature=0.1,
            top_p=0.9,
            num_return_sequences=1,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            do_sample=True,
        )

        correct = 0
        total = 0

        for data_point in mc_data:
            question = data_point["question"]
            prompt = f"""Answer the following multiple choice question:
            {question}"""
            prompt_to_tokenize = self._form_prompt(system_message, prompt)
            inputs = self.tokenizer(prompt_to_tokenize, return_tensors="pt").to(self.device)  # .to("cuda:0")

            generated_ids = self.model.generate(**inputs, generation_config=generation_config)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            assistant_start = generated_text.rfind("assistant")
            assistant_answer = generated_text[assistant_start + len("assistant") :]
            try:
                answer_start_token = "The correct answer is:"
                answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "The correct answer is:"
                    answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "Answer:"
                    answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "answer is"
                    answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "answer is:"
                    answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "Let's think: ("
                    answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "Let's think: "
                    answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = ":"
                    answer_start_index = assistant_answer.find(answer_start_token)

                answer_text = assistant_answer[answer_start_index + len(answer_start_token) :].strip()
                correct_answer = data_point["answer"].upper()
                logger.debug(f"Answer: {answer_text}, Correct answer: {correct_answer}")
                answer_text = answer_text[0].upper()
                if answer_text not in ["A", "B", "C", "D"]:
                    if answer_text == "1":
                        answer_text = "A"
                    elif answer_text == "2":
                        answer_text = "B"
                    elif answer_text == "3":
                        answer_text = "C"
                    elif answer_text == "4":
                        answer_text = "D"

                if correct_answer == answer_text:
                    correct += 1
                total += 1

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


def evaluate_all_datasets(evaluator, multiple_data, system_message):
    """Evaluate all datasets and return combined results."""
    total_correct = 0
    total_questions = 0

    dataset_results = {}

    for dataset_name in multiple_data.keys():
        logger.info(f"Running Multiple Choice evaluation on {dataset_name}...")
        results = evaluator.evaluate_multiple_choice(multiple_data[dataset_name], system_message)

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

    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model.")
    parser.add_argument("--model", type=str, required=True, help="The model path or name.")
    parser.add_argument("--initial_model", type=str, required=True, help="The initial model")
    parser.add_argument("--small", type=str, help="The initial model")
    args = parser.parse_args()
    system_message = """You have to answer multiple choice questions. You have to choose the correct answer from the options.
    IMPORTANT Please answer in the form
'The correct answer is A' or 'The correct answer is B' or 'The correct answer is C' or 'The correct answer is D'.
EXAMPLE:
User: What is the capital of France? A: Paris B: London C: Rome D: Madrid
Assistant: The correct answer is A

User: What is the capital of Germany? A: Paris B: Berlin C: Rome D: Madrid
Assistant: The correct answer is B

"""

    model_path = args.model
    initial_model = args.initial_model
    logging_file = "/logs/evaluate_multiple_choice_local.log"
    logger.add(logging_file, format="{time} {level} {message}", level="INFO")
    logger.info(f"Evaluating model: {model_path}")

    evaluator = MultipleChoiceEvaluator(model_path, initial_model)

    logger.info("Running Multiple Choice evaluation on special data...")
    special = "/specific_dataset/special_multiple_choice.jsonl"

    multiple_data = load_dataset(
        "json", data_files={"special": special}
    )

    all_results = evaluate_all_datasets(evaluator, multiple_data, system_message)
    logger.info("Use whole Multiple Choices...")
    mc_file = "/datasets/all_multiple_choice.jsonl"

    multiple_data = load_dataset(
        "json", data_files={"all": mc_file}
    )

    all_results = evaluate_all_datasets(evaluator, multiple_data, system_message)
    haverkamp = "/datasets/haverkamp_evaluation/haverkamp_multiple_choices.jsonl"

    logger.info("Evaluate Haverkamps now...")

    multiple_data = load_dataset("json", data_files={"haverkamp": haverkamp})
    all_results = evaluate_all_datasets(evaluator, multiple_data, system_message)
