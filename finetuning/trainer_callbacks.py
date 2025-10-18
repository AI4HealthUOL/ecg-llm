import os
import time
from datetime import datetime

import mlflow
import torch
import transformers
from accelerate import Accelerator
from config import Config
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback


class Evaluation_Callback(TrainerCallback):
    def __init__(self, data_files: list, script_path: str, tokenizer, evaluation_dataset, multiple_dataset):
        mlflow_dir = "/mlflowlogs"
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlflow_dir}")
        experiment_name = "finetuning_experiment"
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        self.data_files = data_files
        self.script_path = script_path
        self.tokenizer = tokenizer
        self.evaluation_dataset = evaluation_dataset

        # prometheus and generate whole answers is expensive -> only multiple choices
        self.multiple_choice_dataset = multiple_dataset
        self.config = Config()
        self.writer = SummaryWriter(log_dir=self.config.training_args.output_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        for file in self.data_files:
            mlflow.log_artifact(file)
        mlflow.log_artifact(self.script_path)
        model = kwargs["model"]
        print(f"Model: {model}")
        # multiple_choices_acc = self._calc_multiple_choices_accuracy(model)
        # mlflow.log_metric("multiple_choices_accuracy", multiple_choices_acc, step=state.global_step)
        # self.writer.add_scalar("multiple_choices_accuracy", multiple_choices_acc, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):

        model = kwargs["model"]

        config = Config()
        try:
            run_id = state.global_step
        except:
            run_id = "0"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(config.training_args.output_dir, f"final_model_{run_id}_{timestamp}")

        accelerator = Accelerator()
        model = accelerator.unwrap_model(model)
        model.save_pretrained(save_path)
        mlflow.log_artifacts(save_path)

    def on_epoch_end(self, args, state, control, **kwargs):

        model = kwargs["model"]

        config = Config()
        try:
            run_id = state.global_step
        except:
            run_id = "0"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(config.training_args.output_dir, f"epoch_model_{run_id}_{timestamp}")

        accelerator = Accelerator()
        model = accelerator.unwrap_model(model)
        model.save_pretrained(save_path)

    def on_evaluate(self, args, state, control, **kwargs):

        model = kwargs["model"]
        if state.global_step % self.config.custom_eval_steps == 0:
            model.eval()

            
            generation_config = transformers.GenerationConfig(
                max_new_tokens=200,
                temperature=0.1,
                top_p=0.9,
                num_return_sequences=1,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                decoder_start_token_id=self.tokenizer.bos_token_id,
                do_sample=True,
            )
            if not hasattr(model, "generation_config"):
                model.generation_config = generation_config

            multiple_choices_acc = self._calc_multiple_choices_accuracy(model)
            mlflow.log_metric("multiple_choices_accuracy", multiple_choices_acc, step=state.global_step)
            self.writer.add_scalar("multiple_choices_accuracy", multiple_choices_acc, state.global_step)

    def _calc_multiple_choices_accuracy(self, model):
        correct = 0
        total = 0
        start = time.time()
        print(f"Calculating multiple choices accuracy with {len(self.multiple_choice_dataset)} data points...")

        generation_config = transformers.GenerationConfig(
            max_new_tokens=10,
            temperature=0.1,
            top_p=0.9,
            num_return_sequences=1,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            do_sample=True,
        )
        if not hasattr(model, "generation_config"):
            model.generation_config = generation_config
        for data_point in self.multiple_choice_dataset:
            input_ids = torch.tensor(data_point["input_ids"]).unsqueeze(0).to("cuda:0")
            generated_ids = model.generate(input_ids, generation_config=generation_config)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if self.config.testing:
                print(f"Generated text: {generated_text}")

            try:
                answer_start_token = "answer is:"
                answer_start_index = generated_text.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "Answer:"
                    answer_start_index = generated_text.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "answer is"
                    answer_start_index = generated_text.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "assistant"
                    answer_start_index = generated_text.find(answer_start_token)
                answer_text = generated_text[
                    answer_start_index + len(answer_start_token) : len(generated_text)
                ].strip()

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
                correct_answer = data_point["answer"]
                if correct_answer == answer_text:
                    correct += 1
                else:
                    print(f"Correct answer: {correct_answer}, generated answer: {generated_text}")
                total += 1
            except:
                print("Error with multiple choice")

        accuracy = correct / total
        end = time.time()
        print(
            f"Multiple choices start: {start}, end: {end}, total: {total}, correct: {correct}, accuracy: {accuracy:.2f}"
        )
        print(f"Time taken: {end - start:.2f} seconds")
        return accuracy
