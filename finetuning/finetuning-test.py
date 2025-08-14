import logging
import os

import numpy as np
import torch
import transformers
from config import Config
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rouge_score import rouge_scorer
from trainer_callbacks import Evaluation_Callback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

load_dotenv()

logging.basicConfig(
    filename="training.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
login(token)


def test_cuda_available():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


def get_bigger_dataset(output_folder: str):
    data = load_dataset(
        "json",
        data_files={
            "train": os.path.join(output_folder, "bigger_training.jsonl"),
            "test": os.path.join(output_folder, "bigger_test.jsonl"),
            "val": os.path.join(output_folder, "bigger_val.jsonl"),
        },
    )
    return data["train"], data["test"], data["val"]


def get_multiple_choices_specific_train(output_folder: str):
    # only 1000 for training -> ensure that the output is suitable for the model
    data = load_dataset(
        "json",
        data_files={
            "train": os.path.join(output_folder, "1000_train_multiple_choice.jsonl"),
            "test": os.path.join(output_folder, "test_multiple_choice.jsonl"),
            "val": os.path.join(output_folder, "val_multiple_choice.jsonl"),
        },
    )
    return data["train"], data["val"]


class ModelLoader:
    def __init__(self, model_name: str, tokenizer_name: str, config: Config):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.config = config

    def print_trainable_parameters(self, model):
        """Prints the number of trainable parameters in the model."""
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def load_model(self):
        # for QLora
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            # use 16bit more efficient
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN"),
            force_download=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # pad_token_id set to token <|finetune_right_pad_id|>
        # tokenizer.pad_token_id = 128004
        # the module tranformers says eos or pad
        # cannot use pad, than there is a cuda error
        tokenizer.pad_token_id = 128004  # tokenizer.eos_token_id
        model = self.prepare_model(model)
        model = self.configure_training(model)
        return model, tokenizer

    def prepare_model(self, model):
        # less vram
        model.gradient_checkpointing_enable()
        # improve preformance when finetuning with LoRA
        model.enable_input_require_grads()
        model = prepare_model_for_kbit_training(model)
        return model

    def configure_training(self, model):
        config = LoraConfig(
            r=self.config.lora_config.r,
            lora_alpha=self.config.lora_config.lora_alpha,
            target_modules=self.config.lora_config.target_modules,
            lora_dropout=self.config.lora_config.lora_dropout,
            bias=self.config.lora_config.bias,
            task_type=self.config.lora_config.task_type,
        )
        model = get_peft_model(model, config)
        self.print_trainable_parameters(model)
        return model

    def change_generation_config(self, model, tokenizer):
        if not hasattr(model, "generation_config") or model.generation_config is None:
            model.generation_config = transformers.GenerationConfig(
                max_new_tokens=200,
                temperature=0.1,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            generation_config = model.generation_config
            generation_config.max_new_tokens = 200
            generation_config.temperature = 0.1
            generation_config.top_p = 0.9
            generation_config.num_return_sequences = 1
            generation_config.pad_token_id = tokenizer.pad_token_id
            generation_config.eos_token_id = tokenizer.eos_token_id

    def generate_and_tokenize_multiple_choices(self, data_point, tokenizer):
        question = data_point["question"]
        answer = data_point["answer"]
        question_tokenized = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(tokenizer.bos_token)
            + tokenizer.tokenize("<|start_header_id|>user<|end_header_id|> ")
            + tokenizer.tokenize("Answer the following multiple-choice question: ")
            + tokenizer.tokenize(question)
            + tokenizer.tokenize("<|eot_id|>")
        )
        answer_tokenized = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<|start_header_id|>assistant<|end_header_id|> ")
            + tokenizer.tokenize(f"The correct answer is {answer}.")
        )
        answer_tokenized += tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<|eot_id|>") + tokenizer.tokenize(tokenizer.eos_token)
        )
        attention_mask = [1] * len(question_tokenized + answer_tokenized)
        input_ids = question_tokenized + answer_tokenized
        labels = [-100] * len(question_tokenized) + answer_tokenized
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def generate_and_tokenize_prompt(self, data_point, tokenizer):
        question = data_point["question"]
        answer = data_point["answer"]
        question_tokenized = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(tokenizer.bos_token)
            + tokenizer.tokenize("<|start_header_id|>user<|end_header_id|> ")
            + tokenizer.tokenize(question)
            + tokenizer.tokenize("<|eot_id|>")
        )
        answer_tokenized = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<|start_header_id|>assistant<|end_header_id|> ") + tokenizer.tokenize(answer)
        )
        answer_tokenized += tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<|eot_id|>") + tokenizer.tokenize(tokenizer.eos_token)
        )

        #   we need labels https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy

        # no padding, because the default datacollator does that, maybe thats more accurate
        # https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L52

        attention_mask = [1] * len(question_tokenized + answer_tokenized)
        input_ids = question_tokenized + answer_tokenized
        labels = [-100] * len(question_tokenized) + answer_tokenized
        # Batchencoding is the result of the tokenizer https://medium.com/@awaldeep/hugging-face-understanding-tokenizers-1b7e4afdb154
        # batch = BatchEncoding(data={"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}), but we use a datacollator

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def prepare_evaluation_dataset(self, data_point, tokenizer):
        question = data_point["question"]
        answer = data_point["answer"]
        input_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(tokenizer.bos_token)
            + tokenizer.tokenize("<|start_header_id|>user<|end_header_id|> ")
            + tokenizer.tokenize(question)
            + tokenizer.tokenize("<|eot_id|>")
            + tokenizer.tokenize("<|start_header_id|>assistant<|end_header_id|> ")
        )
        answer_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"The correct answer is {answer}."))
        answer_tokenized += tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<|eot_id|>") + tokenizer.tokenize(tokenizer.eos_token)
        )
        return {"input_ids": input_ids, "tokenized_answer": answer_tokenized}

    def prepare_evaluation_multiple(self, data_point, tokenizer):
        question = data_point["question"]
        answer = data_point["answer"]
        input_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(tokenizer.bos_token)
            + tokenizer.tokenize("<|eot_id|>")
            + tokenizer.tokenize("<|start_header_id|>user<|end_header_id|> ")
            + tokenizer.tokenize("Answer the following multiple-choice question: ")
            + tokenizer.tokenize(question)
            + tokenizer.tokenize(" <|eot_id|>")
            + tokenizer.tokenize("<|start_header_id|>assistant<|end_header_id|> ")
        )
        answer_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answer))
        answer_tokenized += tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("<|eot_id|>") + tokenizer.tokenize(tokenizer.eos_token)
        )
        return {"input_ids": input_ids, "tokenized_answer": answer_tokenized}

    def load_data(self, data_path: str, tokenizer):
        train_multiple_choices, val_multiple_choices = get_multiple_choices_specific_train(
            "/dss/work/toex4699/datasets"
        )
        train_multiple_choices = train_multiple_choices.shuffle()

        train_data, test_data, val_data = get_bigger_dataset("/dss/work/toex4699/datasets/")

        print(train_data)
        # Tokenize the datasets
        train_data = train_data.shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, tokenizer))
        test_data = test_data.shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, tokenizer))
        train_multiple_choices = train_multiple_choices.shuffle().map(
            lambda x: self.generate_and_tokenize_multiple_choices(x, tokenizer)
        )
        val_data = val_data.shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, tokenizer))
        evaluation_dataset = val_data.shuffle().map(lambda x: self.prepare_evaluation_dataset(x, tokenizer))
        evaluation_multiple_choices = val_multiple_choices.shuffle().map(
            lambda x: self.prepare_evaluation_multiple(x, tokenizer)
        )
        print(train_data)
        # Concatenate the datasets to also use some of the multiple choice data -> the model should be able to answer multiple choices!
        if self.config.include_test_data:
            if self.config.use_test_as_val:
                train_data_both = concatenate_datasets([train_data, train_multiple_choices, val_data])
            else:
                train_data_both = concatenate_datasets([train_data, train_multiple_choices, test_data])
        else:
            train_data_both = concatenate_datasets([train_data, train_multiple_choices])

        train_data_both = train_data_both.shuffle()
        if self.config.use_test_as_val:
            val_data_both = concatenate_datasets([test_data])
        else:
            val_data_both = concatenate_datasets([val_data])
        print(val_data)
        print(evaluation_dataset)
        all_evaluation_multiple_choices = concatenate_datasets([evaluation_multiple_choices])
        if self.config.testing:
            train_data_both = train_data_both.select(range(100))
            val_data_both = val_data_both.select(range(10))

        return train_data_both, val_data_both, evaluation_dataset, all_evaluation_multiple_choices


def training(model_name, tokenizer_name, input_data):
    config = Config()
    model, tokenizer, train_data, val_data, evaluation_dataset, multiple_dataset = prepare_training(
        model_name, tokenizer_name, input_data
    )
    do_training(model, tokenizer, train_data, val_data, evaluation_dataset, multiple_dataset, config)


def prepare_training(model, tokenizer, input_data: str):
    model_loader = ModelLoader(model, tokenizer, config)
    model, tokenizer = model_loader.load_model()
    train_data, val_data, evaluation_dataset, multiple_dataset = model_loader.load_data(input_data, tokenizer)
    return model, tokenizer, train_data, val_data, evaluation_dataset, multiple_dataset


def compute_metrics(pred, tokenizer):
    predictions = pred.predictions
    references = pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)
    if isinstance(references, np.ndarray):
        references = torch.tensor(references)

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    bleu_scores = []
    rouge_precisions = []
    rouge_recalls = []
    rouge_measures = []
    for reference, generated_orig in zip(references, predictions):
        # we only want to compare the answer.
        generated = [generated_orig[i] for i in range(len(reference)) if reference[i] != -100]
        reference = [token for token in reference if token != -100]

        reference_text = tokenizer.decode(reference, skip_special_tokens=True)
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)

        bleu_score = sentence_bleu([reference_text], generated_text)
        bleu_scores.append(bleu_score)

        scores = scorer.score(reference_text, generated_text)
        rouge_precisions.append(scores["rouge1"].precision)
        rouge_recalls.append(scores["rouge1"].recall)
        rouge_measures.append(scores["rouge1"].fmeasure)

    # bert score causes memory errors
    return {
        "bleu": sum(bleu_scores) / len(bleu_scores),
        "rouge_precision": sum(rouge_precisions) / len(rouge_precisions),
        "rouge_recall": sum(rouge_recalls) / len(rouge_recalls),
        "rouge_fmeasure": sum(rouge_measures) / len(rouge_measures),
    }


def preprocess_logits_for_metrics(logits, labels):
    """Original Trainer may have a memory leak.

    This is a workaround to avoid storing too many tensors that are not needed. Found at
    https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941
    """
    # they are using model(input_ids) and only predict the next token. To get the right output, we have to use
    # https://discuss.huggingface.co/t/generate-without-using-the-generate-method/11379
    # put we are not able to come to that code
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def do_training(model, tokenizer, train_data, val_data, evaluation_dataset, multiple_dataset, config):
    training_args = TrainingArguments(
        per_device_train_batch_size=config.training_args.per_device_train_batch_size,
        per_device_eval_batch_size=config.training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training_args.gradient_accumulation_steps,
        evaluation_strategy=config.training_args.evaluation_strategy,
        do_train=config.training_args.do_train,
        do_eval=config.training_args.do_eval,
        num_train_epochs=config.training_args.num_train_epochs,
        learning_rate=config.training_args.learning_rate,
        fp16=config.training_args.fp16,
        save_total_limit=config.training_args.save_total_limit,
        logging_steps=config.training_args.logging_steps,
        eval_steps=config.training_args.eval_steps,
        save_steps=config.training_args.save_steps,
        output_dir=config.training_args.output_dir,
        max_steps=config.training_args.max_steps,
        max_grad_norm=config.training_args.max_grad_norm,
        optim=config.training_args.optim,
        lr_scheduler_type=config.training_args.lr_scheduler_type,
        warmup_ratio=config.training_args.warmup_ratio,
        report_to=config.training_args.report_to,
        include_for_metrics=["inputs", "model"],
        # remove_unused_columns=False,# all but attention_mask, lables and input ids is deleted ?
    )

    data_files = [
        "/dss/work/toex4699/datasets/train.jsonl",
        "/dss/work/toex4699/datasets/val.jsonl",
        "/dss/work/toex4699/datasets/test.jsonl",
        "/user/toex4699/master_thesis/finetuning/config.py",
        "/user/toex4699/master_thesis/finetuning/trainer_callbacks.py",
    ]
    script_path = "/user/toex4699/master_thesis/finetuning/finetuning-test.py"
    evaluation_callback = Evaluation_Callback(
        data_files, script_path, tokenizer, evaluation_dataset, multiple_dataset
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        # this collator padds also the labels https://github.com/huggingface/transformers/pull/8274 , because the default (DataCollatorWithPadding) does not (https://github.com/huggingface/transformers/issues/8146), no own padding neccessary -> save costs, because have to set the largest padding
        data_collator=transformers.DataCollatorForTokenClassification(tokenizer),
        #   data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False), # it changes our labels!!!https://discuss.huggingface.co/t/how-labelled-data-is-processed-transformers-trainer/80644/9
        callbacks=[evaluation_callback],
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    model.config.use_cache = False
    trainer.train()


if __name__ == "__main__":
    test_cuda_available()
    config = Config()
    training(config.model, config.tokenizer_name, "/dss/work/toex4699/datasets/dataset_with_context_q_a.jsonl")
