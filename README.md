# ECG LLM

This folder contains the complete implementation for the master thesis project on fine-tuning large language models for medical question-answering tasks. The project includes data generation, model fine-tuning, RAG (Retrieval-Augmented Generation) implementation, and comprehensive evaluation frameworks.


## Setup

### Prerequisites

- Python 3.10
- CUDA-compatible GPU (for training and inference)
- Access to Hugging Face
- API keys for OpenRouter and LiteLLM for evaluation

### Installation

Install required dependencies for finetuning and evaluation
```bash
pip install -r requirments.txt
```
Edit `.env` file to set API keys. You can set the huggingface token, open router token and litellm key. The litellm referes to justadd.ai. To use other providers, you have to rewrite the corresponding code. The litellm key was only used to evaluate the API models Claude and LLama 3.1 70B. LLama 3.1 70B can also be evaluated on an H100. Therefore, you can specify the model in the corresponding local evaluation.

To run the datageneration, you have to install MinerU. Please refer to their documentation to install the GPU-version. Please be sure, that GPU local usage is possible
https://opendatalab.github.io/MinerU/quick_start/, https://github.com/opendatalab/MinerU?tab=readme-ov-file
To install the other required dependencies for datagneration
```bash
cd datageneration
pip install -r requirements_datageneration.txt
```

## Project Structure

```
submission/
├── datageneration/         # Data generation and preprocessing. Here, a separate requirements.txt is set
├── finetuning/            # Model fine-tuning components
├── rag/                   # RAG implementation
├── evaluation/           # Evaluation (multiple choices, bleu and rouge for api and local models), LLM as a judge implementation
├── .env.template         # Environment variables template
├── requirments.txt      # Python dependencies for evaluation, rag and finetuning
```

## Data Generation

The datageneration pipeline converts pdf files using MinerU. Then, the corresponding markdowns are cleaned. For example, files, that cannot be proceeded at once are merged and unneccessary files are deleted. Based on the markdowns, the LLama 3.3 70B generates question-answer pairs which are saved to json. Also, multiple choices are generated and saved to jsons.

you maybe have to change the paths.

Run the complete data generation pipeline by:
```bash
python datageneration/datageneration_pipeline.py
```
Adjust it, if you do not want to run every function.
## Finetuning

The finetuning module uses QLoRA. To run the finetuning, you have to edit the `finetuning/config.py` to customize the LoRA parameters, training argeuments and the model selection and dataset configurations. You can specify also using test data in training data or change test and val data. The `trainer_callbacks.py`allows for customized evaluation during training and saving customized data also to mlflow. For now, only a small multiple choice evaluation is integrated.


### Usage

```bash
python finetuning/finetuning-test.py
```

## RAG Implementation

In the RAG implementation, there is the Q&A Evaluation and the multiple choices evaluation for the rag model. To run the RAG, first, the documents should be embedded (embedding_documents.py). Therefore, pubmedbert embedding and recursive splitting is used. Then, the corresponding files could be run. Therefore, you can specifiy different parameters in the files:

  -  db_path = "/dss/work/toex4699/chroma_db_master_thesis_pubmedbert_recursive_reranking" # dbpath of the vector database
  -  model_path = "meta-llama/Llama-3.1-8B-Instruct" # model path (llama 3.1 8b ist used here)
  -  initial_model = "meta-llama/Llama-3.1-8B-Instruct" # initial model for tokenizer settings
  -  top_k = 20 # rag-parameter. This is used in the master thesis
   - reranking = 5 # reranking parameter. This is used in the master thesis
- collection_name = "ecg_haverkamps_markdowns" # collection name
- test_file = "/dss/work/toex4699/datasets/data_with_chatgpt/bigger_test.jsonl" # test datafile


## Evaluation

Comprehensive evaluation framework supporting multiple metrics and methodologies. The evaluation system includes multiple choice evaluation, Q&A evaluation with automatic metrics (BLEU, BERTScore, ROUGE), and LLM-as-a-judge evaluation using Prometheus.

### Multiple Choice Evaluation

#### Local Evaluation (GPU-based)
Evaluates models running locally on GPU with support for quantization and memory optimization. LLama 3.1 70B is loaded in 4bit otherwise it cannot be evaluated on H100 locally.

```bash
python evaluation/local_evaluation/evaluation_local_multiple_choices.py \
    --model="meta-llama/Llama-3.1-8B-Instruct" \
    --initial_model="meta-llama/Llama-3.1-8B-Instruct" \

```

#### API Evaluation (Cloud-based)
Evaluates models via API calls using LiteLLM

```bash
python evaluation/api_evaluation/evaluation_multiple_choice.py \
    --model="claude-3-7-sonnet-latest"
```
Use any model from LiteLLM. Apart from claude, llama-3.1-70b-instruct was used.

### Q&A Evaluation with Automatic Metrics

#### Local Evaluation

```bash
python evaluation/local_evaluation/evaluation_local_q_a_bleu.py \
    --model="meta-llama/Llama-3.1-8B-Instruct" \
    --initial_model="meta-llama/Llama-3.1-8B-Instruct"
```


#### API Evaluation (Cloud-based metrics)

```bash
python evaluation/api_evaluation/evaluation_q_a.py \
    --model="claude-3-7-sonnet-latest"
```

### LLM-as-a-Judge Evaluation

Uses LLM as a judge framework for evaluation of correctness. The prompt is based on https://arxiv.org/abs/2405.01535
The LLM as a judge loads Deepseek from openrouter
```bash
python evaluation/llmjudge/prometheus_try.py
```

you have to specifiy the json file path. The json has to include the question, the answers to evaluate and the models. Also, the context, where the question comes from, is saved here and loaded by the judge LLM.
Example for the json:
```json
{
  "question": "Question?",
  "answer": [
    "answer 1",
    "answer 2"...
  ],
  "models": [
    "Model A",
    "Model B"
  ],
  "context": "context help to evaluate the correctness of the answers to the question"
}
```

### Configuration and Data Paths

Before running the evaluations, you have to specify the dataset paths and configurations.

## Notes

- Maybe you have to specify paths, when logging, loading datasets ect....

