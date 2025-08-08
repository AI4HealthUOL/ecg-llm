import os
import sys

import torch
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import sys
from typing import Any, List

import chromadb
import tiktoken
import torch
import transformers
from chromadb import Documents, EmbeddingFunction, Embeddings
from datasets import load_dataset
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import transformers
from bert_score import score as bert_score
from dotenv import load_dotenv

load_dotenv()
import os
from typing import List

import chromadb
import tiktoken
import transformers
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer

load_dotenv()
from loguru import logger


class PubMedEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input[0], str):
            return self.model.encode(input)
        return self.model.encode([doc.page_content for doc in input])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.encode(text)


def token_length(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# https://docs.llamaindex.ai/en/stable/examples/embeddings/custom_embeddings/
class LLamaindexEmbedding(BaseEmbedding):
    #  _model: SentenceTransformer = PrivateAttr()
    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> None:

        super().__init__(**kwargs)
        object.__setattr__(self, "_model", SentenceTransformer(model_name))

    #       self._model = SentenceTransformer("neuml/pubmedbert-base-embeddings")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode(query)
        return embeddings

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode(text)
        return embeddings


class MultilingualEmbedding(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input[0], str):
            return self.model.encode(input)
        return self.model.encode([doc.page_content for doc in input])

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.encode(text)


class ModelEvaluator:
    def __init__(
        self,
        model_path,
        initial_model,
        test_file_path,
        db_path,
        collection_name,
        pubmedbert=True,
        topk=20,
        reranking=5,
    ):  # embedding_model="neuml/pubmedbert-base-embeddings"):
        logger.info("Initializing model evaluator...")
        self.is_initial_model = model_path == initial_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=False,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            # )
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     model_path,
            #     quantization_config=bnb_config,
            #     torch_dtype=torch.bfloat16,
            # )
            if "70" in model_path:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,  # Use bfloat16 for 70B models
                    device_map={"": 0},  # Force everything to GPU
                    max_memory={0: "40GB"},  # Use almost all GPU memory
                    use_cache=True,
                    low_cpu_mem_usage=True,
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
        self.test_data = self._load_test_data(test_file_path)
        logger.info("Test data loaded.")
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        logger.info("Rouge scorer loaded.")

        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        if pubmedbert:
            self.collection = client.get_or_create_collection(collection_name, embedding_function=PubMedEmbedding())
        else:
            self.collection = client.get_or_create_collection(
                collection_name, embedding_function=MultilingualEmbedding()
            )

        if pubmedbert:
            embedding_model = "neuml/pubmedbert-base-embeddings"
        else:
            embedding_model = "intfloat/multilingual-e5-large-instruct"
        self.embedder = SentenceTransformer(embedding_model)
        # use chromadb with llamaindex

        logger.info("Model loaded.")
        logger.info(f"Use Collection: {collection_name}")
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=LLamaindexEmbedding(model_name=embedding_model),
            show_progress=True,
        )
        reranker = SentenceTransformerRerank(
            top_n=reranking, model="cross-encoder/ms-marco-MiniLM-L-2-v2"  # Crossencoder llamaindex
        )

        self.retriever = index.as_retriever(similarity_top_k=topk, node_postprocessors=[reranker])

    def _get_relevant_context(self, question):
        try:
            retrieved_nodes_en = self.retriever.retrieve(question)
            retrieved_nodes = []
            retrieved_nodes.extend(retrieved_nodes_en)

            context_parts = [f"[{i+1}] {node.text.strip()}" for i, node in enumerate(retrieved_nodes)]
            context = "\n\n".join(context_parts)
            logger.debug(f"Retrieved context (combined): {context}")
        except Exception as e:
            context = "No relevant context found."
            logger.error(f"Error retrieving context: {e}")
        return context

    def _load_test_data(self, file_path):
        logger.info("Loading test data...")
        data = load_dataset("json", data_files={"test": file_path})
        test_data = data["test"]
        logger.info(data["test"])
        self.test_data = test_data
        logger.info("Test data loaded.")
        return data["test"]

    def _form_prompt(self, system_message, user_message):
        if system_message != "":
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {user_message}
        <|eot_id|>
 <|start_header_id|>assistant<|end_header_id|>""".strip()

    # #         else:
    #         return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    #  {user_message}<|eot_id|>
    #  <|start_header_id|>assistant<|end_header_id|>""".strip()

    def evaluate(self, system_message=""):
        self.model.eval()
        generation_config = transformers.GenerationConfig(
            max_new_tokens=200,
            temperature=0.1,
            top_p=0.9,
            num_return_sequences=1,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
            do_sample=True,
        )

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
            data_point_question = data_point["question"]
            context = self._get_relevant_context(data_point_question)
            if self.is_initial_model:
                question = self._form_prompt(
                    system_message,
                    f"""{data_point_question} Please answer shortly in one sentence without lists!
                                             Here is the context which helps to answer the question: {context}""",
                )
            else:
                question = self._form_prompt(
                    system_message,
                    f"""{data_point_question}
                                             Here is the context which helps to answer the question: {context}""",
                )
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(**inputs, generation_config=generation_config)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            try:
                answer_start_token = "assistant"
                answer_start_index = generated_text.find(answer_start_token)
                answer_text = generated_text[answer_start_index:].strip()
                if not answer_text:
                    answer_text = generated_text
            except:
                answer_text = generated_text

            reference_text = data_point["answer"]
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
            bertscore_f1_scores.append(bert_f1.mean().item())
            bertscore_precision_scores.append(bert_precision.mean().item())
            bertscore_recall_scores.append(bert_recall.mean().item())
            total += 1

        # Calculate all metrics
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
        logger.info(f"rouge_1 scores: {rouge_1_scores}")
        logger.info(f"rouge_2 scores: {rouge_2_scores}")
        logger.info(f"rouge_L scores: {rouge_L_scores}")
        logger.info(f"rouge_1 precisions: {rouge_1_precisions}")
        logger.info(f"rouge_1 recalls: {rouge_1_recalls}")
        logger.info(f"rouge_2 precisions: {rouge_2_precisions}")
        logger.info(f"rouge_2 recalls: {rouge_2_recalls}")
        logger.info(f"rouge_L precisions: {rouge_L_precisions}")
        logger.info(f"rouge_L recalls: {rouge_L_recalls}")
        logger.info(f"bertscore_f1_scores: {bertscore_f1_scores}")
        logger.info(f"bertscore_precision_scores: {bertscore_precision_scores}")
        logger.info(f"bertscore_recall_scores: {bertscore_recall_scores}")
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

    db_path = "/dss/work/toex4699/chroma_db_master_thesis_pubmedbert_recursive_reranking"
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    initial_model = "meta-llama/Llama-3.1-8B-Instruct"
    top_k = 20
    reranking = 5
    collection_name = "ecg_haverkamps_markdowns"
    test_file = "/dss/work/toex4699/datasets/data_with_chatgpt/bigger_test.jsonl"

    logging_file = "/dss/work/toex4699/logs/evaluate_qa_local.log"
    logger.add(logging_file, format="{time} {level} {message}", level="INFO")

    logger.info(f"Evaluating model: {model_path}")
    evaluator = ModelEvaluator(
        model_path,
        initial_model,
        test_file,
        db_path=db_path,
        pubmedbert=True,
        collection_name=collection_name,
        topk=top_k,
        reranking=reranking,
    )
    system_message = ""
    initial_results = evaluator.evaluate(system_message)
    for metric, value in initial_results.items():
        logger.info(f"{metric}: {value:.4f}")

    logger.info("Evaluation Haverkamps testset: ")
    evaluator._load_test_data("/dss/work/toex4699/datasets/haverkamp_evaluation/chat_gpt_pairs.json")
    initial_results = evaluator.evaluate(system_message)
    for metric, value in initial_results.items():
        logger.info(f"{metric}: {value:.4f}")

    logger.info("Evaluation special questions ")
    evaluator._load_test_data("/dss/work/toex4699/datasets/specific_dataset/specific_test.jsonl")
    initial_results = evaluator.evaluate(system_message)
    for metric, value in initial_results.items():
        logger.info(f"{metric}: {value:.4f}")
