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
from llama_index.vector_stores.chroma import ChromaVectorStore
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transformers
from dotenv import load_dotenv

load_dotenv()
import os
from typing import List

import chromadb
import tiktoken
from chromadb import Documents, EmbeddingFunction, Embeddings
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.storage.storage_context import StorageContext
from loguru import logger
from sentence_transformers import SentenceTransformer

# https://medium.com/@praveencs87/chunking-in-rag-retrieval-augmented-generation-a-beginners-guide-28b5a81a8877 markdown splitting, weil markdown dokumente
# https://docs.trychroma.com/docs/embeddings/embedding-functions

# https://research.trychroma.com/evaluating-chunking
# https://huggingface.co/NeuML/pubmedbert-base-embeddings
# recursive ist nicht so gut


# https://python.langchain.com/docs/how_to/markdown_header_metadata_splitter/
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
class PubMedEmbeddingLLamaindex(BaseEmbedding):
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


class RAGMultipleChoiceEvaluator:
    def __init__(
        self,
        model_path,
        initial_model,
        db_path,
        collection_name="ecg_haverkamps_markdowns",
        embedding_model="neuml/pubmedbert-base-embeddings",
        top_k=20,
        reranking=5,
    ):
        logger.add("/logs/rag_evaluation.log", format="{time} {level} {message}", level="INFO")
        logger.info("Initializing RAG model evaluator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            if model_path == "meta-llama/Llama-3.1-70B-Instruct":
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
                    torch_dtype=torch.float32,
                    device_map={"": 0},
                    max_memory={0: "40GB"},
                    use_cache=True,
                    low_cpu_mem_usage=True,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(initial_model)

        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=db_path)
        self.collection = client.get_or_create_collection(
            "ecg_haverkamps_markdowns", embedding_function=PubMedEmbedding()
        )
        self.embedder = SentenceTransformer(embedding_model)
        # use chromadb with llamaindex

        logger.info("Model loaded.")
        logger.info(f"Use Collection: {collection_name}")

        vector_store = ChromaVectorStore(chroma_collection=self.collection, text_key="text")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=PubMedEmbeddingLLamaindex(model_name="neuml/pubmedbert-base-embeddings"),
            show_progress=True,
        )

        reranker = SentenceTransformerRerank(
            top_n=reranking, model="cross-encoder/ms-marco-MiniLM-L-2-v2"  # oder eigener CrossEncoder
        )

        self.retriever = index.as_retriever(similarity_top_k=top_k, node_postprocessors=[reranker])

    def _get_token_length(self, text):
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            return 50000

    def _get_relevant_context(self, question):

        retrieved_nodes = self.retriever.retrieve(question)

        context_parts = [f"[{i+1}] {node.text.strip()}" for i, node in enumerate(retrieved_nodes)]
        context = "\n\n".join(context_parts)
        # logger.debug(f"Retrieved context (combined): {context}")
        return context

    def _form_prompt(self, system_message, context, question):
        system_message = (
            system_message
        ) = """You have to answer multiple choice questions. You have to choose the correct answer from the options.
    IMPORTANT Please answer in the form
'The correct answer is A' or 'The correct answer is B' or 'The correct answer is C' or 'The correct answer is D'.
EXAMPLE:
User: What is the capital of France? A: Paris B: London C: Rome D: Madrid
Assistant: The correct answer is A

User: What is the capital of Germany? A: Paris B: Berlin C: Rome D: Madrid
Assistant: The correct answer is B

"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is some relevant context to help answer the question:
{context}

Now, please answer this multiple choice question:
{question}. Please only output A, B, C or D<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""".strip()

    def _extract_question_text(self, full_question):
        """Extract the main question text without the multiple choice options."""
        # Common patterns for multiple choice answers
        patterns = ["A)", "a)", "A.", "a.", "1)", "1.", " A "]

        earliest_pos = len(full_question)
        for pattern in patterns:
            pos = full_question.find(pattern)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos

        if earliest_pos < len(full_question):
            question = full_question[:earliest_pos].strip()
        #  logger.info(f"Extracted question: {question}")
        return question.strip()  # Return full text if no pattern found

    def evaluate_multiple_choice(self, mc_data, system_message=""):
        """Evaluate model on multiple choice questions with RAG."""
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
            full_question = data_point["question"]

            question_text = self._extract_question_text(full_question)
            logger.debug(f"Question: {question_text}")
            context = self._get_relevant_context(question_text)

            prompt = self._form_prompt(system_message, context, full_question)
            #   logger.debug(f"Context: {context}")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(**inputs, generation_config=generation_config)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            try:
                assistant_start = generated_text.rfind("assistant")
                assistant_answer = generated_text[assistant_start + 9 :].strip()
                # logger.debug(f"Assistant answer: {assistant_answer}")
                # Extract answer letter using different possible formats
                answer_start_token = "The correct answer is:"
                answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "The correct answer is"
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
                    answer_start_token = "Let's think:"
                    answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index < 0:
                    answer_start_token = "Let's think"
                    answer_start_index = assistant_answer.find(answer_start_token)
                if answer_start_index >= 0:
                    answer_text = assistant_answer[answer_start_index + len(answer_start_token) :].strip()

                    answer_text = answer_text[0].upper()
                else:
                    answer_text = assistant_answer[0].upper()

                if answer_text not in ["A", "B", "C", "D"]:
                    if answer_text == "1":
                        answer_text = "A"
                    elif answer_text == "2":
                        answer_text = "B"
                    elif answer_text == "3":
                        answer_text = "C"
                    elif answer_text == "4":
                        answer_text = "D"

                # Check if correct
                correct_answer = data_point["answer"].upper()
                logger.debug(f"Answer: {assistant_answer}, Correct answer: {correct_answer}")
                if correct_answer == answer_text:
                    correct += 1
                total += 1

            except Exception as e:
                logger.error(f"Error processing answer: {e}")
                continue

        results = {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
        }
        return results


def evaluate_all_datasets(evaluator, multiple_data, system_message):
    """Evaluate all datasets and return combined results."""
    total_correct = 0
    total_questions = 0

    # Dictionary to store individual results
    dataset_results = {}

    for dataset_name in multiple_data.keys():
        logger.info(f"Running Multiple Choice evaluation on {dataset_name}...")
        results = evaluator.evaluate_multiple_choice(multiple_data[dataset_name], system_message)

        # Store individual results
        dataset_results[dataset_name] = results

        # Add to totals
        total_correct += results["correct_answers"]
        total_questions += results["total_questions"]

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    # Log overall results
    logger.info("=" * 50)
    logger.info("Overall Results:")
    logger.info(f"Total questions across all datasets: {total_questions}")
    logger.info(f"Total correct answers: {total_correct}")
    logger.info(f"Overall accuracy: {overall_accuracy:.4f}")

    # Log individual results
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
    FOLDER_PATH = "/output_cleaned_markdown_pipeline"
    DB_PATH = "/chroma_db_master_thesis_pubmedbert_recursive_reranking"
    model_path = "meta-llama/Llama-3.1-8B-Instruct"
    initial_model = "meta-llama/Llama-3.1-8B-Instruct"
    top_k = 20
    reranking = 5
    logger.add(
        "/logs/markdown_splitting_ecg.log", format="{time} {level} {message}", level="INFO"
    )
    client = chromadb.PersistentClient(path=DB_PATH)

    # https://huggingface.co/NeuML/pubmedbert-base-embeddings
    embedding_model = "neuml/pubmedbert-base-embeddings"

    evaluator = RAGMultipleChoiceEvaluator(
        model_path, initial_model, DB_PATH, "ecg_haverkamps_markdowns", embedding_model, top_k, reranking
    )

    logger.info("Evaluating Multiple Choices on special ...")
    special = "/datasets/specific_dataset/special_multiple_choice.jsonl"

    multiple_data = load_dataset(
        "json", data_files={"special": special}
    )
    system_message = ""
    all_results = evaluate_all_datasets(evaluator, multiple_data, system_message)

    logger.info("Evaluating Multiple Choices on whole ...")
    mc_file = "/datasets/multiple_choice.jsonl"

    multiple_data = load_dataset(
        "json", data_files={"multiple_choice": mc_file}
    )
    system_message = ""
    all_results = evaluate_all_datasets(evaluator, multiple_data, system_message)
    haverkamp = "/datasets/haverkamp_evaluation/haverkamp_multiple_choices.jsonl"

    logger.info("Evaluating Multiple Choices on haverkamp ...")

    multiple_data = load_dataset("json", data_files={"haverkamp": haverkamp})
    all_results = evaluate_all_datasets(evaluator, multiple_data, system_message)
