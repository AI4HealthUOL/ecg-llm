import json
import os
import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_core.output_parsers.pydantic import (
    JsonOutputParser,
    PydanticOutputParser,
)

from utils.generator import Generator


class QAPair(BaseModel):
    question: str = Field(description="The question based on given context (ENGLISH)", title="Question")
    answer: str = Field(description="The answer to the question (ENGLISH)", title="Answer")


class QAPairList(BaseModel):
    qa_pairs: List[QAPair] = Field(description="List of question-answer pairs", title="Q&A Pairs")


class QAGenerator(Generator):
    def __init__(
        self,
        api_token: str,
        logger,
        processing_content: str,
        output_folder: str,
        tokencount_limit: int = 50000,
        use_huggingface=True,
    ):
        super().__init__(api_token, logger, processing_content, output_folder, tokencount_limit, use_huggingface)
        self.parser = PydanticOutputParser(pydantic_object=QAPairList)
        self.json_parser = JsonOutputParser(pydantic_object=QAPairList)

    def generate_prompt_from_context(self, context: str) -> List:

        system_message = f"""You are a Teacher/ Professor in the medical field. Your task is to setup a examination with free text answers. Using the provided context, formulate questions with different difficulties that capture the medical content from the context. Please also give the answers, so that the test could be corrected afterwards.
        If you cannot generate any question to medical content, please skip it.
    You MUST obey the following criteria:
    - Restrict the question to the context information provided.
    - vary between different question words (e.g. what, why, which, how, etc.)
    - Ensure every question is fully self-contained and answerable without requiring additional context or prior questions/answers.
    - Do NOT ask for figures, algortihms, tables, names of the present study or similar.
    - Do NOT put phrases like "given provided context" or "in this work" or "in this case" or "what is algorithm a" or some questions regarding the study
    - Replace these terms with specific details
    - ONLY ask for medical details. DO NOT ask about the study, author or index, IGNORE them

    BAD questions:
    - What are the symptoms of the disease described
    - How many patients were included in the study

    GOOD questions:
    - What are the symptoms of Corona
    - Why should the patient drink so much water when having a fever

    Sometimes the context may contain overhead such as titles, authors, study information or similar. Please only use the content that has a medical context.

    Output: ONLY JSON."""

        user_message = f"""Here is the context for generating medical questions:

    {context}
    {self.parser.get_format_instructions()}
    Please answer in English and do not use any German words or phrases. Translate technical terams into English if necessary.

    """

        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
        return messages

    def save_pairs(self, pairs: dict, filename: str):
        """Save QA pairs to JSON file."""
        if not pairs:
            self.logger.warning("No QA pairs to save")
            return

        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"qa_pairs_{filename}.json"

        try:
            qa_dicts = []
            for page_range, q_a_pairs in pairs.items():
                if q_a_pairs:
                    for qa in q_a_pairs:
                        qa_dicts.append({"question": qa.question, "answer": qa.answer, "context": f"{page_range}"})
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(qa_dicts, f, indent=2)
            self.logger.success(f"Saved {len(qa_dicts)} q-a pairs to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving QA pairs: {e}")
