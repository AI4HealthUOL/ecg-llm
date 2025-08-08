import json
from pathlib import Path
from typing import List

# from dotenv import load_dotenv
from langchain_core.output_parsers.pydantic import (
    JsonOutputParser,
    PydanticOutputParser,
)
from pydantic import BaseModel, Field

from utils.generator import Generator


class Option(BaseModel):
    text: str = Field(description="The text of the answer option (ENGLISH)")
    is_correct: bool = Field(description="Whether this option is correct")


class MultipleChoiceQA(BaseModel):
    question: str = Field(description="The question text (ENGLISH)")
    options: List[Option] = Field(description="List of 4 possible answers (ENGLISH)")


class MultipleChoiceList(BaseModel):
    multiple_choice_questions: List[MultipleChoiceQA] = Field(
        description="List of multiple choice questions (ENGLISH)"
    )


class MultipleChoiceGenerator(Generator):
    def __init__(
        self, api_token: str, logger, processing_content: str, output_folder: str, tokencount_limit: int = 48000
    ):
        super().__init__(api_token, logger, processing_content, output_folder, tokencount_limit)
        self.parser = PydanticOutputParser(pydantic_object=MultipleChoiceList)
        self.json_parser = JsonOutputParser(pydantic_object=MultipleChoiceList)

    def generate_prompt_from_context(self, context: str) -> List:
        parser = PydanticOutputParser(pydantic_object=MultipleChoiceList)

        system_message = f"""You are a Teacher/Professor in the medical field.
        Your task is to create multiple choice questions with different difficulties based on the medical context provided.

        For each question:
        - Generate one question about important medical content
        - Create exactly 4 answer options
        - One option must be correct and from the context
        - Three options must be plausible but incorrect (make these up)
        - All options should have similar length and style
        - Focus only on medical content
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

        user_message = f"""Generate multiple choice questions from this context:

    {context}
    {parser.get_format_instructions()}
    Please answer in English and do not use any German words or phrases. Translate technical terams into English if necessary.

    """
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
        return messages

    def save_pairs(self, pairs: dict, filename: str, output_dir: str = "evaluation/data"):
        """Save multiple choice questions to JSON file."""
        if not pairs:
            self.logger.warning("No multiple choices to save")
            return

        output_path = Path(self.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"multiple_choices_{filename}.json"

        try:
            mc_dicts = []
            for page_range, multiple_choices in pairs.items():
                if multiple_choices:
                    for qa in multiple_choices:
                        mc_dicts.append(
                            {
                                "question": qa.question,
                                "options": [
                                    {"text": option.text, "is_correct": option.is_correct} for option in qa.options
                                ],
                                "context": f"{page_range}",
                            }
                        )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(mc_dicts, f, indent=2)
            self.logger.success(f"Saved {len(mc_dicts)} multiple choice questions to {output_file}")
            self.logger.success(f"Successfully processed {filename}")
        except Exception as e:
            self.logger.error(f"Error saving multiple choice questions: {e}")
