import json
import os
import sys
from typing import List

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from loguru import logger
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re

from dotenv import load_dotenv
from json_repair import repair_json
from openai import OpenAI

load_dotenv()


class EvaluationRankingCorrected(BaseModel):
    ranking: List[str] = Field(
        description="The ranking of the answers from best to worst, using the letters A, B, C, D, E, F, G, H, I, J, K, L. The ranking should be based on the quality of the responses and the feedback provided."
    )
    style: List[int] = Field(description="The ranking of the style of the answers A, B, C, D, E, F, G, H, I, J, K, L.")
    correctness_responses: List[bool] = Field(
        description="The correctness of the answers A, B, C, D, E, F, G, H, I, J, K, L. The correctness is based on the reference answer and the context."
    )
    scores: List[int] = Field(
        description="The scores of the answers A, B, C, D, E, F, G, H, I, J, K, L. The scores are based on the context."
    )


class PrometheusEvaluator:
    def __init__(
        self,
        api_token: str,
        logger,
        processing_content: str = "",
        output_folder: str = "",
        tokencount_limit: int = 40000,
        model: str = "",
        score_rubric: str = "0,1,2,3,4,5,6,7,8,9,10",
    ):
        self.parser = PydanticOutputParser(pydantic_object=EvaluationRankingCorrected)
        self.score_rubric = score_rubric
        self.logger = logger

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_token,
        )

    def remove_think_tags(self, text):
        cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned_text.strip()

    def get_system_message(
        self,
        question,
        response_A,
        response_B,
        response_C,
        response_D,
        response_E,
        response_F,
        response_G,
        response_H,
        response_I,
        response_J,
        response_K,
        response_L,
        context,
    ):
        criteria = f"""
                    Please use the context as the basis for your assessment. Use this for ranking the answers.
If there is any incorrect information in the response to evaluate that is not in the given context or could not be interpreted from it, take it into account while ranking the answers. We do not want incorrect answers!
Long answers that do not contain correct extra information will score just as highly as short answers."""
        system_message = f"""
                                        ###Task Description:
A question, responses to evaluate (denoted
as Response A, B, C, D, E, F, G, H, I, J , K, L) and the context where the context comes from for extra knowledge are given.
1. Think about why and how you would rank the responses, focusing strictly on the knowledge provided in the context.
2. Make comparisons
between Responses A, B, C, D, E, F, G, H, I, J , K, L. Instead of examining
the Responses separately,
go straight to the point and thought about
the commonalities and differences between
them
3. {criteria}
5. Please do not generate any other opening,
closing, and explanations. Just output JSON following the instruction given by the user.
###Initial Question: {question}
###Response A: {response_A}
###Response B: {response_B}
###Response C: {response_C}
###Response D: {response_D}
###Response E: {response_E}
###Response F: {response_F}
###Response G: {response_G}
###Response H: {response_H}
###Response I: {response_I}
###Response J: {response_J}
###Response K: {response_K}
###Response L: {response_L}
###Context for knowledge: {context}"""
        system_message += """
        It is very important to output ONE JSON that only contains the fields.
        The instructions, how the json should look like, are further described in the user message.
        """
        return system_message

    def evaluate(
        self,
        question,
        response_A,
        response_B,
        response_C,
        response_D,
        response_E,
        response_F,
        response_G,
        response_H,
        response_I,
        response_J,
        response_K,
        response_L,
        context,
    ) -> EvaluationRankingCorrected:
        system_message = self.get_system_message(
            question,
            response_A,
            response_B,
            response_C,
            response_D,
            response_E,
            response_F,
            response_G,
            response_H,
            response_I,
            response_J,
            response_K,
            response_L,
            context,
        )
        parser = PydanticOutputParser(pydantic_object=EvaluationRankingCorrected)

        user_message = f"""
It is very important to answer with json.
{parser.get_format_instructions()}"""
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=120000,
            temperature=0.1,
            model="deepseek/deepseek-r1:free",
        )
        self.logger.info(response)

        try:
            cleaned_output = response.choices[0].message.content

            print(cleaned_output)
            self.logger.info(f"Response generated: {cleaned_output}")

            fixed_output = self.validate_and_fix_json(cleaned_output)
            return parser.parse(fixed_output)
        except Exception as e:
            self.logger.info(f"Error generating/parsing response: {e}")
            return None

    def validate_and_fix_json(self, json_string):
        try:
            json.loads(json_string)
            return json_string
        except json.JSONDecodeError as e:
            fixed_json = repair_json(json_string)
            return fixed_json

    def get_context(self, qa_file_name: str, markdown_lines: str) -> str:
        def extract_line_numbers(markdown_lines: str) -> tuple[int, int]:
            """Extract start and end line numbers from a string like '0-231'."""
            try:
                start, end = map(int, markdown_lines.split("-"))
                return start, end
            except Exception as e:
                logger.error(f"Error extracting line numbers: {e}")
                return 0, 0

        try:
            markdown_folder = "/output_cleaned_markdown_pipeline"
            markdown_file = qa_file_name.replace("qa_pairs_", "").replace(".json", "")
            markdown_start_line, markdown_end_line = extract_line_numbers(markdown_lines)
            self.logger.info(f"Markdownfolder: {markdown_folder}")
            self.logger.info(f"Markdownfile: {markdown_file}")
            self.logger.info(f"Markdownstartline: {markdown_start_line}")
            self.logger.info(f"Markdownendline: {markdown_end_line}")
            markdown_file_path = os.path.join(markdown_folder, markdown_file)
            with open(markdown_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Extract the relevant lines based on the start and end line numbers
                context_lines = lines[markdown_start_line:markdown_end_line]
                # Join the lines to form the context string
                context = "".join(context_lines)
            return context
        except Exception as e:
            self.logger.error(f"Error reading markdown file: {e}")
            return ""


def evaluate_with_prometheus(logger, question, response, reference_answer, context) -> EvaluationRankingCorrected:
    # huggingface
    api_token = os.environ.get("HF_API_TOKEN")
    # openrouter
    api_token = os.environ.get("OPEN_ROUTER_TOKEN")
    if not api_token:
        raise ValueError("Please set HF_API_TOKEN environment variable")

    evaluator = PrometheusEvaluator(
        api_token,
        logger,
        model="deepseek-ai/DeepSeek-R1",
        output_folder="/evaluation/prometheus",
        score_rubric="0, 1, 2, 3, 4, 5",
        tokencount_limit=4000,
        processing_content="evaluation",
    )
    evaluation = evaluator.evaluate_response(question, response, reference_answer, context)
    return evaluation
