import json
import os
from typing import List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from json_repair import repair_json
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from pydantic import BaseModel

load_dotenv()


class Generator:
    def __init__(
        self,
        api_token: str,
        logger,
        processing_content: str,
        output_folder: str,
        tokencount_limit: int = 50000,
        model="meta-llama/Llama-3.3-70B-Instruct",
        use_huggingface=True,
    ):
        self.use_huggingface = use_huggingface
        if use_huggingface:
            if model == "deepseek-ai/DeepSeek-R1":
                self.client = InferenceClient(model=model, token=api_token, timeout=600, provider="together")
            else:
                self.client = InferenceClient(model=model, token=api_token, timeout=600)
        
        self.logger = logger
        self.tokencount_limit = tokencount_limit
        self.processing_content = processing_content
        self.output_folder = output_folder

    def validate_and_fix_json(self, json_string):
        try:
            json.loads(json_string)
            return json_string
        except json.JSONDecodeError as e:
            fixed_json = repair_json(json_string)
            return fixed_json

    def convert_messages(self, messages):

        role_map = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        return [role_map[m["role"]](content=m["content"]) for m in messages]

    def generate_response(self, messages: List) -> BaseModel:
        try:
            if self.use_huggingface:
                response = self.client.chat_completion(messages, max_tokens=50000, temperature=0.1)
                self.logger.info("Response generated")
                result = response.choices[0].message.content
            else:
                converted_messages = self.convert_messages(messages)
                response = self.client(converted_messages)
                result = response.content
                self.logger.info("Response generated")

            fixed_json = self.validate_and_fix_json(result)
            return self.parser.parse(fixed_json)
        except Exception as e:
            self.logger.error(f"Error generating/parsing response: {e}")
            return None

    def save_pairs(self, pairs: dict, filename: str):
        raise NotImplementedError("Method save_pairs not implemented")
