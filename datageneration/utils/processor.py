import os
from typing import List

import tiktoken

from utils.generator import Generator


class Processor:
    def __init__(self, generator: Generator, logger, logging_file: str = "processing.log"):
        self.generator = generator
        self.logger = logger
        self.logging_file = logging_file

    def count_tokens(self, text: str) -> int:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            return 0

    def split_document(self, content: str, chunksize: int = 10) -> List[tuple[str, int, int]]:
        content_lines = content.splitlines()

        sections = []
        current_section = []
        current_start = 0

        for i, line in enumerate(content_lines):
            # new section when header found (# at beginning of line)
            if line.strip().startswith("#"):
                if current_section:
                    sections.append(("".join(current_section), current_start, i))
                    current_section = []
                current_start = i
            current_section.append(line + "\n")

        if current_section:
            sections.append(("".join(current_section), current_start, len(content_lines)))

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = 0

        for section_text, start, end in sections:
            if current_size >= chunksize and current_chunk:
                chunks.append(("\n".join(current_chunk), chunk_start, end))
                current_chunk = []
                current_size = 0
                chunk_start = start

            current_chunk.append(section_text)
            current_size += 1

        if current_chunk:
            chunks.append(("\n".join(current_chunk), chunk_start, len(content_lines)))

        return chunks

    def process_markdown_dir(self, markdown_dir: str, generator: Generator):
        pairs = {}
        for md_file in markdown_dir.rglob("*.md"):
            if self.file_exists(md_file.name):
                self.logger.info(f"Skipping {md_file} as it has already been processed")
                continue
            self.logger.info(f"Processing {md_file}")
            context = self.read_markdown_content(str(md_file))
            if context:
                try:

                    chunksize = 20
                    chunks = [(context, 0, len(context.splitlines()))]

                    while (
                        any(self.count_tokens(chunk[0]) > generator.tokencount_limit for chunk in chunks)
                        and chunksize > 1
                    ):
                        new_chunks = []
                        chunksize = chunksize // 2
                        if chunksize < 1:
                            chunksize = 1
                        self.logger.info(f"Splitting document into chunks of size {chunksize}")
                        for chunk_content, start_line, end_line in chunks:
                            if self.count_tokens(chunk_content) > generator.tokencount_limit:

                                self.logger.info(f"Splitting document into chunks of size {chunksize}")
                                split_chunks = self.split_document(chunk_content, chunksize)
                                for split_content, split_start, split_end in split_chunks:
                                    new_chunks.append(
                                        (split_content, start_line + split_start, start_line + split_end)
                                    )
                            else:
                                new_chunks.append((chunk_content, start_line, end_line))
                        chunks = new_chunks
                    self.logger.info(f"Split document into {len(chunks)} chunks")
                    for chunk_content, start_line, end_line in chunks:
                        chunk_prompt = generator.generate_prompt_from_context(chunk_content)
                        self.logger.info(f"Generated prompt for chunk {start_line}-{end_line}")
                        result = generator.generate_response(chunk_prompt)
                        if result and hasattr(result, self.generator.processing_content):
                            page_range = f"{start_line}-{end_line}"
                            self.logger.info(f"try to save pairs for {page_range}")
                            if self.generator.processing_content == "qa_pairs":
                                pairs[page_range] = result.qa_pairs
                            elif self.generator.processing_content == "multiple_choice_questions":
                                pairs[page_range] = result.multiple_choice_questions
                            elif self.generator.processing_content == "correction_corrupted":
                                pairs[page_range] = result.correction_corrupted
                            self.logger.info(
                                f"Generated {len(pairs[page_range])} {self.generator.processing_content} for {page_range}"
                            )
                            try:
                                generator.save_pairs(pairs, md_file.name)
                            except Exception as e:
                                self.logger.error(f"Error saving question pairs: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing {md_file}: {e}")
                    continue

            if pairs:
                self.logger.success(f"Successfully processed {md_file.name}")
            else:
                self.logger.warning("No valid question pairs generated")
            pairs = {}

    def read_markdown_content(self, markdown_path: str) -> str:
        try:
            with open(markdown_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading markdown file: {e}")
            return ""

    def has_been_processed(self, filename: str) -> bool:
        try:
            if not os.path.exists(self.logging_file):
                return False

            # Use grep to search for filename in log
            result = os.system(f'grep -q "Successfully processed {filename}" {self.logging_file}')
            return result == 0  # grep returns 0 if found, non-zero if not found

        except Exception as e:
            self.logger.error(f"Error checking process status for {filename}: {e}")
            return False

    def file_exists(self, filename_keyword: str) -> bool:
        """Check if a specific file exists in the given directory."""
        for root, _, files in os.walk(self.generator.output_folder):
            for file in files:
                if filename_keyword in file:
                    # TODO: Check if file is not empty and proceed with last line number
                    return True
        return False
