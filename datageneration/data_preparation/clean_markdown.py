import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List


class MarkdownCleaner:
    def __init__(self, logger, section_titles: List[str], logging_file: str):
        self.logger = logger
        self.section_titles = section_titles
        self.logging_file = logging_file

    def clean_and_merge_markdowns(self, input_dir: str, output_dir: str) -> None:

        # Group files by name
        files = defaultdict(list)
        page_pattern = re.compile(r"_pages_(\d+)-(\d+)\.md$")

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for md_file in input_path.rglob("*.md"):
            match = page_pattern.search(md_file.name)
            if match:
                file_name = md_file.name[: md_file.name.find("_pages_")]
                start_page = int(match.group(1))
                files[file_name].append((start_page, md_file))
            else:
                # Handle articles (single files)
                file_name = md_file.stem
                files[file_name].append((0, md_file))

        # Process each file
        for file_name, chunks in files.items():
            if self.has_been_processed(file_name):
                self.logger.info(f"Skipping {file_name} as it has been processed yet.")
                continue
            if chunks:
                self.logger.info(f"Processing {file_name}")
                sorted_chunks = sorted(chunks, key=lambda x: x[0])
                combined_content = []

                # Clean and combine content
                for _, chunk_path in sorted_chunks:
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    for title in self.section_titles:
                        if title.lower() == "index":
                            # Match main index header and all single-letter subsections
                            pattern = rf"([#]{{1,3}} *{title}.*?(?:# *[A-Z]\s*(?:.*?))*?)(?=# (?![A-Z]$)|\Z)"
                        else:
                            # Match headers with 1-3 # characters
                            pattern = rf"([#]{{1,3}} *{title}[^#]*?)(?=[#]{{1,3}}|\Z)"
                        content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.DOTALL)

                    combined_content.append(content.strip())

                output_file = output_path / f"{file_name}_cleaned.md"
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(combined_content))

                self.logger.success(f"Created cleaned merged file: {output_file}")
                self.logger.success(f"Successfully processed {file_name}")

    def has_been_processed(self, filename: str) -> bool:
        try:
            if not os.path.exists(self.logging_file):
                return False

            result = os.system(f'grep -q "Successfully processed {filename}" {self.logging_file}')
            return result == 0  # grep returns 0 if found, non-zero if not found

        except Exception as e:
            self.logger.error(f"Error checking process status for {filename}: {e}")
            return False
