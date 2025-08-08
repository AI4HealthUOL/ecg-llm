import os
from pathlib import Path
from time import time

import fitz  # PyMuPDF for page counting
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.pipe.OCRPipe import OCRPipe


class PDFToMarkdownGenerator:
    def __init__(self, logger, logging_file):
        self.logger = logger
        self.logging_file = logging_file

    def get_pdf_page_count(self, pdf_path: str) -> int:
        try:
            doc = fitz.open(pdf_path)
            count = doc.page_count
            doc.close()
            return count
        except Exception as e:
            self.logger.error(f"Error getting page count for {pdf_path}: {e}")
            return 0

    def process_pdf_chunk(
        self, pdf_bytes, start_page: int, end_page: int, image_writer, md_writer, output_name: str
    ):
        pipe = OCRPipe(
            pdf_bytes, model_list=[], image_writer=image_writer, start_page_id=start_page, end_page_id=end_page
        )

        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()

        image_dir = "images"
        md_content = pipe.pipe_mk_markdown(image_dir, drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD)

        if isinstance(md_content, list):
            md_writer.write_string(output_name, "\n".join(md_content))
        else:
            md_writer.write_string(output_name, md_content)

    def process_single_pdf(self, pdf_path: str, output_base: str, file) -> None:
        try:
            local_image_dir = os.path.join(output_base, "images")
            os.makedirs(local_image_dir, exist_ok=True)

            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(output_base)
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(pdf_path)
            page_count = self.get_pdf_page_count(pdf_path)
            if page_count > 300:

                chunk_size = 300

                for chunk_start in range(0, page_count, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, page_count)
                    chunk_name = f"{Path(pdf_path).stem}_pages_{chunk_start+1}-{chunk_end}.md"

                    start_time = time()
                    self.logger.info(f"Processing {pdf_path} pages {chunk_start+1}-{chunk_end}")

                    self.process_pdf_chunk(pdf_bytes, chunk_start, chunk_end, image_writer, md_writer, chunk_name)
                    self.logger.info(f"Chunk processing time: {time() - start_time:.2f} seconds")
            else:
                start_time = time()
                self.process_pdf_chunk(pdf_bytes, 0, 500, image_writer, md_writer, f"{Path(pdf_path).stem}.md")
                self.logger.info(f"Processing time: {time() - start_time:.2f} seconds")
            self.logger.info(f"Successfully processed {file}")
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {e}")

    def process_directory(self, input_dir: str, output_base: str) -> None:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    if not self.has_been_processed(file):
                        pdf_path = os.path.join(root, file)
                        rel_path = os.path.relpath(root, input_dir)
                        output_dir = os.path.join(output_base, rel_path)
                        os.makedirs(output_dir, exist_ok=True)
                        self.process_single_pdf(pdf_path, output_dir, file)

                    else:
                        self.logger.info(f"Skipping already processed file: {file}")

    def has_been_processed(self, filename: str) -> bool:
        # check if the file has been processed by looking in logging file
        try:
            if not os.path.exists(self.logging_file):
                return False

            # Use grep to search for filename in log
            result = os.system(f'grep -q "Successfully processed {filename}" {self.logging_file}')
            return result == 0  # grep returns 0 if found, non-zero if not found

        except Exception as e:
            self.logger.error(f"Error checking process status for {filename}: {e}")
            return False
