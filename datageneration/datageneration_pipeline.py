import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from multiple_choice_generator import MultipleChoiceGenerator
from q_a_generator import QAGenerator

from data_preparation.clean_markdown import MarkdownCleaner
#from data_preparation.pdf_to_markdown import PDFToMarkdownGenerator
from utils.processor import Processor

load_dotenv()


def create_markdowns(input, output):
    # Process articles
    logging_file_markdown_generator = "/dss/work/toex4699/data_preparation/generate_markdown.log"
    logger.add(logging_file_markdown_generator, format="{time} {level} {message}", level="INFO")
    generator = PDFToMarkdownGenerator(logger, logging_file_markdown_generator)
    generator.process_directory(input, output)


def clean_markdown(input, cleaned):
    logging_file = "/dss/work/toex4699/data_preparation/clean_markdown.log"
    logger.add(logging_file, format="{time} {level} {message}", level="INFO")
    section_titles = [
        "References",
        "Bibliography",
        "Contents",
        "Content",
        "Inhalt",
        "Herausgegeben von",
        "Vorwort",
        "Inhaltsverzeichnis",
        "Table of Contents",
        "List of Contributors",
        "List of Editors",
        "List of Abbreviations",
        "Über den Autor",
        "Empfohlene Literatur",
        "Über die Herausgeber",
        "Über die Autoren",
        "Korrespondenz",
        "Literatur",
        "Contributors",
        "Editors",
        "Index",
        "Abbreviations",
        "Anhang",
        "Stichwortverzeichnis",
        "Foreword",
        "Preface",
        "Acknowledgements",
        "About the Author",
        "Recommended Reading",
        "About the Editors",
        "About the Contributors",
        "Correspondence",
        "Abstract",
        "Appendix",
        "Keywords",
        "Abbreviations",
        "Glossary",
        "List of Figures",
        "List of Tables",
        "Author",
        "Acknowledgments",
        "Ethics Statement",
        " Disclosure Statement",
        "Funding Information",
        "Referenzen",
        "Bibliographie",
        "Inhalt",
        "Inhaltsverzeichnis",
        "Verzeichnis der Mitwirkenden",
        "Verzeichnis der Herausgeber",
        "Abkürzungsverzeichnis",
        "Mitwirkende",
        "Herausgeber",
        "Index",
        "Abkürzungen",
        "Vorwort",
        "Vorwort",
        "Danksagung",
        "Über den Autor",
        "Empfohlene Lektüre",
        "Über die Herausgeber",
        "Über die Mitwirkenden",
        "Korrespondenz",
        "Zusammenfassung",
        "Anhang",
        "Schlüsselwörter",
        "Abkuerzungen",
        "Glossar",
        "Abbildungsverzeichnis",
        "Tabellenverzeichnis",
        "Autor",
        "Danksagungen",
        "Ethikerklärung",
        "Offenlegungserklärung",
        "Finanzierungsinformationen",
    ]
    cleaner = MarkdownCleaner(logger, section_titles, logging_file)
    cleaner.clean_and_merge_markdowns(f"{input}", cleaned)


def generate_qa_pairs(cleaned, output):
    logging_file = "/dss/work/toex4699/english_datasets/generate_qa.log"
    logger.add(logging_file, format="{time} {level} {message}", level="INFO")
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("Please set HF_API_TOKEN environment variable")

    generator = QAGenerator(api_token, logger, "qa_pairs", output)
    processor = Processor(generator, logger, logging_file)
    markdown_dir = Path(cleaned)

    processor.process_markdown_dir(markdown_dir, generator)


def generate_multiple_choices(cleaned, output):
    logging_file = "/dss/work/toex4699/english_datasets/generate_multiple_choices.log"
    logger.add(logging_file, format="{time} {level} {message}", level="INFO")
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        raise ValueError("Please set HF_API_TOKEN environment variable")

    generator = MultipleChoiceGenerator(api_token, logger, "multiple_choice_questions", output)
    processor = Processor(generator, logger, logging_file)
    markdown_dir = Path(cleaned)

    processor.process_markdown_dir(markdown_dir, generator)


if __name__ == "__main__":
    # Please specify the paths according to your environment
    output = "/dss/work/toex4699/output_markdown_pipeline"
    cleaned = f"/dss/work/toex4699/output_cleaned_markdown_pipeline"
    input = "/nfs/group/agaifh/datasets/texts_ecg"
    qa_pairs_folder = "/dss/work/toex4699/english_datasets/q_a_pairs"
    multiple_choices_folder = "/dss/work/toex4699/english_datasets/multiple_choices"

  #  create_markdowns(input, output)
   # clean_markdown(output, cleaned)
    generate_qa_pairs(cleaned, qa_pairs_folder)
    generate_multiple_choices(cleaned, multiple_choices_folder)
