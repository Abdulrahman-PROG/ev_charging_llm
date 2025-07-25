import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
import mlflow
from datetime import datetime

# Add /app to sys.path
sys.path.append('/app')

from common.logging import setup_logging
from common.config import config
from common.utils import format_prompt

# Download NLTK data
nltk.download('punkt')

# Setup logging
logger = setup_logging(__name__)

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {url}: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ' '.join([page.extract_text() or '' for page in pdf.pages])
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def generate_qa_pairs(text):
    sentences = sent_tokenize(text)
    qa_pairs = []
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) >= config.DataCollection.MIN_TEXT_LENGTH:
            question = f"What is the significance of '{sentence[:50]}...' in EV charging?"
            qa_pairs.append({
                "instruction": question,
                "output": sentence,
                "input": ""
            })
    return qa_pairs

def collect_data():
    with mlflow.start_run(run_name=f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        data = []
        urls = config.DataCollection.WEB_SCRAPING_URLS
        pdf_dir = config.DataCollection.PDF_DIR

        # Extract from URLs
        for url in urls:
            logger.info(f"Extracting data from {url}")
            text = extract_text_from_url(url)
            if text:
                data.append({"source": url, "text": text, "type": "web"})

        # Extract from PDFs
        if os.path.exists(pdf_dir):
            for pdf_file in os.listdir(pdf_dir):
                if pdf_file.endswith('.pdf'):
                    pdf_path = os.path.join(pdf_dir, pdf_file)
                    logger.info(f"Extracting data from {pdf_path}")
                    text = extract_text_from_pdf(pdf_path)
                    if text:
                        data.append({"source": pdf_path, "text": text, "type": "pdf"})

        # Save raw data
        df = pd.DataFrame(data)
        output_csv = os.path.join(config.DataCollection.OUTPUT_DIR, "ev_data.csv")
        df.to_csv(output_csv, index=False)
        mlflow.log_artifact(output_csv)

        # Generate Q&A pairs
        qa_pairs = []
        for _, row in df.iterrows():
            qa_pairs.extend(generate_qa_pairs(row['text']))

        # Save Q&A pairs
        output_json = os.path.join(config.DataCollection.OUTPUT_DIR, "ev_training_alpaca.json")
        pd.DataFrame(qa_pairs).to_json(output_json, orient='records', lines=True)
        mlflow.log_artifact(output_json)

        logger.info(f"Collected {len(data)} sources and generated {len(qa_pairs)} Q&A pairs")
        mlflow.log_metric("num_sources", len(data))
        mlflow.log_metric("num_qa_pairs", len(qa_pairs))

if __name__ == "__main__":
    collect_data()