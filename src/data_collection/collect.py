import os
import json
import time
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import sys
import nltk
from nltk.tokenize import sent_tokenize


sys.path.append('/app')

from common.logging import setup_logging
from common.config import config
from common.utils import format_prompt


# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup logging
logger = setup_logging(__name__)

class DataCollector:
    """Class for collecting and processing EV charging data"""
    
    def __init__(self):
        self.data = []
        self.qa_pairs = []
        logger.info("DataCollector initialized")
    
    def extract_pdf_text(self, pdf_path: str) -> list:
        """Extract text and tables from PDF files"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                data = []
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    table_text = "\n".join([" | ".join(row) for table in tables for row in table]) if tables else ""
                    combined_text = f"{text}\n{table_text}".strip()
                    if combined_text:
                        data.append({
                            'source': 'pdf',
                            'filename': os.path.basename(pdf_path),
                            'page': page_num,
                            'text': combined_text,
                            'length': len(combined_text),
                            'timestamp': datetime.now().isoformat()
                        })
                logger.info(f"Extracted {len(data)} pages from {pdf_path}")
                return data
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
            return []
    
    def scrape_webpage(self, url: str) -> dict:
        """Scrape text from a webpage"""
        try:
            headers = {'User-Agent': config.DataCollection.USER_AGENT}
            response = requests.get(url, headers=headers, timeout=config.DataCollection.REQUEST_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title').get_text() if soup.find('title') else url
            main_content = soup.find('article') or soup.find('div', class_=['content', 'main'])
            text = main_content.get_text(separator=' ', strip=True) if main_content else soup.get_text(separator=' ', strip=True)
            if text and len(text.strip()) > 0:
                data = {
                    'source': 'web',
                    'url': url,
                    'title': title,
                    'text': text,
                    'length': len(text),
                    'timestamp': datetime.now().isoformat()
                }
                logger.info(f"Scraped {len(text)} characters from {url}")
                return data
            logger.warning(f"No content extracted from {url}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def is_relevant_content(self, text: str) -> bool:
        """Check if content is relevant to EV charging"""
        if not text or not isinstance(text, str):
            return False
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in config.DataCollection.EV_KEYWORDS if keyword.lower() in text_lower)
        relevance_score = keyword_count / len(config.DataCollection.EV_KEYWORDS) if config.DataCollection.EV_KEYWORDS else 0
        is_valid = (
            len(text) >= config.DataCollection.MIN_TEXT_LENGTH and 
            len(text) <= config.DataCollection.MAX_TEXT_LENGTH and 
            relevance_score >= config.DataCollection.QUALITY_THRESHOLD
        )
        logger.debug(f"Relevance check: length={len(text)}, score={relevance_score}, valid={is_valid}")
        return is_valid
    
    def collect_from_pdfs(self):
        """Collect data from PDF files in the specified folder"""
        pdf_folder = config.DataCollection.PDF_FOLDER
        if not os.path.exists(pdf_folder):
            logger.warning(f"PDF folder {pdf_folder} does not exist")
            return
        
        pdf_count = 0
        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_folder, filename)
                logger.info(f"Processing PDF: {pdf_path}")
                pdf_data = self.extract_pdf_text(pdf_path)
                for item in pdf_data:
                    if self.is_relevant_content(item['text']):
                        self.data.append(item)
                pdf_count += 1
        logger.info(f"Processed {pdf_count} PDFs")
    
    def collect_from_web(self):
        """Collect data from specified web URLs"""
        url_count = 0
        for url in config.DataCollection.WEB_SCRAPING_URLS:
            if url.strip():
                logger.info(f"Scraping URL: {url}")
                web_data = self.scrape_webpage(url)
                if web_data and self.is_relevant_content(web_data['text']):
                    self.data.append(web_data)
                url_count += 1
                time.sleep(config.DataCollection.DELAY_BETWEEN_REQUESTS)
        logger.info(f"Processed {url_count} URLs")
    
    def generate_qa_pairs(self):
        """Generate Q&A pairs from collected data"""
        if not self.data:
            logger.warning("No data available to generate Q&A pairs")
            return
        
        for item in self.data:
            text = item['text']
            sentences = sent_tokenize(text)
            for i in range(len(sentences) - 1):
                question = sentences[i].strip()
                if question.endswith('.'):
                    question = question[:-1] + '?'
                answer = sentences[i + 1].strip()
                if len(question) > 10 and len(answer) > 20 and self.is_relevant_content(question + answer):
                    qa_pair = {
                        'instruction': question,
                        'output': answer,
                        'input': f"Source: {item['source']}, {item.get('filename', item.get('url', ''))}"
                    }
                    self.qa_pairs.append(qa_pair)
        logger.info(f"Generated {len(self.qa_pairs)} Q&A pairs")
    
    def save_data(self):
        """Save collected data and Q&A pairs"""
        try:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            
            # Save raw data
            if self.data:
                output_csv = os.path.join(config.OUTPUT_DIR, 'ev_data.csv')
                pd.DataFrame(self.data).to_csv(output_csv, index=False, encoding='utf-8')
                logger.info(f"Saved {len(self.data)} data items to {output_csv}")
            else:
                logger.warning("No data to save to ev_data.csv")
            
            # Save Q&A pairs
            if self.qa_pairs:
                output_json = os.path.join(config.OUTPUT_DIR, 'ev_training_alpaca.json')
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(self.qa_pairs)} Q&A pairs to {output_json}")
            else:
                logger.warning("No Q&A pairs to save to ev_training_alpaca.json")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

def main():
    """Main function to run data collection"""
    logger.info("Starting data collection process...")
    collector = DataCollector()
    
    # Collect data
    collector.collect_from_pdfs()
    collector.collect_from_web()
    
    # Generate Q&A pairs
    collector.generate_qa_pairs()
    
    # Save results
    collector.save_data()
    
    logger.info(f"Collected {len(collector.data)} data items and generated {len(collector.qa_pairs)} Q&A pairs")

if __name__ == "__main__":
    main()