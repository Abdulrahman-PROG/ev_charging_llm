import sys
sys.path.append('/app')
import json
import pandas as pd
import torch
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import mlflow
import time
from common.logging import setup_logging
from common.utils import format_prompt, generate_response
from common.config import config
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = setup_logging(__name__)

# Download NLTK data
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup MLflow
mlflow.set_tracking_uri(config.Training.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.Training.MLFLOW_EXPERIMENT_NAME)
mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
mlflow.log_params(config.get_config_summary())

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Initialize sentence transformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Sentence transformer loaded")

# Load models
quantization_config = None
if config.Model.QUANTIZATION_ENABLED:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.Model.QUANTIZATION_TYPE,
        bnb_4bit_compute_dtype=torch.float32  # Use float32 for CPU compatibility
    )

logger.info("Loading base model...")
base_tokenizer = AutoTokenizer.from_pretrained(config.Model.BASE_MODEL_NAME, token=config.Model.HF_TOKEN)
base_model = AutoModelForCausalLM.from_pretrained(
    config.Model.BASE_MODEL_NAME,
    quantization_config=quantization_config,
    device_map="cpu" if not torch.cuda.is_available() else "auto",
    token=config.Model.HF_TOKEN
)
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

logger.info("Loading fine-tuned model...")
model_loaded = True
try:
    ft_tokenizer = AutoTokenizer.from_pretrained(config.Model.FINAL_MODEL_PATH)
    ft_model = AutoModelForCausalLM.from_pretrained(
        config.Model.FINAL_MODEL_PATH,
        quantization_config=quantization_config,
        device_map="cpu" if not torch.cuda.is_available() else "auto"
    )
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token
except Exception as e:
    logger.error(f"Could not load fine-tuned model: {e}")
    ft_model = base_model
    ft_tokenizer = base_tokenizer
    model_loaded = False

def create_ev_benchmark():
    """Create EV charging domain benchmark questions"""
    # Fallback benchmark questions since DOMAIN_QUESTIONS is not in config.py
    default_benchmark = [
        {
            "question": "What are the different types of EV charging stations?",
            "expected_keywords": ["Level 1", "Level 2", "DC fast charging", "Tesla Supercharger"]
        },
        {
            "question": "How long does it take to charge an electric vehicle at home?",
            "expected_keywords": ["Level 1", "Level 2", "hours", "overnight"]
        },
        {
            "question": "What is the cost of installing a home EV charging station?",
            "expected_keywords": ["installation", "cost", "Level 2", "electrician"]
        }
    ]
    logger.info("Using default EV benchmark questions")
    return default_benchmark

def measure_inference_metrics(model, tokenizer, prompts, num_runs=5):
    """Measure inference latency and throughput"""
    latencies = []
    for prompt in prompts:
        start_time = time.time()
        for _ in range(num_runs):
            generate_response(model, tokenizer, prompt)
        latency = (time.time() - start_time) / num_runs
        latencies.append(latency)
    throughput = len(prompts) * num_runs / sum(latencies) if latencies else 0
    return {"avg_latency": np.mean(latencies) if latencies else 0, "throughput": throughput}

def calculate_rouge_scores(reference, hypothesis):
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(config.Evaluation.ROUGE_METRICS, use_stemmer=True)
    return scorer.score(reference, hypothesis)

def calculate_bleu_score(reference, hypothesis):
    """Calculate BLEU score"""
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing)

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity"""
    embeddings = sentence_model.encode([text1, text2])
    return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

def keyword_coverage(text, keywords):
    """Calculate keyword coverage"""
    text_lower = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in text_lower) / len(keywords) if keywords else 0

# Load test data
test_data_path = os.path.join(config.OUTPUT_DIR, "ev_training_alpaca.json")
try:
    with open(test_data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    test_size = min(config.Evaluation.TEST_SET_SIZE, len(all_data) // 5)
    test_data = all_data[-test_size:] if all_data else []
    if not test_data:
        logger.warning("No test data available in ev_training_alpaca.json")
except Exception as e:
    logger.error(f"Failed to load test data: {e}")
    test_data = []

# Create domain benchmark
domain_benchmark = create_ev_benchmark()
logger.info(f"Created benchmark with {len(domain_benchmark)} questions")
logger.info(f"Using {len(test_data)} test examples")

# Evaluate on domain-specific benchmark
domain_results = []
prompts = [format_prompt(item["question"]) for item in domain_benchmark]
base_metrics = measure_inference_metrics(base_model, base_tokenizer, prompts)
ft_metrics = measure_inference_metrics(ft_model, ft_tokenizer, prompts) if model_loaded else base_metrics
mlflow.log_metrics({
    "base_avg_latency": base_metrics["avg_latency"],
    "base_throughput": base_metrics["throughput"],
    "ft_avg_latency": ft_metrics["avg_latency"],
    "ft_throughput": ft_metrics["throughput"]
})

for i, item in enumerate(domain_benchmark):
    question = item["question"]
    prompt = format_prompt(question)
    base_response = generate_response(base_model, base_tokenizer, prompt)
    ft_response = generate_response(ft_model, ft_tokenizer, prompt) if model_loaded else base_response
    result = {
        "question_id": i + 1,
        "question": question,
        "base_response": base_response,
        "ft_response": ft_response,
        "base_keyword_coverage": keyword_coverage(base_response, item["expected_keywords"]),
        "ft_keyword_coverage": keyword_coverage(ft_response, item["expected_keywords"]),
        "semantic_similarity": calculate_semantic_similarity(base_response, ft_response)
    }
    domain_results.append(result)

# Evaluate on test set
test_results = []
for i, item in enumerate(test_data[:config.Evaluation.TEST_SET_SIZE]):
    instruction = item["instruction"]
    reference = item["output"]
    prompt = format_prompt(instruction)
    base_response = generate_response(base_model, base_tokenizer, prompt)
    ft_response = generate_response(ft_model, ft_tokenizer, prompt) if model_loaded else base_response
    base_rouge = calculate_rouge_scores(reference, base_response)
    ft_rouge = calculate_rouge_scores(reference, ft_response) if model_loaded else base_rouge
    result = {
        "test_id": i + 1,
        "instruction": instruction,
        "reference": reference,
        "base_response": base_response,
        "ft_response": ft_response,
        "base_rouge1": base_rouge['rouge1'].fmeasure,
        "base_rouge2": base_rouge['rouge2'].fmeasure,
        "base_rougeL": base_rouge['rougeL'].fmeasure,
        "ft_rouge1": ft_rouge['rouge1'].fmeasure,
        "ft_rouge2": ft_rouge['rouge2'].fmeasure,
        "ft_rougeL": ft_rouge['rougeL'].fmeasure,
        "base_bleu": calculate_bleu_score(reference, base_response),
        "ft_bleu": calculate_bleu_score(reference, ft_response) if model_loaded else calculate_bleu_score(reference, base_response),
        "base_semantic": calculate_semantic_similarity(reference, base_response),
        "ft_semantic": calculate_semantic_similarity(reference, ft_response) if model_loaded else calculate_semantic_similarity(reference, base_response)
    }
    test_results.append(result)

# Save results
os.makedirs(config.RESULTS_DIR, exist_ok=True)
domain_df = pd.DataFrame(domain_results)
test_df = pd.DataFrame(test_results)
domain_df.to_csv(os.path.join(config.RESULTS_DIR, 'domain_benchmark_results.csv'), index=False, encoding='utf-8')
test_df.to_csv(os.path.join(config.RESULTS_DIR, 'test_set_results.csv'), index=False, encoding='utf-8')

# Log metrics to MLflow
mlflow.log_metrics({
    "avg_keyword_coverage_base": float(domain_df['base_keyword_coverage'].mean()) if domain_results else 0,
    "avg_keyword_coverage_ft": float(domain_df['ft_keyword_coverage'].mean()) if domain_results else 0,
    "avg_rouge1_base": float(test_df['base_rouge1'].mean()) if test_results else 0,
    "avg_rouge1_ft": float(test_df['ft_rouge1'].mean()) if test_results else 0,
    "avg_rougeL_base": float(test_df['base_rougeL'].mean()) if test_results else 0,
    "avg_rougeL_ft": float(test_df['ft_rougeL'].mean()) if test_results else 0,
    "avg_bleu_base": float(test_df['base_bleu'].mean()) if test_results else 0,
    "avg_bleu_ft": float(test_df['ft_bleu'].mean()) if test_results else 0
})

# Create visualizations
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# Keyword Coverage Comparison
if domain_results:
    coverage_data = {
        'Base Model': domain_df['base_keyword_coverage'].tolist(),
        'Fine-tuned Model': domain_df['ft_keyword_coverage'].tolist()
    }
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df.boxplot(ax=axes[0,0])
    axes[0,0].set_title('Keyword Coverage Comparison')
    axes[0,0].set_ylabel('Coverage Score')

# ROUGE Scores Comparison
if test_results:
    rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    base_rouge = [test_df['base_rouge1'].mean(), test_df['base_rouge2'].mean(), test_df['base_rougeL'].mean()]
    ft_rouge = [test_df['ft_rouge1'].mean(), test_df['ft_rouge2'].mean(), test_df['ft_rougeL'].mean()]
    x = np.arange(len(rouge_metrics))
    width = 0.35
    axes[0,1].bar(x - width/2, base_rouge, width, label='Base Model', alpha=0.8)
    axes[0,1].bar(x + width/2, ft_rouge, width, label='Fine-tuned Model', alpha=0.8)
    axes[0,1].set_title('ROUGE Scores Comparison')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(rouge_metrics)
    axes[0,1].legend()

# BLEU and Semantic Similarity
if test_results:
    metrics = ['BLEU', 'Semantic Similarity']
    base_scores = [test_df['base_bleu'].mean(), test_df['base_semantic'].mean()]
    ft_scores = [test_df['ft_bleu'].mean(), test_df['ft_semantic'].mean()]
    axes[1,0].bar(x - width/2, base_scores, width, label='Base Model', alpha=0.8)
    axes[1,0].bar(x + width/2, ft_scores, width, label='Fine-tuned Model', alpha=0.8)
    axes[1,0].set_title('BLEU and Semantic Similarity')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(metrics)
    axes[1,0].legend()

# Improvement Summary
if test_results and domain_results:
    improvements = {
        'Keyword Coverage': domain_df['ft_keyword_coverage'].mean() - domain_df['base_keyword_coverage'].mean(),
        'ROUGE-1': test_df['ft_rouge1'].mean() - test_df['base_rouge1'].mean(),
        'ROUGE-L': test_df['ft_rougeL'].mean() - test_df['base_rougeL'].mean(),
        'BLEU': test_df['ft_bleu'].mean() - test_df['base_bleu'].mean(),
        'Semantic Sim': test_df['ft_semantic'].mean() - test_df['base_semantic'].mean()
    }
    axes[1,1].bar(list(improvements.keys()), list(improvements.values()), color=['green' if v > 0 else 'red' for v in improvements.values()])
    axes[1,1].set_title('Performance Improvements')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
output_plot = os.path.join(config.RESULTS_DIR, 'evaluation_results.png')
plt.savefig(output_plot, dpi=300)
mlflow.log_artifact(output_plot)
mlflow.end_run()
logger.info("Evaluation completed successfully")