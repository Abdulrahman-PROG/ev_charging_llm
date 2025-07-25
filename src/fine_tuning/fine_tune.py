import sys
sys.path.append('/app')
import os
import json
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import mlflow
from common.logging import setup_logging
from common.utils import format_prompt
from common.config import config
import numpy as np
from rouge_score import rouge_scorer

# Setup logging
logger = setup_logging(__name__)

# Configuration
CONFIG = {
    "data_path": os.path.join(config.DataCollection.OUTPUT_DIR, "ev_training_alpaca.json"),  # Fixed
    "output_dir": config.Model.FINAL_MODEL_PATH,  # Use FINAL_MODEL_PATH directly
    "final_model_path": config.Model.FINAL_MODEL_PATH,
    "model_name": config.Model.BASE_MODEL_NAME,
    "max_length": config.Model.GENERATION_MAX_LENGTH,  # Use GENERATION_MAX_LENGTH
    "train_batch_size": config.Model.TRAIN_BATCH_SIZE,
    "eval_batch_size": config.Model.EVAL_BATCH_SIZE,
    "learning_rate": config.Model.LEARNING_RATE,
    "num_epochs": config.Model.NUM_EPOCHS,
    "warmup_steps": config.Model.WARMUP_STEPS,
    "lora_r": config.Model.LORA_RANK,  # Use LORA_RANK
    "lora_alpha": config.Model.LORA_ALPHA,
    "lora_dropout": config.Model.LORA_DROPOUT,
    "experiment_name": config.MLFLOW_EXPERIMENT_NAME  # Fixed: Use MLFLOW_EXPERIMENT_NAME directly
}

# Setup MLflow
mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
mlflow.start_run(run_name=CONFIG["experiment_name"])
mlflow.log_params(CONFIG)

def load_data():
    """Load and preprocess training data"""
    try:
        with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            logger.error("No data found in ev_training_alpaca.json")
            raise ValueError("No data found in ev_training_alpaca.json")
        dataset = Dataset.from_list(data)
        # Use fixed split ratios since config.Training.TRAIN_SPLIT doesn't exist
        train_size = int(len(dataset) * 0.8)  # 80% for training
        val_size = int(len(dataset) * 0.2)    # 20% for validation
        train_data = dataset.select(range(train_size))
        val_data = dataset.select(range(train_size, train_size + val_size))
        logger.info(f"Loaded {len(train_data)} training and {len(val_data)} validation examples")
        return train_data, val_data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise e

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset"""
    prompts = [format_prompt(ex["instruction"], ex.get("input", "")) for ex in examples]
    outputs = [ex["output"] for ex in examples]
    inputs = [p + o for p, o in zip(prompts, outputs)]
    tokenized = tokenizer(inputs, max_length=CONFIG["max_length"], truncation=True, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def compute_metrics(eval_pred):
    """Compute ROUGE metrics for evaluation"""
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    metrics = {
        "rouge1": np.mean([score['rouge1'].fmeasure for score in rouge_scores]),
        "rougeL": np.mean([score['rougeL'].fmeasure for score in rouge_scores])
    }
    mlflow.log_metrics(metrics)
    return metrics

def main():
    """Main fine-tuning function"""
    logger.info("Starting fine-tuning process...")

    # Check if quantization is enabled
    quantization_config = None
    if config.Model.QUANTIZATION_ENABLED:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.Model.QUANTIZATION_TYPE,
            bnb_4bit_compute_dtype=torch.float32  # Use float32 for CPU compatibility
        )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], token=config.Model.HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=quantization_config,
        device_map="cpu" if not torch.cuda.is_available() else "auto",
        token=config.Model.HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["q_proj", "v_proj"]  # Fixed: Define target modules explicitly
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")

    # Load data
    train_data, val_data = load_data()

    # Tokenize datasets
    train_dataset = train_data.map(lambda x: tokenize_function([x], tokenizer), batched=False)
    val_dataset = val_data.map(lambda x: tokenize_function([x], tokenizer), batched=False)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=100,  # Fixed: Use default value
        evaluation_strategy="steps",
        eval_steps=500,     # Fixed: Use default value
        save_steps=500,     # Fixed: Use default value
        save_total_limit=2, # Fixed: Use default value
        weight_decay=0.01,  # Fixed: Use default value
        max_grad_norm=1.0,  # Fixed: Use default value
        fp16=False,         # Disable fp16 for CPU compatibility
        report_to="mlflow"
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()
    mlflow.log_artifact(CONFIG["output_dir"])

    # Save final model
    trainer.save_model(CONFIG["final_model_path"])
    tokenizer.save_pretrained(CONFIG["final_model_path"])
    logger.info(f"Model saved to {CONFIG['final_model_path']}")

    # Log model to MLflow
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

if __name__ == "__main__":
    main()