import os
import json
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import mlflow
from common.logging import setup_logging
from common.utils import format_prompt
from config import config
import numpy as np
from rouge_score import rouge_scorer

# Setup logging
logger = setup_logging(__name__)

# Configuration
CONFIG = {
    "data_path": os.path.join(config.OUTPUT_DIR, "ev_training_alpaca.json"),
    "output_dir": config.Model.FINE_TUNED_MODEL_PATH,
    "final_model_path": config.Model.FINAL_MODEL_PATH,
    "model_name": config.Model.BASE_MODEL_NAME,
    "max_length": config.Model.MAX_LENGTH,
    "train_batch_size": config.Model.TRAIN_BATCH_SIZE,
    "eval_batch_size": config.Model.EVAL_BATCH_SIZE,
    "learning_rate": config.Model.LEARNING_RATE,
    "num_epochs": config.Model.NUM_EPOCHS,
    "warmup_steps": config.Model.WARMUP_STEPS,
    "lora_r": config.Model.LORA_R,
    "lora_alpha": config.Model.LORA_ALPHA,
    "lora_dropout": config.Model.LORA_DROPOUT,
    "experiment_name": config.Training.EXPERIMENT_NAME
}

# Setup MLflow
mlflow.set_tracking_uri(config.Training.MLFLOW_TRACKING_URI)
mlflow.set_experiment(config.Training.MLFLOW_EXPERIMENT_NAME)
mlflow.start_run(run_name=CONFIG["experiment_name"])
mlflow.log_params(CONFIG)

def load_data():
    """Load and preprocess training data"""
    try:
        with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
        train_size = int(len(dataset) * config.Training.TRAIN_SPLIT)
        val_size = int(len(dataset) * config.Training.VAL_SPLIT)
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

    # Load data
    train_data, val_data = load_data()

    # Load tokenizer and model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.Model.QUANTIZATION_TYPE,
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], token=config.Model.HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
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
        target_modules=config.Model.LORA_TARGET_MODULES
    )
    model = get_peft_model(model, lora_config)
    logger.info(f"Model loaded with {model.num_parameters():,} parameters")

    # Tokenize datasets
    train_dataset = train_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=config.Training.LOGGING_STEPS,
        evaluation_strategy="steps",
        eval_steps=config.Training.EVAL_STEPS,
        save_steps=config.Training.SAVE_STEPS,
        save_total_limit=config.Training.SAVE_TOTAL_LIMIT,
        weight_decay=config.Training.WEIGHT_DECAY,
        max_grad_norm=config.Training.MAX_GRAD_NORM,
        fp16=config.Training.FP16,
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