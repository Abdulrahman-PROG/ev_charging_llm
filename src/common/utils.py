import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import config

def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format the prompt for the model"""
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

def generate_response(model, tokenizer, prompt: str, max_length: int = None, temperature: float = None) -> str:
    """Generate response using the given model"""
    max_length = max_length or config.Model.GENERATION_MAX_LENGTH
    temperature = temperature or config.Model.GENERATION_TEMPERATURE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=min(max_length, inputs.shape[1] + 150),
            num_return_sequences=1,
            temperature=temperature,
            top_p=config.Model.GENERATION_TOP_P,
            top_k=config.Model.GENERATION_TOP_K,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    return response