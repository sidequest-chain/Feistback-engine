import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

model_name = "Qwen/Qwen2.5-Coder-0.5B"
print(f"Loading tokenizer {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded successfully! Vocab size:", len(tokenizer))

print(f"Loading model {model_name} in bfloat16 on cuda...")
model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
print("Model loaded successfully!")
print("Num layers:", len(model.model.layers))
print("Hidden size:", model.config.hidden_size)
print("Vocab size:", model.config.vocab_size)
