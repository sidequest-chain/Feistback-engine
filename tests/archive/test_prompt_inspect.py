import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16, device_map="cuda")

prompt = 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n'
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
raw = tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("RAW:")
print(repr(raw))
print("PRINT:")
print(raw)
