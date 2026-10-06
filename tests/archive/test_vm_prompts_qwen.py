import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16, device_map="cuda")

vm_prompts = [
    "Calculate total memory: 1024 + 2048 using VM bytecode:\n<EXEC_VM>",
    "Task: Check if latency 420 exceeds threshold 300 using MiniStackVM bytecode:\n<EXEC_VM>",
    "Task: Compute batch dimension 16 * 64 via deterministic stack operations:\n<EXEC_VM>",
    "Task: Verify whether HTTP status code equals 200 using VM comparison bytecode:\n<EXEC_VM>",
    "Task: Subtract baseline offset 512 from 2048 using stack VM instructions:\n<EXEC_VM>",
]

for p in vm_prompts:
    inputs = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    raw = tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("=" * 50)
    print("PROMPT:", p.split(chr(10))[0])
    print("GENERATED:\n", raw)
