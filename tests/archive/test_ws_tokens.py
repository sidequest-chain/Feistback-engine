from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
ws_tokens = []
for i in range(tokenizer.vocab_size):
    try:
        text = tokenizer.decode([i])
        if text and text.strip() == "":
            ws_tokens.append(i)
    except Exception:
        pass

print(f"Total whitespace tokens identified: {len(ws_tokens)}")
print(f"Sample whitespace tokens: {ws_tokens[:10]}")
for t in ws_tokens[:5]:
    print(f"ID {t}: {repr(tokenizer.decode([t]))}")

ws_tensor = torch.tensor(ws_tokens, device="cuda")
sample_labels = torch.randint(0, tokenizer.vocab_size, (8, 256), device="cuda")
is_ws = torch.isin(sample_labels, ws_tensor)
print(f"Sample matches: {is_ws.sum().item()} / {sample_labels.numel()}")
