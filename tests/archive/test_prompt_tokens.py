from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B')
prompt = 'def fibonacci(n: int) -> int:\n    """Return the n-th Fibonacci number."""\n'
enc = tok(prompt)
print('Tokens:')
for id in enc['input_ids']:
    print(f'  {id}: {repr(tok.decode([id]))}')
