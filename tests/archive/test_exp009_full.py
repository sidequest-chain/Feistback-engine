from pathlib import Path
import time
import json
import torch
import torch.nn as nn
from mini_vm.vm import MiniStackVM
from mini_vm.ops import OpCode
from runner.adaptive_loop import AdaptiveFeistelRunner

class TestRunner:
    def __init__(self, device=None, latency_threshold=300):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.latency_threshold = latency_threshold
        self.pattern = r"(ERROR|WARN|FATAL|50[0-9]|timeout|fail)"
        self.d_model = 128
        self.seq_len = 64
        self.vocab_size = 124

        self.vm = MiniStackVM(max_steps=500)
        self.runner = AdaptiveFeistelRunner(d_model=self.d_model, max_seq_len=self.seq_len).to(self.device)
        self.runner.eval()
        self.embed = nn.Embedding(self.vocab_size, self.d_model).to(self.device)

        self.unified_bytecode = [
            OpCode.LOAD_ARG, 0,
            OpCode.LOAD_ARG, 2,
            OpCode.REGEX_MATCH,
            OpCode.BRANCH_IF, 30,
            OpCode.LOAD_ARG, 0,
            OpCode.PUSH_CONST, "status",
            OpCode.LOAD_JSON_KEY,
            OpCode.PUSH_CONST, 499,
            OpCode.CMP_GT,
            OpCode.BRANCH_IF, 30,
            OpCode.LOAD_ARG, 0,
            OpCode.PUSH_CONST, "latency_ms",
            OpCode.LOAD_JSON_KEY,
            OpCode.LOAD_ARG, 1,
            OpCode.CMP_GT,
            OpCode.BRANCH_IF, 30,
            OpCode.PUSH_CONST, 0,
            OpCode.RET,
            OpCode.PUSH_CONST, 1,
            OpCode.RET,
        ]

    def _encode_line_to_tokens(self, line: str):
        raw_bytes = line.encode("utf-8", errors="replace")[:self.seq_len]
        token_ids = [b % self.vocab_size for b in raw_bytes]
        if len(token_ids) < self.seq_len:
            token_ids += [0] * (self.seq_len - len(token_ids))
        return token_ids

    def process_line(self, line: str):
        try:
            if not line or not line.strip():
                return (0, None)
            res = self.vm.execute(
                self.unified_bytecode,
                args=[line, self.latency_threshold, self.pattern],
            )
            if res == 1:
                tokens = self._encode_line_to_tokens(line)
                token_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    z = self.embed(token_tensor)
                    z_delib, _, _ = self.runner.forward_adaptive(z, max_steps=1, epsilon=0.01)
                return (1, z_delib)
            return (0, None)
        except Exception:
            return (0, None)

runner = TestRunner()
print("Initialized runner on:", runner.device)

# Generate 10000 lines
lines = []
for i in range(8500):
    lines.append(f'{{"timestamp": "2026-10-05T18:00:{i % 50:02d}.{i % 499:03d}Z", "service": "api", "status": 200, "latency_ms": {20 + (i % 130)}}}')
for i in range(500):
    lines.append(f"Oct 05 18:00:{i % 50:02d} host-{i % 10:02d} worker[{100 + i % 100}]: info ping ok latency={20 + (i % 50)}ms")
for i in range(500):
    if i < 250:
        lines.append(f'{{"timestamp": "2026-10-05T18:01:00Z", "service": "api", "status": {500 if i % 2 == 0 else 502}, "latency_ms": 50}}')
    else:
        lines.append(f'{{"timestamp": "2026-10-05T18:01:00Z", "service": "api", "status": 200, "latency_ms": {350 + (i % 200)}}}')
for i in range(300):
    lines.append(f"Oct 05 18:01:00 host auth[{1000 + i}]: [ERROR] connection timed out fail")
corrupt_pool = ['{"bad_json: 123', '', r'\x00\xffgarbage']
for i in range(200):
    lines.append(corrupt_pool[i % len(corrupt_pool)])

# Write to temp file
test_file = Path("data/test_sample_logs.log")
test_file.parent.mkdir(exist_ok=True)
with open(test_file, "w", encoding="utf-8") as f:
    for l in lines:
        f.write(l.strip() + "\n")

# Process lines
t0 = time.perf_counter()
anomalies = 0
for l in lines:
    status, latent = runner.process_line(l)
    if status == 1:
        anomalies += 1
if runner.device.type == "cuda":
    torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1000.0

print(f"Time: {dt:.2f} ms ({len(lines) / (dt / 1000.0):.0f} lines/sec), Anomalies: {anomalies}")
assert anomalies == 800, f"Expected 800, got {anomalies}"
print("TEST SUCCESSFUL!")
