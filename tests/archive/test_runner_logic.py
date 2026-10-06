import time
import torch
import torch.nn as nn
from mini_vm.ops import OpCode
from mini_vm.vm import MiniStackVM
from runner.adaptive_loop import AdaptiveFeistelRunner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Testing on device:", device)

vm = MiniStackVM(max_steps=500)
runner = AdaptiveFeistelRunner(d_model=128, max_seq_len=64).to(device)
runner.eval()
embed = nn.Embedding(124, 128).to(device)

unified_bytecode = [
    # 0, 1: LOAD_ARG 0 (line)
    OpCode.LOAD_ARG, 0,
    # 2, 3: LOAD_ARG 2 (regex pattern)
    OpCode.LOAD_ARG, 2,
    # 4: REGEX_MATCH
    OpCode.REGEX_MATCH,
    # 5, 6: BRANCH_IF target 30
    OpCode.BRANCH_IF, 30,
    # 7, 8: LOAD_ARG 0
    OpCode.LOAD_ARG, 0,
    # 9, 10: PUSH_CONST "status"
    OpCode.PUSH_CONST, "status",
    # 11: LOAD_JSON_KEY
    OpCode.LOAD_JSON_KEY,
    # 12, 13: PUSH_CONST 499
    OpCode.PUSH_CONST, 499,
    # 14: CMP_GT
    OpCode.CMP_GT,
    # 15, 16: BRANCH_IF target 30
    OpCode.BRANCH_IF, 30,
    # 17, 18: LOAD_ARG 0
    OpCode.LOAD_ARG, 0,
    # 19, 20: PUSH_CONST "latency_ms"
    OpCode.PUSH_CONST, "latency_ms",
    # 21: LOAD_JSON_KEY
    OpCode.LOAD_JSON_KEY,
    # 22, 23: LOAD_ARG 1 (threshold)
    OpCode.LOAD_ARG, 1,
    # 24: CMP_GT
    OpCode.CMP_GT,
    # 25, 26: BRANCH_IF target 30
    OpCode.BRANCH_IF, 30,
    # 27, 28: RET 0 (normal)
    OpCode.PUSH_CONST, 0,
    OpCode.RET,
    # 30, 31: RET 1 (anomaly)
    OpCode.PUSH_CONST, 1,
    OpCode.RET,
]

pattern = r"(ERROR|WARN|FATAL|50[0-9]|timeout|fail)"

def check_line(line: str) -> int:
    try:
        return vm.execute(unified_bytecode, args=[line, 300, pattern])
    except Exception:
        return 0

test_lines = [
    '{"timestamp": "2026-10-05T18:00:00Z", "status": 200, "latency_ms": 45}',
    '{"timestamp": "2026-10-05T18:00:01Z", "status": 500, "latency_ms": 45}',
    '{"timestamp": "2026-10-05T18:00:02Z", "status": 200, "latency_ms": 420}',
    'Oct 5 18:00:03 auth[123]: [ERROR] Connection timed out',
    '{"bad_json: 123',
    '\\x00\\xffgarbage??!@#',
    '',
]

results = [check_line(l) for l in test_lines]
print("Results:", results)
assert results == [0, 1, 1, 1, 0, 0, 0], f"Unexpected results: {results}"
print("All assertions passed!")
