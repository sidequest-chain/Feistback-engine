from mini_vm.vm import MiniStackVM
from mini_vm.ops import OpCode

vm = MiniStackVM()
pattern = r"(ERROR|WARN|FATAL|50[0-9]|timeout|fail)"
threshold = 300

unified_bytecode = [
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

lines = []
# 8500 normal JSON
for i in range(8500):
    lines.append(f'{{"timestamp": "2026-10-05T18:00:{i % 50:02d}.{i % 499:03d}Z", "service": "api", "status": 200, "latency_ms": {20 + (i % 130)}}}')

# 500 normal syslog
for i in range(500):
    lines.append(f"Oct 05 18:00:{i % 50:02d} host-{i % 10:02d} worker[{100 + i % 100}]: info ping ok latency={20 + (i % 50)}ms")

# 500 anomalous JSON
for i in range(500):
    if i < 250:
        lines.append(f'{{"timestamp": "2026-10-05T18:01:00Z", "service": "api", "status": {500 if i % 2 == 0 else 502}, "latency_ms": 50}}')
    else:
        lines.append(f'{{"timestamp": "2026-10-05T18:01:00Z", "service": "api", "status": 200, "latency_ms": {350 + (i % 200)}}}')

# 300 anomalous syslog
for i in range(300):
    lines.append(f"Oct 05 18:01:00 host auth[{1000 + i}]: [ERROR] connection timed out fail")

# 200 corrupt lines
corrupt_pool = ['{"bad_json: 123', '', r'\x00\xffgarbage']
for i in range(200):
    lines.append(corrupt_pool[i % len(corrupt_pool)])

anomalies = 0
for l in lines:
    try:
        res = vm.execute(unified_bytecode, args=[l, threshold, pattern])
        if res == 1:
            anomalies += 1
    except Exception as e:
        pass

print(f"Total lines: {len(lines)}, Anomalies detected: {anomalies}")
assert len(lines) == 10000, f"Expected 10000 lines, got {len(lines)}"
assert anomalies == 800, f"Expected 800 anomalies, got {anomalies}"
print("Verification SUCCESS!")
