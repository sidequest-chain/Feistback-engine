from mini_vm.vm import MiniStackVM
from mini_vm.ops import OpCode

vm = MiniStackVM()
unified_bytecode = [
    # 0, 1: LOAD_ARG 0 (line)
    OpCode.LOAD_ARG, 0,
    # 2, 3: LOAD_ARG 2 (regex pattern)
    OpCode.LOAD_ARG, 2,
    # 4: REGEX_MATCH
    OpCode.REGEX_MATCH,
    # 5, 6: BRANCH_IF target
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
    # 15, 16: BRANCH_IF target
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
    # 25, 26: BRANCH_IF target
    OpCode.BRANCH_IF, 30,
    # 27, 28: RET 0 (normal)
    OpCode.PUSH_CONST, 0,
    OpCode.RET,
    # 30, 31: RET 1 (anomaly)
    OpCode.PUSH_CONST, 1,
    OpCode.RET,
]

print("Unified Disassembly:")
print(vm.disassemble(unified_bytecode))
pattern = r"(ERROR|WARN|FATAL|50[0-9]|timeout|fail)"
args_normal_json = ['{"status": 200, "latency_ms": 50}', 300, pattern]
args_status_500 = ['{"status": 500, "latency_ms": 50}', 300, pattern]
args_latency_420 = ['{"status": 200, "latency_ms": 420}', 300, pattern]
args_syslog_norm = ['Oct 5 18:00:00 auth[123]: info login ok', 300, pattern]
args_syslog_err = ['Oct 5 18:00:00 auth[123]: [ERROR] connection timed out', 300, pattern]
args_corrupt = ['{"bad_json: 123', 300, pattern]
args_empty = ['', 300, pattern]

print("Normal JSON:", vm.execute(unified_bytecode, args=args_normal_json))
print("Status 500 JSON:", vm.execute(unified_bytecode, args=args_status_500))
print("Latency 420 JSON:", vm.execute(unified_bytecode, args=args_latency_420))
print("Syslog Normal:", vm.execute(unified_bytecode, args=args_syslog_norm))
print("Syslog Error:", vm.execute(unified_bytecode, args=args_syslog_err))
print("Corrupt JSON:", vm.execute(unified_bytecode, args=args_corrupt))
print("Empty Line:", vm.execute(unified_bytecode, args=args_empty))


regex_bytecode = [
    OpCode.LOAD_ARG, 0, # line
    OpCode.LOAD_ARG, 1, # pattern
    OpCode.REGEX_MATCH,
    OpCode.RET,
]
err_pattern = r"(ERROR|WARN|FATAL|500|502|504|timeout|timed out|fail)"
print("Syslog normal:", vm.execute(regex_bytecode, args=['Oct 5 18:00:00 auth[123]: normal ping', err_pattern]))
print("Syslog error:", vm.execute(regex_bytecode, args=['Oct 5 18:00:00 auth[123]: [ERROR] connection failed', err_pattern]))

