"""
Dynamic Dataset Generator: Dynamic Numerical Operands with LOAD_ARG Registers
=============================================================================
Generates dynamic training and evaluation samples on the fly with random integers
in range [1, 500].

Uses abstract register lookups:
  - <OP:LOAD_ARG> <ARG:0>
  - <OP:LOAD_ARG> <ARG:1>

Supported operations:
  1. Addition (Liitmine): a + b
  2. Subtraction (Lahutamine): a - b
  3. Multiplication (Korrutamine): a * b
  4. Maximum (Suurima leidmine): max(a, b)
"""

import random
from typing import List, Dict, Any, Optional


OP_TYPES = ["add", "sub", "mul", "max"]

PROMPT_TEMPLATES = {
    "add": [
        "Küsimus: liida arvud {a} ja {b}. ",
        "Küsimus: liida {a} ja {b}. ",
        "Küsimus: arvuta {a} + {b}. "
    ],
    "sub": [
        "Küsimus: lahuta arvust {a} arv {b}. ",
        "Küsimus: lahuta {a} ja {b}. ",
        "Küsimus: arvuta {a} - {b}. "
    ],
    "mul": [
        "Küsimus: korruta arvud {a} ja {b}. ",
        "Küsimus: korruta {a} ja {b}. ",
        "Küsimus: arvuta {a} * {b}. "
    ],
    "max": [
        "Küsimus: leia arvudest {a} ja {b} suurim. ",
        "Küsimus: kumb on suurem kas {a} või {b}. ",
        "Küsimus: kumb arv on suurem kas {a} või {b}. "
    ]
}

BYTECODE_TEMPLATES = {
    "add": "<OP:LOAD_ARG> <ARG:0> <OP:LOAD_ARG> <ARG:1> <OP:ADD> <OP:RET>",
    "sub": "<OP:LOAD_ARG> <ARG:0> <OP:LOAD_ARG> <ARG:1> <OP:SUB> <OP:RET>",
    "mul": "<OP:LOAD_ARG> <ARG:0> <OP:LOAD_ARG> <ARG:1> <OP:MUL> <OP:RET>",
    "max": "<OP:LOAD_ARG> <ARG:0> <OP:LOAD_ARG> <ARG:1> <OP:CMP_GT> <OP:JMP_IF_FALSE> <ARG:3> <OP:LOAD_ARG> <ARG:0> <OP:RET> <OP:LOAD_ARG> <ARG:1> <OP:RET>"
}


def calculate_op(op: str, a: int, b: int) -> int:
    """Computes exact mathematical ground truth."""
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    elif op == "max":
        return max(a, b)
    else:
        raise ValueError(f"Unknown operation: {op}")


def apply_jitter(prompt: str) -> str:
    """
    Applies subtle random whitespace jittering between words so that convolutional
    receptive fields become invariant to token shifts caused by multi-digit numbers.
    """
    words = prompt.strip().split()
    parts = []
    for i, w in enumerate(words):
        parts.append(w)
        if i < len(words) - 1:
            extra = " " * random.randint(0, 2)
            parts.append(" " + extra)
    return "".join(parts) + " "


def generate_dynamic_sample(
    op: Optional[str] = None,
    min_val: int = 1,
    max_val: int = 500,
    jitter: bool = True
) -> Dict[str, Any]:
    """Generates a single dynamic math task with random integers."""
    if op is None:
        op = random.choice(OP_TYPES)

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    res = calculate_op(op, a, b)

    template = random.choice(PROMPT_TEMPLATES[op])
    raw_prompt = template.format(a=a, b=b)
    prompt = apply_jitter(raw_prompt) if jitter else raw_prompt

    vm_block = BYTECODE_TEMPLATES[op]
    full_sequence = f"{prompt}<EXEC_VM> {vm_block} </EXEC_VM> <RES> Tulemuseks saame {res}."

    return {
        "op": op,
        "a": a,
        "b": b,
        "args": [a, b],
        "expected_val": res,
        "prompt": prompt,
        "vm_block": vm_block,
        "full_sequence": full_sequence
    }


def generate_dynamic_dataset(
    num_samples: int = 2000,
    seed: int = 42,
    min_val: int = 1,
    max_val: int = 500,
    jitter: bool = True
) -> List[Dict[str, Any]]:
    """Generates a diverse set of dynamic training samples."""
    random.seed(seed)
    dataset: List[Dict[str, Any]] = []
    
    per_op = num_samples // len(OP_TYPES)
    for op in OP_TYPES:
        for _ in range(per_op):
            dataset.append(generate_dynamic_sample(op=op, min_val=min_val, max_val=max_val, jitter=jitter))

    while len(dataset) < num_samples:
        dataset.append(generate_dynamic_sample(min_val=min_val, max_val=max_val, jitter=jitter))

    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    ds = generate_dynamic_dataset(num_samples=3)
    for i, s in enumerate(ds):
        print(f"[{i+1}] {s['full_sequence']}")
