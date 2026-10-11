"""
Synthetic Slang & Estengrish Dataset Builder for eesti_keele_sl2ng
==================================================================
Generates 10,000 high-quality synthetic instruction-contract training pairs
covering Estonian colloquial speech, IT-slang, Estengrish, and compound operations
(kernel fusion) for training the dedicated eesti_keele_sl2ng micro-model (RFC-002).

Categories:
  1. Multi-number base arithmetic (30% -> 3,000 pairs)
  2. IT-slang and Estengrish (25% -> 2,500 pairs)
  3. Chained compound operations / Kernel Fusion (35% -> 3,500 pairs)
  4. Missing argument queries / Slot-filling (10% -> 1,000 pairs)

Output format per line:
{
  "instruction": "...",
  "contract": {
    "intent_name": "...",
    "strategy": "single_pass" | "need_input",
    "signature": "def ... -> int:",
    "args": [...]
  }
}
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Tuple

# Fix random seed for strict reproducibility
random.seed(42)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "data" / "slang_train_10k.jsonl"


# ==============================================================================
# Helper generators & number helpers
# ==============================================================================

EST_NUMBER_WORDS = {
    0: "null",
    1: "üks",
    2: "kaks",
    3: "kolm",
    4: "neli",
    5: "viis",
    6: "kuus",
    7: "seitse",
    8: "kaheksa",
    9: "üheksa",
    10: "kümme",
    15: "viisteist",
    20: "kakskümmend",
}


def format_num_choice(n: int, prob_word: float = 0.15) -> str:
    """Returns number either as a digit string or Estonian word string."""
    if prob_word > 0 and n in EST_NUMBER_WORDS and random.random() < prob_word:
        return EST_NUMBER_WORDS[n]
    return str(n)


# ==============================================================================
# Category 1: Multi-number base arithmetic (30% -> 3,000 pairs)
# ==============================================================================

def generate_category_1(count: int = 3000) -> List[Dict[str, Any]]:
    """Generates multi-number base operations: sums, differences, products of 2, 3, 4 numbers."""
    records: List[Dict[str, Any]] = []

    # 1.1 Two-number sums (600 items)
    two_sum_templates = [
        "liida kokku {a} ja {b}",
        "liida {a} ning {b}",
        "löö punti {a} ja {b}",
        "viska kokku {a} ja {b}",
        "pane ritta ja summeeri {a} ja {b}",
        "arvuta summa arvudel {a} ja {b}",
        "liida kokku kaks arvu {a} ja {b}",
        "liida kaks arvu: {a} ja {b}",
        "liida {a}-le otsa {b}",
        "pane kokku {a} ja {b}",
        "summeeri {a} ja {b}",
        "arvuta {a} pluss {b}",
        "liida kokku arvud {a} ja {b}",
        "viska kokku kaks numbrit: {a} ja {b}",
        "löö kokku {a} ning {b}",
    ]
    for _ in range(600):
        a = random.randint(1, 150)
        b = random.randint(1, 150)
        tmpl = random.choice(two_sum_templates)
        instruction = tmpl.format(a=format_num_choice(a), b=format_num_choice(b))
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "add",
                "strategy": "single_pass",
                "signature": "def add(a: int, b: int) -> int:",
                "args": [a, b],
            },
        })

    # 1.2 Three-number sums (600 items)
    three_sum_templates = [
        "liida kokku {a}, {b} ja {c}",
        "summeeri kolm arvu {a}, {b}, {c}",
        "löö punti {a}, {b} ja {c}",
        "viska kokku kolm numbrit {a}, {b} ja {c}",
        "pane ritta ja liida {a}, {b} ja {c}",
        "arvuta summa arvudest {a}, {b} ning {c}",
        "liida kokku kolm numbrit: {a}, {b} ja {c}",
        "pane kokku {a}, {b} ning {c}",
        "liida {a} pluss {b} pluss {c}",
        "arvuta kolme arvu summa: {a}, {b}, {c}",
        "löö kokku kolm arvu: {a}, {b} ja {c}",
    ]
    for _ in range(600):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        c = random.randint(1, 100)
        tmpl = random.choice(three_sum_templates)
        instruction = tmpl.format(
            a=format_num_choice(a),
            b=format_num_choice(b),
            c=format_num_choice(c),
        )
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "sum_three",
                "strategy": "single_pass",
                "signature": "def sum_three(a: int, b: int, c: int) -> int:",
                "args": [a, b, c],
            },
        })

    # 1.3 Four-number sums (400 items)
    four_sum_templates = [
        "liida neli arvu {a}, {b}, {c} ja {d}",
        "löö punti neli arvu {a}, {b}, {c}, {d}",
        "pane ritta neli numbrit {a}, {b}, {c} ja {d} ning liida",
        "summeeri {a}, {b}, {c} ja {d}",
        "viska kokku 4 arvu {a}, {b}, {c} ja {d}",
        "arvuta nelja arvu summa: {a}, {b}, {c} ning {d}",
        "liida kokku arvud {a}, {b}, {c} ja {d}",
        "pane kokku neli numbrit: {a}, {b}, {c}, {d}",
    ]
    for _ in range(400):
        nums = [random.randint(1, 80) for _ in range(4)]
        tmpl = random.choice(four_sum_templates)
        instruction = tmpl.format(
            a=format_num_choice(nums[0]),
            b=format_num_choice(nums[1]),
            c=format_num_choice(nums[2]),
            d=format_num_choice(nums[3]),
        )
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "sum_four",
                "strategy": "single_pass",
                "signature": "def sum_four(a: int, b: int, c: int, d: int) -> int:",
                "args": nums,
            },
        })

    # 1.4 Two-number differences (500 items)
    two_sub_templates = [
        "lahuta {a}-st {b}",
        "lahuta arvust {a} arv {b}",
        "arvuta vahe arvude {a} ja {b} vahel",
        "tõmba {a}-st maha {b}",
        "võta {a}-st ära {b}",
        "lahuta kaks arvu {a} ja {b}",
        "leia diferents {a} ja {b}",
        "arvuta vahe {a} miinus {b}",
        "arvuta {a} miinus {b}",
        "viska vahe arvude {a} ja {b} vahel",
    ]
    for _ in range(500):
        a = random.randint(10, 200)
        b = random.randint(1, a - 1 if a > 1 else 1)
        tmpl = random.choice(two_sub_templates)
        instruction = tmpl.format(a=format_num_choice(a), b=format_num_choice(b))
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "sub",
                "strategy": "single_pass",
                "signature": "def sub(a: int, b: int) -> int:",
                "args": [a, b],
            },
        })

    # 1.5 Three-number sequential differences (300 items)
    three_sub_templates = [
        "lahuta arvust {a} arvud {b} ja {c}",
        "võta {a}-st maha {b} ja {c}",
        "lahuta kolm arvu {a}, {b} ja {c}",
        "tõmba {a}-st ära {b} ning {c}",
        "arvuta {a} miinus {b} miinus {c}",
        "arvuta kolme arvu järjestikune vahe: {a}, {b} ja {c}",
    ]
    for _ in range(300):
        a = random.randint(50, 300)
        b = random.randint(1, 40)
        c = random.randint(1, 40)
        tmpl = random.choice(three_sub_templates)
        instruction = tmpl.format(
            a=format_num_choice(a),
            b=format_num_choice(b),
            c=format_num_choice(c),
        )
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "sub_three",
                "strategy": "single_pass",
                "signature": "def sub_three(a: int, b: int, c: int) -> int:",
                "args": [a, b, c],
            },
        })

    # 1.6 Two-number multiplications (350 items)
    two_mul_templates = [
        "korruta {a} ja {b}",
        "korruta omavahel {a} ja {b}",
        "arvuta korrutis {a} korda {b}",
        "korruta kokku kaks arvu {a} ja {b}",
        "viska korrutis arvudele {a} ja {b}",
        "korruta arv {a} arvuga {b}",
        "arvuta kahe arvu korrutis: {a} ja {b}",
        "arvuta {a} korda {b}",
    ]
    for _ in range(350):
        a = random.randint(2, 50)
        b = random.randint(2, 50)
        tmpl = random.choice(two_mul_templates)
        instruction = tmpl.format(a=format_num_choice(a), b=format_num_choice(b))
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "mul",
                "strategy": "single_pass",
                "signature": "def mul(a: int, b: int) -> int:",
                "args": [a, b],
            },
        })

    # 1.7 Three- and four-number multiplications (250 items)
    multi_mul_templates = [
        ("korruta kolm arvu {a}, {b} ja {c}", 3),
        ("korruta läbi {a}, {b} ja {c}", 3),
        ("arvuta kolme arvu korrutis {a}, {b} ja {c}", 3),
        ("korruta neli arvu {a}, {b}, {c} ja {d}", 4),
        ("arvuta nelja arvu korrutis {a}, {b}, {c} ja {d}", 4),
    ]
    for _ in range(250):
        tmpl, count_args = random.choice(multi_mul_templates)
        if count_args == 3:
            nums = [random.randint(2, 15) for _ in range(3)]
            instruction = tmpl.format(
                a=format_num_choice(nums[0]),
                b=format_num_choice(nums[1]),
                c=format_num_choice(nums[2]),
            )
            records.append({
                "instruction": instruction,
                "contract": {
                    "intent_name": "mul_three",
                    "strategy": "single_pass",
                    "signature": "def mul_three(a: int, b: int, c: int) -> int:",
                    "args": nums,
                },
            })
        else:
            nums = [random.randint(2, 10) for _ in range(4)]
            instruction = tmpl.format(
                a=format_num_choice(nums[0]),
                b=format_num_choice(nums[1]),
                c=format_num_choice(nums[2]),
                d=format_num_choice(nums[3]),
            )
            records.append({
                "instruction": instruction,
                "contract": {
                    "intent_name": "mul_four",
                    "strategy": "single_pass",
                    "signature": "def mul_four(a: int, b: int, c: int, d: int) -> int:",
                    "args": nums,
                },
            })

    assert len(records) == count, f"Expected {count}, got {len(records)}"
    return records


# ==============================================================================
# Category 2: IT-slang and Estengrish (25% -> 2,500 pairs)
# ==============================================================================

def generate_category_2(count: int = 2500) -> List[Dict[str, Any]]:
    """Generates IT-slang and Estengrish: clamping, xor, shifts, masks."""
    records: List[Dict[str, Any]] = []

    # 2.1 Clamping / Klembi (700 items)
    clamp_templates = [
        "clämbi see counter {val} vahemikku {low} ja {high}",
        "klembi see counter {val} vahemikku {low} ja {high}",
        "clämbi counter {val} vahemikku {low} ja {high}",
        "klembi {val} vahemikku {low} ja {high}",
        "klämbi see väärtus {val} piiridesse {low} ja {high}",
        "clampi counter {val} vahemikku {low} ja {high}",
        "suru {val} vahemikku {low} ja {high}",
        "suru counter {val} vahemikku {low} kuni {high}",
        "piira counter {val} vahemikku {low} ja {high}",
        "pane clamp peale väärtusele {val} piiridega {low} ja {high}",
        "clämbi number {val} vahemikku {low} ja {high}",
        "hoia counter {val} vahemikus {low} kuni {high} clampiga",
        "piira muutuja {val} min {low} ja max {high} vahele",
        "cap'i counter {val} vahemikku {low} ja {high}",
    ]
    for _ in range(700):
        low = random.randint(0, 50)
        high = random.randint(low + 10, low + 100)
        val = random.choice([
            random.randint(low - 30, low - 1),
            random.randint(low, high),
            random.randint(high + 1, high + 40),
        ])
        tmpl = random.choice(clamp_templates)
        instruction = tmpl.format(val=val, low=low, high=high)
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "clamp_math",
                "strategy": "single_pass",
                "signature": "def clamp_range_1(val: int, low: int, high: int) -> int:",
                "args": [val, low, high],
            },
        })

    # 2.2 Bitwise XOR / Comb (600 items)
    xor_templates = [
        "viska xor {a} ja {b} vahel",
        "viska xor {a} ja {b} vahele",
        "lase xor {a} ja {b} peale",
        "xor'i {a} ja {b}",
        "xori väärtused {a} ja {b}",
        "tee bitwise xor arvudele {a} ja {b}",
        "toggle bitid xoriga arvudel {a} ja {b}",
        "bitivexori {a} ja {b}",
        "tee xor {a} ning {b} vahel",
        "viska bitwise xor arvude {a} ja {b} vahele",
        "arvuta {a} XOR {b}",
        "lase bitikombinatsioon xorina arvudele {a} ja {b}",
    ]
    for _ in range(600):
        a = random.choice([4, 8, 12, 15, 16, 24, 32, 63, 100, 127, 255, random.randint(1, 128)])
        b = random.choice([1, 2, 4, 7, 8, 15, 31, 64, random.randint(1, 128)])
        tmpl = random.choice(xor_templates)
        instruction = tmpl.format(a=a, b=b)
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "bitwise_comb",
                "strategy": "single_pass",
                "signature": "def bitwise_xor_mask_1(a: int, b: int) -> int:",
                "args": [a, b],
            },
        })

    # 2.3 Bit Shift Left / Right (600 items: 300 left, 300 right)
    shift_left_templates = [
        "shifti {val} vasakule {shift} võrra",
        "shifti väärtus {val} vasakule {shift} bitti",
        "shifteeri {val} vasakule {shift} positsiooni",
        "lükka {val} bitid vasakule {shift} võrra",
        "bit shift left arvule {val} sammuga {shift}",
        "nihuta {val} bitte vasakule {shift} võrra",
        "nihuta vasakule bitte arvul {val} {shift} võrra",
    ]
    shift_right_templates = [
        "shifti {val} paremale {shift} võrra",
        "shifti counter {val} paremale {shift} bitti",
        "shifteeri {val} paremale {shift} positsiooni",
        "lükka {val} bitid paremale {shift} võrra",
        "bit shift right arvule {val} sammuga {shift}",
        "nihuta {val} bitte paremale {shift} võrra",
        "nihuta paremale bitte arvul {val} {shift} võrra",
    ]
    for _ in range(300):
        val = random.randint(1, 255)
        shift = random.randint(1, 8)
        tmpl = random.choice(shift_left_templates)
        instruction = tmpl.format(val=val, shift=shift)
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "shift_left",
                "strategy": "single_pass",
                "signature": "def shift_left(val: int, shift: int) -> int:",
                "args": [val, shift],
            },
        })

    for _ in range(300):
        val = random.randint(16, 512)
        shift = random.randint(1, 6)
        tmpl = random.choice(shift_right_templates)
        instruction = tmpl.format(val=val, shift=shift)
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "shift_right",
                "strategy": "single_pass",
                "signature": "def shift_right(val: int, shift: int) -> int:",
                "args": [val, shift],
            },
        })

    # 2.4 Bitwise AND / Masking (600 items)
    mask_templates = [
        "maski {val} ära maskiga {mask}",
        "pane mask {mask} peale väärtusele {val}",
        "and-i {val} maskiga {mask}",
        "tõmba maskiga {mask} üle arvu {val}",
        "filtreeri bitid arvul {val} maskiga {mask}",
        "rakenda bitmask {mask} arvule {val}",
        "maskeeri {val} kasutades maski {mask}",
        "tee bitwise AND arvule {val} ja maskile {mask}",
        "jäta alles bitid arvul {val} maskiga {mask}",
    ]
    common_masks = [1, 3, 7, 15, 31, 63, 127, 255, 1023]
    for _ in range(600):
        val = random.randint(1, 2048)
        mask = random.choice(common_masks)
        tmpl = random.choice(mask_templates)
        instruction = tmpl.format(val=val, mask=mask)
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "bitwise_mask",
                "strategy": "single_pass",
                "signature": "def bitwise_mask(val: int, mask: int) -> int:",
                "args": [val, mask],
            },
        })

    assert len(records) == count, f"Expected {count}, got {len(records)}"
    return records


# ==============================================================================
# Category 3: Chained compound operations / Kernel Fusion (35% -> 3,500 pairs)
# ==============================================================================

def generate_category_3(count: int = 3500) -> List[Dict[str, Any]]:
    """Generates chained arithmetic operations (kernel fusion): sum3_sub, calc_with_locals, add_mul, etc."""
    records: List[Dict[str, Any]] = []

    # 3.1 sum3_sub: liida 3 ja lahuta 1 (600 items)
    sum3_sub_templates = [
        "liida {a}, {b} ja {c} lahuta neist {d}",
        "liida kokku {a}, {b} ja {c} ning lahuta sellest {d}",
        "summeeri {a}, {b} ja {c} ning võta maha {d}",
        "löö punti {a}, {b} ja {c} ning tõmba maha {d}",
        "liida kolm arvu {a}, {b}, {c} ja lahuta tulemusest {d}",
        "arvuta ({a} + {b} + {c}) miinus {d}",
        "pane kokku {a}, {b} ja {c} ning lahuta {d}",
        "arvuta kolme arvu summa {a}, {b}, {c} ja lahuta {d}",
    ]
    for _ in range(600):
        a = random.randint(5, 50)
        b = random.randint(5, 50)
        c = random.randint(5, 50)
        d = random.randint(1, 20)
        tmpl = random.choice(sum3_sub_templates)
        instruction = tmpl.format(
            a=format_num_choice(a),
            b=format_num_choice(b),
            c=format_num_choice(c),
            d=format_num_choice(d),
        )
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "sum3_sub",
                "strategy": "single_pass",
                "signature": "def sum3_sub(a: int, b: int, c: int, d: int) -> int:",
                "args": [a, b, c, d],
            },
        })

    # 3.2 calc_with_locals: arvuta vahe ruut arvudel a ja b ja liida üks (600 items)
    calc_locals_templates = [
        "arvuta vahe ruut arvudel {a} ja {b} ja liida üks otsa",
        "arvuta vahe ruut arvudel {a} ja {b} ja liida üks",
        "võta {a} ja {b} vahe, pane ruutu ja liida üks juurde",
        "leia arvude {a} ja {b} diferentsi ruut pluss üks",
        "arvuta vahe ruut arvudele {a} ja {b} ning lisa üks",
        "arvuta ({a} - {b})^2 + 1",
        "lahuta {a}-st {b}, võta ruutu ja liida üks otsa",
        "leia vahe ruut arvudel {a} ja {b} ning liida üks",
    ]
    for _ in range(600):
        a = random.randint(5, 60)
        b = random.randint(1, 30)
        tmpl = random.choice(calc_locals_templates)
        instruction = tmpl.format(a=format_num_choice(a), b=format_num_choice(b))
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "calc_with_locals",
                "strategy": "single_pass",
                "signature": "def calc_with_locals(a: int, b: int) -> int:",
                "args": [a, b],
            },
        })

    # 3.3 add_mul: liida a ja b ning korruta c-ga (600 items)
    add_mul_templates = [
        "liida {a} ja {b} ning korruta {c}-ga",
        "liida kokku {a} ja {b} ja korruta tulemus {c}-ga",
        "summeeri {a} ja {b} ning korda {c} korda",
        "arvuta ({a} + {b}) * {c}",
        "pane kokku {a} ja {b} ning korruta arvuga {c}",
        "liida arvud {a} ja {b} ning korruta see {c}-ga",
        "arvuta summa arvudest {a} ja {b} ning korruta {c}-ga",
    ]
    for _ in range(600):
        a = random.randint(1, 40)
        b = random.randint(1, 40)
        c = random.randint(2, 12)
        tmpl = random.choice(add_mul_templates)
        instruction = tmpl.format(
            a=format_num_choice(a),
            b=format_num_choice(b),
            c=format_num_choice(c),
        )
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "add_mul",
                "strategy": "single_pass",
                "signature": "def add_mul(a: int, b: int, c: int) -> int:",
                "args": [a, b, c],
            },
        })

    # 3.4 mul_add: fused multiply add (korruta a ja b ning liida c) (600 items)
    mul_add_templates = [
        "korruta {a} ja {b} ning liida {c}",
        "korruta omavahel {a} ja {b} ja pane {c} otsa",
        "arvuta {a} korda {b} pluss {c}",
        "fused multiply add: korruta {a} ja {b} ning lisa {c}",
        "korruta {a} arvuga {b} ning liida juurde {c}",
        "arvuta korrutis {a} * {b} ja liida sellele {c}",
        "korruta kokku {a} ja {b} ning summeeri {c}-ga",
    ]
    for _ in range(600):
        a = random.randint(2, 25)
        b = random.randint(2, 25)
        c = random.randint(1, 50)
        tmpl = random.choice(mul_add_templates)
        instruction = tmpl.format(
            a=format_num_choice(a),
            b=format_num_choice(b),
            c=format_num_choice(c),
        )
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "mul_add",
                "strategy": "single_pass",
                "signature": "def mul_add(a: int, b: int, c: int) -> int:",
                "args": [a, b, c],
            },
        })

    # 3.5 sub_mul: lahuta a-st b ja korruta c-ga (500 items)
    sub_mul_templates = [
        "lahuta {a}-st {b} ja korruta {c}-ga",
        "võta {a}-st maha {b} ning korruta tulemus {c}-ga",
        "arvuta vahe {a} ja {b} vahel ning korruta {c}-ga",
        "arvuta ({a} - {b}) * {c}",
        "tõmba {a}-st ära {b} ja korruta see {c}-ga",
        "lahuta arvud {a} ja {b} ning korruta {c} korda",
    ]
    for _ in range(500):
        a = random.randint(15, 60)
        b = random.randint(1, a - 1)
        c = random.randint(2, 10)
        tmpl = random.choice(sub_mul_templates)
        instruction = tmpl.format(
            a=format_num_choice(a),
            b=format_num_choice(b),
            c=format_num_choice(c),
        )
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "sub_mul",
                "strategy": "single_pass",
                "signature": "def sub_mul(a: int, b: int, c: int) -> int:",
                "args": [a, b, c],
            },
        })

    # 3.6 sum_of_squares: ruutude summa arvudel a ja b (600 items)
    sum_sq_templates = [
        "arvuta ruutude summa arvudel {a} ja {b}",
        "pane ruutu {a} ja {b} ning liida kokku",
        "arvuta {a} ruut pluss {b} ruut",
        "leia ruutude summa: {a} ja {b}",
        "arvuta {a}^2 + {b}^2",
        "võta ruutu {a} ja {b} ning liida need",
        "arvuta ruutude summa: arvud {a} ja {b}",
    ]
    for _ in range(600):
        a = random.randint(1, 30)
        b = random.randint(1, 30)
        tmpl = random.choice(sum_sq_templates)
        instruction = tmpl.format(a=format_num_choice(a), b=format_num_choice(b))
        records.append({
            "instruction": instruction,
            "contract": {
                "intent_name": "sum_of_squares",
                "strategy": "single_pass",
                "signature": "def sum_of_squares(a: int, b: int) -> int:",
                "args": [a, b],
            },
        })

    assert len(records) == count, f"Expected {count}, got {len(records)}"
    return records


# ==============================================================================
# Category 4: Missing argument queries / Slot-filling (10% -> 1,000 pairs)
# ==============================================================================

def generate_category_4(count: int = 1000) -> List[Dict[str, Any]]:
    """Generates queries with missing arguments requiring clarification (slot-filling)."""
    records: List[Dict[str, Any]] = []

    # Distribution across core operations
    sub_categories = [
        # (intent, signature, templates, count)
        (
            "add",
            "def add(a: int, b: int) -> int:",
            [
                "palun liida",
                "liida kokku",
                "summeeri arve",
                "löö numbrid punti",
                "pane kokku",
                "tee liitmine",
                "palun arvuta summa",
                "liida palun",
                "viska kokku",
                "arvuta summa",
                "liida",
                "tahan liita arve",
                "tee kahe arvu summa",
            ],
            150,
        ),
        (
            "sub",
            "def sub(a: int, b: int) -> int:",
            [
                "palun lahuta",
                "lahuta arvud",
                "arvuta vahe",
                "tõmba maha",
                "võta ära",
                "tee lahutamine",
                "leia vahe",
                "lahuta palun",
                "arvuta diferents",
                "tee kahe arvu vahe",
            ],
            120,
        ),
        (
            "mul",
            "def mul(a: int, b: int) -> int:",
            [
                "palun korruta",
                "korruta numbrid läbi",
                "arvuta korrutis",
                "tee korrutamine",
                "korruta omavahel",
                "korruta läbi",
                "korruta palun",
                "tee kahe arvu korrutis",
            ],
            120,
        ),
        (
            "clamp_math",
            "def clamp_range_1(val: int, low: int, high: int) -> int:",
            [
                "suru vahemikku",
                "clämbi ära",
                "piira väärtus",
                "klembi counterit",
                "pane clamp peale",
                "klämbi number",
                "clampi",
                "suru piiridesse",
                "clämbi counter",
                "klembi ära",
                "palun clämbi see väärtus",
            ],
            150,
        ),
        (
            "bitwise_comb",
            "def bitwise_xor_mask_1(a: int, b: int) -> int:",
            [
                "viska xor",
                "tee bitwise xor",
                "xori",
                "lase xor peale",
                "bitivexor",
                "tee xor tehe",
                "palun viska xor",
                "tee xor",
                "viska xor vahele",
            ],
            120,
        ),
        (
            "shift_left",
            "def shift_left(val: int, shift: int) -> int:",
            [
                "shifti vasakule",
                "lükka bittides vasakule",
                "nihuta bitte vasakule",
                "tee bit shift vasakule",
                "shifteeri vasakule",
                "palun shifti",
                "nihuta bitte",
            ],
            120,
        ),
        (
            "bitwise_mask",
            "def bitwise_mask(val: int, mask: int) -> int:",
            [
                "maski ära",
                "pane mask peale",
                "tõmba maskiga üle",
                "rakenda bitmask",
                "filtreeri maskiga",
                "tee bitwise mask",
                "maskeeri",
            ],
            100,
        ),
        (
            "calc_with_locals",
            "def calc_with_locals(a: int, b: int) -> int:",
            [
                "arvuta vahe ruut",
                "arvuta vahe ruut ja liida üks",
                "leia vahe ruut pluss üks",
                "võta vahe ruutu ja liida otsa",
            ],
            60,
        ),
        (
            "add_mul",
            "def add_mul(a: int, b: int, c: int) -> int:",
            [
                "liida ja korruta",
                "liida arve ja korruta tulemus",
                "summeeri ning korruta",
                "tee liitmine ja korrutamine",
            ],
            60,
        ),
    ]

    for intent_name, signature, templates, num_items in sub_categories:
        for _ in range(num_items):
            instruction = random.choice(templates)
            records.append({
                "instruction": instruction,
                "contract": {
                    "intent_name": intent_name,
                    "strategy": "need_input",
                    "signature": signature,
                    "args": [],
                },
            })

    assert len(records) == count, f"Expected {count}, got {len(records)}"
    return records


# ==============================================================================
# Main Generation Runner
# ==============================================================================

def main() -> None:
    print("=" * 80)
    print("SLANG & ESTENGRISH DATASET BUILDER (eesti_keele_sl2ng)")
    print("=" * 80)

    print("[1/4] Generating Category 1: Multi-number base arithmetic (3,000 pairs)...")
    cat1 = generate_category_1(3000)

    print("[2/4] Generating Category 2: IT-slang and Estengrish (2,500 pairs)...")
    cat2 = generate_category_2(2500)

    print("[3/4] Generating Category 3: Chained compound operations (3,500 pairs)...")
    cat3 = generate_category_3(3500)

    print("[4/4] Generating Category 4: Missing arguments / Slot-filling (1,000 pairs)...")
    cat4 = generate_category_4(1000)

    all_records = cat1 + cat2 + cat3 + cat4
    print(f"\nTotal raw records collected: {len(all_records):,}")

    # Shuffle to guarantee homogeneous distribution across categories
    random.shuffle(all_records)

    # Ensure parent output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    file_size_bytes = OUTPUT_FILE.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"Dataset successfully created: {OUTPUT_FILE}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"Total lines: {len(all_records):,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
