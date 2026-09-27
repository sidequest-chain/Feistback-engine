"""
Hybrid Tokenizer: Natural Language (Estonian) + MiniStackVM Abstract Bytecode
=============================================================================
Supports:
  1. Standard character-level Estonian natural language (letters, digits, punctuation, whitespace).
  2. Special control tokens: <EXEC_VM>, </EXEC_VM>, <RES>, <pad>, <s>, </s>, <unk>.
  3. VM OpCode tokens: <OP:LOAD_ARG>, <OP:PUSH_CONST>, <OP:ADD>, <OP:SUB>, <OP:MUL>, etc.
  4. Register argument tokens: <ARG:0>, <ARG:1>, <ARG:2>, <ARG:3>.
  (Zero static <NUM:x> tokens to prevent memorization and token leakage).
"""

import os
import re
import sys
from typing import List, Dict, Tuple, Optional

from mini_vm.ops import OpCode, OPCODE_NAMES, NAME_TO_OPCODE


class HybridTokenizer:
    """
    Compact symbolic tokenizer (124 tokens) that maps dynamic numeric operands
    to abstract register slots (<ARG:0> ... <ARG:3>).
    """
    TOKEN_REGEX = re.compile(r"(<[A-Za-z0-9_:\-/]+>|.)")

    def __init__(self, num_registers: int = 4):
        self.pad_token = "<pad>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        
        self.exec_vm_token = "<EXEC_VM>"
        self.end_vm_token = "</EXEC_VM>"
        self.res_token = "<RES>"

        vocab_list = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.exec_vm_token,
            self.end_vm_token,
            self.res_token
        ]

        # 1. Estonian alphabet, numbers, and punctuation
        char_set = (
            "abcdefghijklmnoprsštzžuvõäöü"
            "ABCDEFGHIJKLMNOPRSŠTZŽUVÕÄÖÜ"
            "qwxyQWXY"
            "0123456789"
            " .,:;!?()[]{}+-*/%=<>'\"_\n\t"
        )
        for char in sorted(list(set(char_set))):
            if char not in vocab_list:
                vocab_list.append(char)

        # 2. MiniStackVM OpCodes
        for op_name in OPCODE_NAMES.values():
            tok = f"<OP:{op_name}>"
            if tok not in vocab_list:
                vocab_list.append(tok)

        # 3. Dynamic Argument Registers (<ARG:0> .. <ARG:3>)
        self.num_registers = num_registers
        for r in range(num_registers):
            tok = f"<ARG:{r}>"
            if tok not in vocab_list:
                vocab_list.append(tok)

        self.vocab: Dict[str, int] = {token: idx for idx, token in enumerate(vocab_list)}
        self.inv_vocab: Dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size: int = len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        """Splits string into characters and multi-character tags (<...>)."""
        return self.TOKEN_REGEX.findall(text)

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """Encodes string to a list of integer token IDs."""
        raw_tokens = self.tokenize(text)
        token_ids = []
        if add_special:
            token_ids.append(self.vocab[self.bos_token])
        
        for tok in raw_tokens:
            token_ids.append(self.vocab.get(tok, self.vocab[self.unk_token]))
            
        if add_special:
            token_ids.append(self.vocab[self.eos_token])
        return token_ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decodes integer token IDs back to a string."""
        tokens = []
        for tid in token_ids:
            tok = self.inv_vocab.get(tid, self.unk_token)
            if skip_special and tok in (self.pad_token, self.bos_token, self.eos_token):
                continue
            tokens.append(tok)
        return "".join(tokens)

    def bytecode_to_tokens(self, bytecode: List[int]) -> List[str]:
        """Converts raw VM bytecode integers into symbolic tokens."""
        tokens = []
        i = 0
        while i < len(bytecode):
            byte = bytecode[i]
            if byte in OPCODE_NAMES:
                name = OPCODE_NAMES[byte]
                tokens.append(f"<OP:{name}>")
                if byte in (OpCode.LOAD_ARG, OpCode.JMP_IF_FALSE):
                    i += 1
                    if i < len(bytecode):
                        arg_idx = bytecode[i]
                        tokens.append(f"<ARG:{arg_idx}>")
                elif byte == OpCode.PUSH_CONST:
                    i += 1
                    if i < len(bytecode):
                        tokens.append(str(bytecode[i]))
            else:
                tokens.append(str(byte))
            i += 1
        return tokens

    def tokens_to_bytecode(self, tokens: List[str]) -> List[int]:
        """Parses symbolic tokens back into raw VM bytecode integers."""
        bc = []
        for tok in tokens:
            if tok.startswith("<OP:") and tok.endswith(">"):
                name = tok[4:-1]
                if name in NAME_TO_OPCODE:
                    bc.append(NAME_TO_OPCODE[name])
            elif tok.startswith("<ARG:") and tok.endswith(">"):
                try:
                    reg_idx = int(tok[5:-1])
                    bc.append(reg_idx)
                except ValueError:
                    pass
            elif tok.isdigit():
                bc.append(int(tok))
        return bc
