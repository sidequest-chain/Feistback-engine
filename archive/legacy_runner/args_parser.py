"""
Argument Parser: Dynamic Numeric Extraction
===========================================
Extracts integer operands from prompt text to populate the MiniStackVM register bank.
"""

import re
from typing import List


def extract_numeric_args(prompt: str) -> List[int]:
    """
    Extracts integers from prompt string in order of appearance.
    
    Example:
      'Küsimus: liida arvud 1500 ja 2500.' -> [1500, 2500]
    """
    return [int(x) for x in re.findall(r"\b\d+\b", prompt)]
