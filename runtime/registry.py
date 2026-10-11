"""
Feistback Model Registry
========================
Discovers, introspects, and instantiates registered PureFeistelLM models and tokenizers
based on declarative `config.json` specifications in `models/`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tokenizers import Tokenizer

from runtime.models.pure_feistel_lm import PureFeistelLM


class ModelRegistry:
    """
    Registry for managing and instantiating modular Feistback specialist models.
    Scans `models_dir` for directory configurations adhering to the Feistback schema.
    """

    def __init__(self, models_dir: Optional[str] = None) -> None:
        if models_dir is None:
            repo_root = Path(__file__).resolve().parent.parent
            self.models_dir = repo_root / "models"
        else:
            p = Path(models_dir)
            if not p.is_absolute():
                repo_root = Path(__file__).resolve().parent.parent
                p = repo_root / models_dir
            self.models_dir = p

        self._models: Dict[str, Dict[str, Any]] = {}
        self.scan_models()

    def scan_models(self) -> Dict[str, Dict[str, Any]]:
        """Scans the models directory and discovers all subdirectories with a valid config.json."""
        self._models.clear()
        if not self.models_dir.exists():
            return self._models

        for entry in sorted(self.models_dir.iterdir()):
            if entry.is_dir():
                cfg_path = entry / "config.json"
                if cfg_path.exists():
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        model_name = cfg.get("name", entry.name)
                        cfg["_dir"] = str(entry)
                        cfg["_config_path"] = str(cfg_path)
                        self._models[model_name] = cfg
                    except Exception as e:
                        print(f"Warning: Failed to load config from {cfg_path}: {e}")

        return self._models

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns a list of all registered models with their metadata."""
        return [
            {k: v for k, v in cfg.items() if not k.startswith("_")}
            for cfg in self._models.values()
        ]

    def get_config(self, model_name: str) -> Dict[str, Any]:
        """Returns the declarative configuration dictionary for a given model."""
        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' not found in registry. Available: {list(self._models.keys())}"
            )
        return self._models[model_name]

    def load_model(
        self, model_name: str, device: str = "cpu"
    ) -> Tuple[PureFeistelLM, Dict[str, Any]]:
        """
        Initializes PureFeistelLM directly from declarative config.json parameters
        (without hardcoded layers or dimensions) and loads corresponding trained weights.
        """
        cfg = self.get_config(model_name)
        model_dir = Path(cfg["_dir"])

        weights_file = cfg.get("weights")
        if not weights_file:
            raise ValueError(f"No 'weights' specified in config for model '{model_name}'")

        weights_path = Path(weights_file)
        if not weights_path.is_absolute():
            weights_path = model_dir / weights_file

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found at: {weights_path}")

        # Resolve optional pruned_weights and vocab_map paths
        pruned_weights_path = None
        if "pruned_weights" in cfg:
            pw = Path(cfg["pruned_weights"])
            pruned_weights_path = str(pw if pw.is_absolute() else (model_dir / pw))

        vocab_map_path = None
        if "vocab_map" in cfg:
            vm = Path(cfg["vocab_map"])
            vocab_map_path = str(vm if vm.is_absolute() else (model_dir / vm))

        # Check torch_dtype
        dtype_str = cfg.get("torch_dtype", "bfloat16" if pruned_weights_path else "float32")
        if dtype_str == "bfloat16" and torch.cuda.is_available() and device.startswith("cuda"):
            torch_dtype = torch.bfloat16
        elif dtype_str == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Inspect checkpoint metadata for fallback synchronization
        ckpt = torch.load(weights_path, map_location=device)
        ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

        vocab_size = cfg.get("vocab_size", ckpt_cfg.get("vocab_size", 16384 if pruned_weights_path else 1327))
        d_model = ckpt.get("d_model", ckpt_cfg.get("d_model", cfg.get("d_model", 896 if pruned_weights_path else 512))) if isinstance(ckpt, dict) else cfg.get("d_model", 896 if pruned_weights_path else 512)
        num_layers = cfg.get("num_layers", ckpt.get("num_layers", ckpt_cfg.get("num_layers", 12))) if isinstance(ckpt, dict) else cfg.get("num_layers", 12)
        state_dim = cfg.get("state_dim", ckpt.get("state_dim", ckpt_cfg.get("state_dim", 128))) if isinstance(ckpt, dict) else cfg.get("state_dim", 128)
        max_seq_len = cfg.get("max_seq_len", ckpt_cfg.get("max_seq_len", 256))
        memory_type = cfg.get("memory_type", ckpt.get("memory_type", ckpt_cfg.get("memory_type", "raf"))) if isinstance(ckpt, dict) else cfg.get("memory_type", "raf")
        num_heads = cfg.get("num_heads", ckpt_cfg.get("num_heads", 4))
        key_dim = cfg.get("key_dim", ckpt_cfg.get("key_dim", 16))
        value_dim = cfg.get("value_dim", ckpt_cfg.get("value_dim", 16))

        model = PureFeistelLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            state_dim=state_dim,
            max_seq_len=max_seq_len,
            memory_type=memory_type,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            torch_dtype=torch_dtype,
            device=device,
            from_pretrained_qwen=False,
            pruned_weights_path=pruned_weights_path,
            vocab_map_path=vocab_map_path,
        )

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_backbone(ckpt)

        model.eval()
        return model, cfg

    def load_tokenizer(self, model_name: str) -> Optional[Tokenizer]:
        """Loads Tokenizer if specified in model config."""
        cfg = self.get_config(model_name)
        tok_file = cfg.get("tokenizer")
        if not tok_file:
            return None

        model_dir = Path(cfg["_dir"])
        tok_path = Path(tok_file)
        if not tok_path.is_absolute():
            tok_path = model_dir / tok_file

        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at: {tok_path}")

        return Tokenizer.from_file(str(tok_path))
