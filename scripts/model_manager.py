#!/usr/bin/env python3
"""
Feistback Model Manager CLI
===========================
Management, verification, installation, and removal of local specialist models
governed by declarative `config.json` specifications and ModelRegistry.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional
import zipfile

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from runtime.registry import ModelRegistry


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Formats rows and headers into a clean CLI ASCII table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    # Header and separator
    header_line = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    separator_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

    out_lines = [header_line, separator_line]
    for row in rows:
        line = " | ".join(f"{str(cell):<{widths[i]}}" for i, cell in enumerate(row))
        out_lines.append(line)

    return "\n".join(out_lines)


def cmd_list(args: argparse.Namespace, registry: ModelRegistry) -> int:
    """Lists all registered models in a formatted terminal table."""
    registry.scan_models()
    models = registry.list_models()

    print("=" * 85)
    print(f"FEISTBACK REGISTERED MODELS (Location: {registry.models_dir})")
    print("=" * 85)

    if not models:
        print(f"No registered models found in {registry.models_dir}.")
        print("Use 'install <source>' to install a new model.")
        return 0

    headers = [
        "Mudeli nimi",
        "Roll",
        "Dim (d_model)",
        "Kihid",
        "S\u00f5nastik",
        "M\u00e4lu",
        "Ketta maht (MB)",
    ]

    rows = []
    for m in models:
        name = m.get("name", "unknown")
        role = m.get("role", "general")
        d_model = str(m.get("d_model", "-"))
        num_layers = str(m.get("num_layers", "-"))
        vocab_size = str(m.get("vocab_size", "-"))
        memory_type = str(m.get("memory_type", "raf")).upper()
        size_mb = f"{registry.get_model_size_mb(name):.2f}"

        rows.append([name, role, d_model, num_layers, vocab_size, memory_type, size_mb])

    print(format_table(headers, rows))
    print("=" * 85)
    print(f"Kokku: {len(models)} registreeritud mudel(it)\n")
    return 0


def cmd_verify(args: argparse.Namespace, registry: ModelRegistry) -> int:
    """Verifies a model's files, config, and performs a CPU dry-run load."""
    model_name = args.model_name
    print(f"Kontrollin mudelit: '{model_name}'...")
    print("-" * 60)

    res = registry.verify_model(model_name)

    # 1. Config check
    if res["config_valid"]:
        print(f"[OK] config.json leitud ja valideeritud ({res['model_dir']})")
    else:
        print(f"[VIGA] config.json puudulik v\u00f5i vigane: {res.get('error')}")
        if res.get("missing_fields"):
            print(f"       Puuduvad v\u00e4ljad: {res['missing_fields']}")
        return 1

    # 2. File checks
    for f in res.get("files", []):
        f_type = f["type"]
        f_name = f["name"]
        f_size = f["size_mb"]
        if f["exists"]:
            if f_type == "weights":
                print(f"[OK] Kaalude fail ({f_name}) leitud ({f_size:.2f} MB)")
            elif f_type == "tokenizer":
                print(f"[OK] Tokenisaator ({f_name}) leitud ({f_size:.2f} MB)")
            elif f_type == "vocab_map":
                print(f"[OK] S\u00f5nastiku kaart ({f_name}) leitud ({f_size:.2f} MB)")
            elif f_type == "pruned_weights":
                print(f"[OK] K\u00e4rbitud kaalud ({f_name}) leitud ({f_size:.2f} MB)")
            else:
                print(f"[OK] Fail ({f_name}) leitud ({f_size:.2f} MB)")
        else:
            print(f"[VIGA] N\u00f5utud fail puudub: {f['path']}")
            return 1

    # 3. Tokenizer loading check
    if res.get("tokenizer_loaded"):
        print("[OK] Tokenisaator edukalt laetud m\u00e4llu")

    # 4. Dry-run check
    if res["dry_run_success"]:
        print("[OK] CPU kuivk\u00e4ivitus: PureFeistelLM mudeli kaalud initsialiseeritud vigadeta")
    else:
        print(f"[VIGA] CPU kuivk\u00e4ivitus eba\u00f5nnestus: {res.get('dry_run_error')}")
        return 1

    print("-" * 60)
    print(f"=> Mudel '{model_name}' on 100% t\u00f6\u00f6korras! (Kogumaht: {res['total_size_mb']:.2f} MB)\n")
    return 0


def _find_model_dir(base_dir: Path) -> Optional[Path]:
    """Finds directory containing config.json within base_dir."""
    if (base_dir / "config.json").exists():
        return base_dir

    for item in base_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            return item

    # Recursive fallback
    for p in base_dir.rglob("config.json"):
        return p.parent

    return None


def cmd_install(args: argparse.Namespace, registry: ModelRegistry) -> int:
    """Installs a model from a local directory or .zip archive."""
    source_path = Path(args.source_path).resolve()
    if not source_path.exists():
        print(f"[VIGA] Allikat '{args.source_path}' ei leitud!")
        return 1

    print(f"Alustan mudeli paigaldamist allikast: {source_path}")

    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    try:
        if source_path.is_file() and source_path.suffix.lower() in (".zip", ".feist"):
            print(f"Arhiivi ({source_path.suffix}) lahtipakkimine ajutisse kausta...")
            temp_dir_obj = tempfile.TemporaryDirectory()
            extract_target = Path(temp_dir_obj.name)
            with zipfile.ZipFile(source_path, "r") as zip_ref:
                zip_ref.extractall(extract_target)
            model_src_dir = _find_model_dir(extract_target)
            if not model_src_dir:
                print(f"[VIGA] Arhiivist ({source_path.name}) ei leitud 'config.json' faili!")
                return 1
        elif source_path.is_dir():
            model_src_dir = _find_model_dir(source_path)
            if not model_src_dir:
                print(f"[VIGA] Kaustas '{source_path}' ei leitud 'config.json' faili!")
                return 1
        else:
            print(f"[VIGA] Toetamata allika t\u00fc\u00fcp: {source_path}. N\u00f5utud on kaust, .feist v\u00f5i .zip fail.")
            return 1

        # Pre-installation verification
        print("Verifitseerin mudeli terviklikkust enne paigaldamist...")
        check_res = registry.verify_model(model_src_dir)
        if not check_res["valid"]:
            print(f"[VIGA] Mudeli verifitseerimine eba\u00f5nnestus enne paigaldust: {check_res.get('error')}")
            return 1

        model_name = check_res["model_name"]
        print(f"[OK] Allikas on verifitseeritud. Mudeli nimi: '{model_name}'")

        target_dir = registry.models_dir / model_name
        if target_dir.exists():
            if not args.overwrite:
                print(f"Hoiatus: Mudel '{model_name}' on kaustas {target_dir} juba olemas.")
                try:
                    choice = input("Kas kirjutada \u00fcle? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "n"
                if choice not in ("y", "yes"):
                    print("Paigaldus t\u00fchistatud kasutaja poolt.")
                    return 0
            print(f"Kirjutan \u00fcle olemasoleva kausta: {target_dir}")
            shutil.rmtree(target_dir)

        print(f"Kopeerin mudelifailid sihtkohta: {target_dir}...")
        shutil.copytree(model_src_dir, target_dir)

        # Re-scan registry and verify target location
        registry.scan_models()
        final_check = registry.verify_model(model_name)
        if not final_check["valid"]:
            print(f"[VIGA] Paigaldusj\u00e4rgne kontroll eba\u00f5nnestus: {final_check.get('error')}")
            return 1

        print("=" * 60)
        print(f"=> Mudel '{model_name}' edukalt paigaldatud ja registreeritud!")
        print(f"   Asukoht : {target_dir}")
        print(f"   Kogumaht: {final_check['total_size_mb']:.2f} MB")
        print("=" * 60)
        return 0

    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def cmd_pack(args: argparse.Namespace, registry: ModelRegistry) -> int:
    """Packages a model into a .feist distribution archive."""
    target = args.model_name_or_dir
    out_path = args.output_path
    print(f"Pakin mudelit '{target}' .feist arhiivi...")

    try:
        res = registry.package_model(target, output_path=out_path)
    except Exception as e:
        print(f"[VIGA] Mudeli pakkimine eba\u00f5nnestus: {e}")
        return 1

    print("=" * 70)
    print("FEISTBACK MUDELI PAKKIMISE RAPORT (.feist arhiiv)")
    print("=" * 70)
    print(f"Mudel          : {res['model_name']}")
    print(f"Sihtarhiiv     : {res['archive_path']}")
    print(f"Arhiivi suurus : {res['archive_size_mb']:.2f} MB")
    print("Pakitud failid :")
    for f in res.get("files_packed", []):
        print(f"  - {f['name']} ({f['size_mb']:.2f} MB)")
    print("=" * 70)
    print(f"=> Mudel '{res['model_name']}' on edukalt pakitud: {res['archive_path']}\n")
    return 0


def cmd_remove(args: argparse.Namespace, registry: ModelRegistry) -> int:
    """Removes a model directory from models/."""
    model_name = args.model_name
    registry.scan_models()

    if model_name not in registry._models:
        print(f"[VIGA] Mudelit '{model_name}' ei leitud registrist ({registry.models_dir})!")
        return 1

    model_dir = Path(registry.get_config(model_name)["_dir"])
    if not model_dir.exists():
        print(f"[VIGA] Mudeli kausta ei leitud kettalt: {model_dir}")
        return 1

    if not args.yes:
        try:
            confirm = input(
                f"Kas oled kindel, et soovid mudeli '{model_name}' ({model_dir}) j\u00e4\u00e4davalt kustutada? [y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = "n"
        if confirm not in ("y", "yes"):
            print("Kustutamine t\u00fchistatud.")
            return 0

    print(f"Kustutan mudelikausta: {model_dir}...")
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        print(f"[VIGA] Kausta kustutamine eba\u00f5nnestus: {e}")
        return 1

    registry.scan_models()
    print(f"[OK] Mudel '{model_name}' on edukalt eemaldatud.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feistback Model Manager: Manage, verify, install, and remove local specialist models."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Optional custom models directory (default: models/).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. list
    subparsers.add_parser("list", help="List all registered models.")

    # 2. verify
    verify_p = subparsers.add_parser("verify", help="Verify model integrity and CPU dry-run.")
    verify_p.add_argument("model_name", type=str, help="Name of model to verify.")

    # 3. pack
    pack_p = subparsers.add_parser("pack", help="Package a model into a .feist distribution archive.")
    pack_p.add_argument("model_name_or_dir", type=str, help="Model name in registry or path to model directory.")
    pack_p.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default=None,
        help="Optional output archive path (default: dist/<name>.feist).",
    )

    # 4. install
    install_p = subparsers.add_parser("install", help="Install a model from directory, .feist, or .zip archive.")
    install_p.add_argument("source_path", type=str, help="Path to model directory, .feist, or .zip file.")
    install_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing model without prompting.",
    )

    # 5. remove
    remove_p = subparsers.add_parser("remove", help="Remove an installed model.")
    remove_p.add_argument("model_name", type=str, help="Name of model to remove.")
    remove_p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm deletion automatically without prompt.",
    )

    args = parser.parse_args()
    registry = ModelRegistry(models_dir=args.models_dir)

    if args.command == "list":
        return cmd_list(args, registry)
    elif args.command == "verify":
        return cmd_verify(args, registry)
    elif args.command == "pack":
        return cmd_pack(args, registry)
    elif args.command == "install":
        return cmd_install(args, registry)
    elif args.command == "remove":
        return cmd_remove(args, registry)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
