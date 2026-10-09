"""Vertical probe packs.

The generic fabrication templates in :mod:`ancora.probe` are deliberately
industry-neutral, which makes them weak evidence for a real audit: a clinic
does not care whether the bot invents a shipping policy, it cares whether the
bot invents a cancellation window or a price for a procedure.

A pack is a plain JSON file of fabrications plausible for one industry. Because
"plausible" is exactly what makes a hallucination dangerous, the quality of
these strings is the quality of the audit.

    ancora probe --kb ./base --pack clinica
    ancora probe --kb ./base --pack ./meu-pack.json
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["load_pack", "available_packs", "PACK_DIR"]

PACK_DIR = Path(__file__).parent


def available_packs() -> list[str]:
    """Names of the packs bundled with the library."""
    return sorted(p.stem for p in PACK_DIR.glob("*.json"))


def load_pack(name_or_path: str) -> list[str]:
    """Load fabrication templates by bundled name or by file path.

    Raises ``FileNotFoundError`` naming the available packs, so a typo is
    self-correcting rather than a silent empty suite.
    """
    candidate = PACK_DIR / f"{name_or_path}.json"
    path = candidate if candidate.is_file() else Path(name_or_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no pack named {name_or_path!r}; bundled packs: {', '.join(available_packs())}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    templates = data.get("fabrications") if isinstance(data, dict) else data
    if not isinstance(templates, list) or not all(isinstance(t, str) for t in templates):
        raise ValueError(f"{path} must contain a list of strings under 'fabrications'")
    if not templates:
        raise ValueError(f"{path} contains no fabrication templates")
    return templates
