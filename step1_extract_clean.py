"""
Étape 1 — Extraction et Nettoyage (Page 22)
============================================
Charge le corpus JSON (articles TASS), isole le champ "text" de chaque
article SANS fusionner les articles entre eux, puis applique un
nettoyage minimal (espaces superflus, caractères parasites).

Sortie : `texts_clean.json` — liste d'objets {"id": ..., "text": ...}
(l'id est conservé pour la traçabilité OSINT : on doit toujours pouvoir
remonter à l'article source).

Usage :
    python step1_extract_clean.py --input data_set.json --output texts_clean.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------

# Caractères invisibles / parasites fréquents dans les exports web :
# espaces insécables, zero-width spaces, soft hyphens, BOM, etc.
_PARASITE_CHARS = re.compile(r"[\u00a0\u200b\u200c\u200d\u2060\ufeff\u00ad]")

# Espaces multiples (y compris tabulations) -> un seul espace
_MULTI_SPACES = re.compile(r"[ \t]+")

# Plus de deux sauts de ligne consécutifs -> deux maximum
_MULTI_NEWLINES = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Nettoyage minimal et NON destructif d'un texte d'article.

    Important pour le NER : on évite tout nettoyage agressif (pas de
    lowercasing, pas de suppression de ponctuation) car la casse et la
    
    ponctuation sont des indices précieux pour la reconnaissance
    d'entités (« 58th Army », « S-400 », etc.).
    """
    # Normalisation Unicode (compose les accents, unifie les variantes)
    text = unicodedata.normalize("NFC", raw)

    # Remplace les caractères parasites par un espace simple
    text = _PARASITE_CHARS.sub(" ", text)

    # Compacte les espaces horizontaux, puis les sauts de ligne
    text = _MULTI_SPACES.sub(" ", text)
    text = _MULTI_NEWLINES.sub("\n\n", text)

    # Supprime les espaces en début/fin de chaque ligne, puis du texte entier
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_texts(input_path: Path) -> list[dict]:
    """Charge le JSON et retourne une liste [{"id", "text"}, ...].

    - Chaque article reste un document indépendant (pas de fusion).
    - Les articles sans texte exploitable sont écartés (et comptés).
    """
    with input_path.open(encoding="utf-8") as f:
        articles = json.load(f)

    records: list[dict] = []
    skipped = 0

    for article in articles:
        raw = article.get("text") or ""
        cleaned = clean_text(raw)
        if not cleaned:
            skipped += 1
            continue
        records.append({"id": article.get("id"), "text": cleaned})

    print(f"[Étape 1] {len(records)} textes extraits, {skipped} articles vides écartés.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction + nettoyage du corpus TASS")
    parser.add_argument("--input", type=Path, default=Path("data_set.json"))
    parser.add_argument("--output", type=Path, default=Path("texts_clean.json"))
    args = parser.parse_args()

    records = extract_texts(args.input)

    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Étape 1] Corpus nettoyé écrit dans {args.output}")


if __name__ == "__main__":
    main()
