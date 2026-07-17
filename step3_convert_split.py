"""
Étape 3 — Préparation du Training & Conversion (Page 25)
=========================================================
- Split du dataset annoté : 80 % train / 20 % dev.
- Conversion des annotations JSON en binaires spaCy via `DocBin` :
    train.spacy  (entraînement)
    dev.spacy    (évaluation)

Les spans invalides (offsets qui ne tombent pas sur des frontières de
tokens) sont réalignés avec `alignment_mode="contract"` ; ceux qui
restent irrécupérables sont écartés et comptés.

Usage :
    python step3_convert_split.py --input annotations.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def build_docbin(nlp, data: list, path: Path) -> None:
    """Convertit [(texte, {"entities": [...]}), ...] en fichier .spacy."""
    db = DocBin()
    dropped = 0

    for text, ann in data:
        doc = nlp.make_doc(text)
        ents = []
        occupied = set()

        for start, end, label in ann["entities"]:
            # char_span aligne les offsets caractères sur les tokens.
            # "contract" rétrécit le span jusqu'à la frontière de token
            # la plus proche au lieu de le rejeter d'emblée.
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None or any(t.i in occupied for t in span):
                dropped += 1
                continue
            ents.append(span)
            occupied.update(t.i for t in span)

        doc.ents = ents
        db.add(doc)

    db.to_disk(path)
    print(f"[Étape 3] {path} : {len(data)} docs écrits, {dropped} spans écartés.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split 80/20 + conversion DocBin")
    parser.add_argument("--input", type=Path, default=Path("annotations.json"))
    parser.add_argument("--train-out", type=Path, default=Path("train.spacy"))
    parser.add_argument("--dev-out", type=Path, default=Path("dev.spacy"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Tokenizer anglais "blank" : suffit pour la conversion (pas besoin
    # du modèle complet ici, seul le découpage en tokens compte).
    nlp = spacy.blank("en")

    data = json.loads(args.input.read_text(encoding="utf-8"))

    # Mélange reproductible puis split 80/20
    random.seed(args.seed)
    random.shuffle(data)
    split_idx = int(len(data) * 0.8)
    train_data, dev_data = data[:split_idx], data[split_idx:]

    print(f"[Étape 3] Split : {len(train_data)} train / {len(dev_data)} dev.")
    build_docbin(nlp, train_data, args.train_out)
    build_docbin(nlp, dev_data, args.dev_out)


if __name__ == "__main__":
    main()
