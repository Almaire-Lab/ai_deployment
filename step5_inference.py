"""
Étape 5 — Inférence (Pages 27-28)
==================================
Charge le modèle fine-tuné et l'applique sur le RESTE du corpus
(articles NON utilisés pour l'annotation).

Sortie : `extracted_entities.jsonl` — une ligne JSON par article :
    {"id": ..., "entities": [{"text", "label", "start", "end"}, ...]}

Usage :
    python step5_inference.py --model ./output/model-best --corpus texts_clean.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

import spacy

# =====================================================================
# CONFIGURATION LOGGING
# =====================================================================

class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour une meilleure lisibilité."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[0m',        # Reset
        'SUCCESS': '\033[32m',    # Vert
        'WARNING': '\033[33m',    # Jaune
        'ERROR': '\033[31m',      # Rouge
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logger(log_file: Path = Path("inference_pipeline.log")) -> logging.Logger:
    """Configure un logger avec file + console output."""
    logger = logging.getLogger("Inference_Pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Format avec timestamp
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Handler console (avec couleurs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier (sans couleurs)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Ajouter une méthode custom pour "SUCCESS"
    logging.addLevelName(25, "SUCCESS")
    def log_success(self, message, *args, **kwargs):
        if self.isEnabledFor(25):
            self._log(25, message, args, **kwargs)
    logger.success = log_success.__get__(logger, logger.__class__)
    
    return logger


# =====================================================================
# FONCTIONS D'INFÉRENCE
# =====================================================================

def load_model(logger: logging.Logger, model_path: Path) -> spacy.Language:
    """Charge le modèle fine-tuné."""
    try:
        logger.info(f"Chargement du modèle : {model_path}")
        
        if not model_path.exists():
            logger.error(f"Modèle '{model_path}' introuvable")
            raise FileNotFoundError(f"Modèle {model_path} not found")
        
        nlp = spacy.load(model_path)
        logger.success(f"✓ Modèle chargé avec succès")
        return nlp
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise


def load_corpus(logger: logging.Logger, corpus_path: Path) -> list:
    """Charge le corpus complet."""
    try:
        logger.info(f"Chargement du corpus : {corpus_path}")
        
        if not corpus_path.exists():
            logger.error(f"Corpus '{corpus_path}' introuvable")
            raise FileNotFoundError(f"Corpus {corpus_path} not found")
        
        records = json.loads(corpus_path.read_text(encoding="utf-8"))
        logger.success(f"✓ {len(records)} articles chargés")
        return records
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du corpus: {str(e)}")
        raise


def load_annotated_ids(logger: logging.Logger, annotated_ids_path: Path) -> set:
    """Charge l'ensemble des IDs annotés."""
    try:
        if not annotated_ids_path.exists():
            logger.warning(f"Fichier '{annotated_ids_path}' introuvable, aucun ID exclu")
            return set()
        
        annotated_ids = set(json.loads(annotated_ids_path.read_text(encoding="utf-8")))
        logger.success(f"✓ {len(annotated_ids)} IDs annotés chargés (seront exclus)")
        return annotated_ids
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des IDs annotés: {str(e)}")
        raise


def filter_remaining_documents(logger: logging.Logger, records: list, annotated_ids: set) -> list:
    """Filtre les documents non annotés."""
    try:
        logger.info("Filtrage des documents non annotés…")
        
        remaining = [r for r in records if r.get("id") not in annotated_ids]
        excluded = len(records) - len(remaining)
        
        logger.success(f"✓ {len(remaining)} articles à traiter ({excluded} exclus)")
        
        if len(remaining) == 0:
            logger.warning("Aucun article à traiter !")
        
        return remaining
        
    except Exception as e:
        logger.error(f"Erreur lors du filtrage: {str(e)}")
        raise


def run_inference(logger: logging.Logger, nlp: spacy.Language, remaining: list, 
                  output_path: Path, batch_size: int = 64) -> tuple[int, int]:
    """
    Lance l'inférence sur les documents restants.
    Retourne (nombre de documents traités, nombre d'entités extraites).
    """
    try:
        logger.info(f"Démarrage de l'inférence (batch_size={batch_size})…")
        
        texts = (r.get("text", "") for r in remaining)
        n_entities = 0
        n_documents = 0
        failed_documents = 0
        
        with output_path.open("w", encoding="utf-8") as out:
            for rec, doc in zip(remaining, nlp.pipe(texts, batch_size=batch_size)):
                try:
                    entities = [
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "start": ent.start_char,
                            "end": ent.end_char,
                        }
                        for ent in doc.ents
                    ]
                    n_entities += len(entities)
                    n_documents += 1
                    
                    out.write(
                        json.dumps(
                            {"id": rec.get("id"), "entities": entities},
                            ensure_ascii=False
                        ) + "\n"
                    )
                    
                    if n_documents % 100 == 0:
                        logger.debug(f"Traitement: {n_documents}/{len(remaining)} documents, {n_entities} entités")
                        
                except Exception as e:
                    logger.warning(f"Erreur lors du traitement du document {rec.get('id')}: {str(e)}")
                    failed_documents += 1
        
        logger.success(f"✓ Inférence complétée: {n_documents} documents traités")
        logger.success(f"✓ Total: {n_entities} entités extraites")
        
        if failed_documents > 0:
            logger.warning(f"{failed_documents} documents n'ont pas pu être traités")
        
        return n_documents, n_entities
        
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence: {str(e)}")
        raise


def compute_label_statistics(logger: logging.Logger, output_path: Path) -> dict:
    """Calcule les statistiques par label."""
    try:
        logger.info("Calcul des statistiques par label…")
        
        counts: Counter = Counter()
        
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                for ent in data.get("entities", []):
                    counts[ent.get("label", "UNKNOWN")] += 1
        
        logger.success(f"✓ Statistiques calculées")
        
        # Log les résultats
        logger.info("Distribution des labels:")
        for label, n in counts.most_common():
            logger.info(f"  {label:15s} : {n:6d} entités")
        
        return dict(counts)
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des statistiques: {str(e)}")
        return {}


# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inférence NER sur le corpus restant (documents non annotés)"
    )
    parser.add_argument(
        "--model", type=Path, default=Path("./output/model-best"),
        help="Chemin du modèle fine-tuné"
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("texts_clean.json"),
        help="Fichier JSON contenant le corpus complet"
    )
    parser.add_argument(
        "--annotated-ids", type=Path, default=Path("annotated_ids.json"),
        help="Fichier JSON contenant les IDs annotés (à exclure)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("extracted_entities.jsonl"),
        help="Fichier JSONL de sortie"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Taille des lots pour l'inférence"
    )
    
    args = parser.parse_args()

    # ===================== SETUP LOGGING =====================
    logger = setup_logger()
    start_time = datetime.now()
    
    logger.info("="*70)
    logger.info("PIPELINE INFÉRENCE NER")
    logger.info("="*70)
    logger.info(f"Modèle           : {args.model}")
    logger.info(f"Corpus           : {args.corpus}")
    logger.info(f"IDs annotés      : {args.annotated_ids}")
    logger.info(f"Sortie           : {args.output}")
    logger.info(f"Batch size       : {args.batch_size}")
    logger.info(f"Démarrage        : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    try:
        # ===================== ÉTAPE 1: Charger le modèle =====================
        logger.info("\n[ÉTAPE 1/5] Chargement du modèle fine-tuné")
        nlp = load_model(logger, args.model)
        step1_ok = True

        # ===================== ÉTAPE 2: Charger le corpus =====================
        logger.info("\n[ÉTAPE 2/5] Chargement du corpus")
        records = load_corpus(logger, args.corpus)
        step2_ok = len(records) > 0

        # ===================== ÉTAPE 3: Charger les IDs annotés =====================
        logger.info("\n[ÉTAPE 3/5] Chargement des IDs annotés")
        annotated_ids = load_annotated_ids(logger, args.annotated_ids)
        step3_ok = True

        # ===================== ÉTAPE 4: Filtrer les documents =====================
        logger.info("\n[ÉTAPE 4/5] Filtrage des documents non annotés")
        remaining = filter_remaining_documents(logger, records, annotated_ids)
        step4_ok = len(remaining) > 0

        if not step4_ok:
            logger.warning("Aucun document à traiter, pipeline terminé")
            sys.exit(0)

        # ===================== ÉTAPE 5: Inférence =====================
        logger.info("\n[ÉTAPE 5/5] Inférence NER")
        n_docs, n_ents = run_inference(logger, nlp, remaining, args.output, args.batch_size)
        step5_ok = n_docs > 0

        # ===================== ÉTAPE 6: Statistiques =====================
        logger.info("\n[BONUS] Statistiques par label")
        stats = compute_label_statistics(logger, args.output)
        step6_ok = True

        # ===================== RÉSUMÉ FINAL =====================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("RÉSUMÉ DU PIPELINE")
        logger.info("="*70)
        logger.success(f"✓ [ÉTAPE 1] Chargement modèle        : RÉUSSI")
        logger.success(f"✓ [ÉTAPE 2] Chargement corpus       : RÉUSSI ({len(records)} articles)")
        logger.success(f"✓ [ÉTAPE 3] Chargement IDs annotés  : RÉUSSI ({len(annotated_ids)} IDs)")
        logger.success(f"✓ [ÉTAPE 4] Filtrage documents      : RÉUSSI ({len(remaining)} à traiter)")
        logger.success(f"✓ [ÉTAPE 5] Inférence NER           : RÉUSSI ({n_docs} annotés, {n_ents} entités)")
        logger.success(f"✓ [BONUS]  Statistiques             : RÉUSSI")
        logger.info("="*70)
        logger.success(f"Pipeline complété en {duration:.2f}s")
        logger.success(f"Fichiers générés:")
        logger.success(f"  - {args.output}")
        logger.success(f"  - inference_pipeline.log")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.error("\n" + "="*70)
        logger.error("ERREUR - Pipeline interrompu")
        logger.error("="*70)
        logger.error(f"Durée avant erreur: {duration:.2f}s")
        logger.error(f"Erreur: {str(e)}")
        logger.error("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()