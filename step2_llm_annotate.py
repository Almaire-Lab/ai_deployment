"""
Étape 2 — Pré-annotation NER via spaCy
=====================================
Utilise un modèle spaCy pré-entraîné pour générer des pré-annotations NER.

Format de sortie (format spaCy "train data") :
    ("Le texte ici", {"entities": [(start, end, "LABEL")]})

Labels : WEAPON, MIL_UNIT, MIL_ORG

Usage :
    python step2_spacy_annotate.py --input texts_clean.json --output annotations.json --model en_core_web_lg
"""

from __future__ import annotations
import argparse
import json
import random
import re
import logging
import sys
from pathlib import Path
from datetime import datetime
import spacy
from spacy.matcher import PhraseMatcher, Matcher

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


def setup_logger(log_file: Path = Path("annotation_pipeline.log")) -> logging.Logger:
    """Configure un logger avec file + console output."""
    logger = logging.getLogger("NER_Pipeline")
    logger.setLevel(logging.DEBUG)
    
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
# CONFIGURATION MILITAIRE
# =====================================================================

# =====================================================================
# CONFIGURATION MILITAIRE
# =====================================================================

MILITARY_WEAPONS = {
    "missile", "drone", "char", "navire", "avion", "hélicoptère",
    "canon", "fusil", "lance-roquettes", "mortier", "tank", "aircraft",
    "S-400", "S-300", "Pantsir", "Tor", "Buk", "Gepard",
    "T-90", "T-80", "T-72", "BMP-1", "BMP-2", "BMP-3",
    "Burevestnik", "Geran-2", "Orlan", "Lancet", "Krasnopol",
    "Iskander", "Kalibr", "Kh-101", "Kh-22", "Kh-55",
    "Mig-29", "Mig-31", "Su-27", "Su-35", "Su-57", "Su-34",
    "F-16", "F-35", "Patriot", "Harpoon", "Tomahawk",
}

MILITARY_UNITS = {
    "armée", "division", "brigade", "régiment", "bataillon",
    "compagnie", "peloton", "escadron", "corps d'armée",
    "groupe d'armées", "front", "army", "corps", "battalion",
}

MILITARY_ORGS = {
    "ministère", "défense", "état-major", "kremlin", "otan",
    "pentagone", "general staff", "armed forces", "military",
    "defense", "ministry", "pentagon", "nato",
}


def load_spacy_model(logger: logging.Logger, model_name: str):
    """Charge un modèle spaCy avec gestion d'erreurs."""
    try:
        logger.info(f"Chargement du modèle spaCy '{model_name}'…")
        nlp = spacy.load(model_name)
        logger.success(f"Modèle '{model_name}' chargé avec succès")
        return nlp
    except OSError as e:
        logger.error(f"Modèle '{model_name}' introuvable")
        logger.error(f"Installe-le: python -m spacy download {model_name}")
        raise SystemExit(1)


def extract_military_entities(nlp, text: str) -> list[tuple[int, int, str]]:
    """
    Extrait les entités militaires via spaCy + patterns personnalisés.
    Retourne une liste de tuples (start, end, label) sans chevauchements.
    """
    doc = nlp(text)
    spans: list[tuple[int, int, str]] = []

    # =========== 1. ORGANISATIONS (spaCy ORG → MIL_ORG) ===========
    org_count = 0
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_text_lower = ent.text.lower()
            if any(kw in org_text_lower for kw in MILITARY_ORGS):
                spans.append((ent.start_char, ent.end_char, "MIL_ORG"))
                org_count += 1

    # =========== 2. ARMES (PhraseMatcher) ===========
    weapon_count = 0
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    weapon_patterns = [nlp.make_doc(w) for w in MILITARY_WEAPONS]
    phrase_matcher.add("WEAPON", weapon_patterns)

    for match_id, start, end in phrase_matcher(doc):
        start_char = doc[start].idx
        end_char = doc[end - 1].idx + len(doc[end - 1].text)
        spans.append((start_char, end_char, "WEAPON"))
        weapon_count += 1

    # =========== 3. UNITÉS MILITAIRES (Token + neighbors) ===========
    unit_count = 0
    for token in doc:
        token_lower = token.text.lower()
        
        # Vérifier si c'est un mot-clé d'unité
        if any(unit in token_lower for unit in MILITARY_UNITS):
            start_char = token.idx
            end_char = start_char + len(token.text)
            
            # Étendre au token suivant si c'est un adjectif ou nom
            if token.i + 1 < len(doc):
                next_token = doc[token.i + 1]
                if next_token.pos_ in ("NOUN", "PROPN", "ADJ"):
                    end_char = next_token.idx + len(next_token.text)
            
            spans.append((start_char, end_char, "MIL_UNIT"))
            unit_count += 1

    # =========== 4. SUPPRESSION DES CHEVAUCHEMENTS ===========
    # Garder les spans les plus longs et non-overlappants
    spans = list(set(spans))  # Supprimer les doublons
    spans.sort(key=lambda x: x[1] - x[0], reverse=True)

    kept: list[tuple[int, int, str]] = []
    occupied: set[int] = set()

    for start, end, label in spans:
        # Vérifier s'il y a chevauchement
        if any(i in occupied for i in range(start, end)):
            continue
        kept.append((start, end, label))
        occupied.update(range(start, end))

    return sorted(kept)


def process_texts(logger: logging.Logger, nlp, records: list, sample_size: int, seed: int) -> tuple[list, list]:
    """
    Traite les textes et retourne (train_data, annotated_ids).
    """
    try:
        logger.info(f"Début du traitement des textes (sample: {sample_size}/{len(records)})")
        
        random.seed(seed)
        sample = random.sample(records, min(sample_size, len(records)))

        train_data: list = []
        annotated_ids: list = []
        failed_count = 0

        for i, record in enumerate(sample, 1):
            text = record.get("text", "")
            doc_id = record.get("id", f"doc_{i}")
            
            if not text.strip():
                logger.warning(f"[{i}/{len(sample)}] Document '{doc_id}' vide, ignoré")
                failed_count += 1
                continue

            try:
                # Extraction des entités
                entities = extract_military_entities(nlp, text)
                
                # Format spaCy : [text, {"entities": [[start, end, label], ...]}]
                train_data.append([text, {"entities": entities}])
                annotated_ids.append(doc_id)
                
                logger.debug(f"[{i}/{len(sample)}] '{doc_id}' ✓ ({len(entities)} entités)")
                
            except Exception as e:
                logger.error(f"[{i}/{len(sample)}] Erreur lors du traitement de '{doc_id}': {str(e)}")
                failed_count += 1

        logger.success(f"Traitement complété: {len(train_data)}/{len(sample)} textes annotés ({failed_count} échoués)")
        return train_data, annotated_ids
        
    except Exception as e:
        logger.error(f"Erreur fatale lors du traitement: {str(e)}")
        raise


def save_results(logger: logging.Logger, train_data: list, annotated_ids: list, output_path: Path) -> bool:
    """Sauvegarde les résultats au format JSON."""
    try:
        logger.info(f"Sauvegarde des résultats…")
        
        # Sauvegarder les annotations
        output_path.write_text(
            json.dumps(train_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.debug(f"✓ Fichier '{output_path}' sauvegardé ({len(train_data)} entrées)")
        
        # Sauvegarder les IDs
        ids_path = Path("annotated_ids.json")
        ids_path.write_text(
            json.dumps(annotated_ids, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.debug(f"✓ Fichier '{ids_path}' sauvegardé ({len(annotated_ids)} IDs)")
        
        logger.success(f"Résultats sauvegardés avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pré-annotation NER via spaCy pour données militaires"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("texts_clean.json"),
        help="Fichier JSON d'entrée (textes nettoyés)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("annotations.json"),
        help="Fichier JSON de sortie (format spaCy train data)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=300,
        help="Nombre d'articles à annoter"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed pour reproductibilité"
    )
    parser.add_argument(
        "--model", type=str, default="en_core_web_lg",
        help="Modèle spaCy (en_core_web_lg, fr_core_news_lg, etc.)"
    )
    
    args = parser.parse_args()

    # ===================== SETUP LOGGING =====================
    logger = setup_logger()
    start_time = datetime.now()
    
    logger.info("="*70)
    logger.info("PIPELINE NER - Pré-annotation via spaCy")
    logger.info("="*70)
    logger.info(f"Modèle       : {args.model}")
    logger.info(f"Entrée       : {args.input}")
    logger.info(f"Sortie       : {args.output}")
    logger.info(f"Sample size  : {args.sample_size}")
    logger.info(f"Démarrage    : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    try:
        # ===================== ÉTAPE 1: Charger le modèle =====================
        logger.info("\n[ÉTAPE 1/4] Chargement du modèle spaCy")
        nlp = load_spacy_model(logger, args.model)
        step1_ok = True

        # ===================== ÉTAPE 2: Charger les textes =====================
        logger.info("\n[ÉTAPE 2/4] Lecture du fichier d'entrée")
        try:
            if not args.input.exists():
                logger.error(f"Fichier '{args.input}' introuvable")
                raise FileNotFoundError(f"Fichier {args.input} introuvable")
            
            records = json.loads(args.input.read_text(encoding="utf-8"))
            logger.success(f"✓ {len(records)} textes chargés depuis '{args.input}'")
            step2_ok = True
        except Exception as e:
            logger.error(f"Échec de lecture: {str(e)}")
            step2_ok = False
            raise

        # ===================== ÉTAPE 3: Traiter les textes =====================
        logger.info("\n[ÉTAPE 3/4] Annotation des textes")
        try:
            train_data, annotated_ids = process_texts(logger, nlp, records, args.sample_size, args.seed)
            step3_ok = len(train_data) > 0
            if step3_ok:
                total_entities = sum(len(item[1]["entities"]) for item in train_data)
                logger.success(f"✓ {len(train_data)} textes traités, {total_entities} entités extraites")
            else:
                logger.warning("Aucun texte n'a été traité avec succès")
        except Exception as e:
            logger.error(f"Échec du traitement: {str(e)}")
            step3_ok = False
            raise

        # ===================== ÉTAPE 4: Sauvegarder =====================
        logger.info("\n[ÉTAPE 4/4] Sauvegarde des résultats")
        try:
            step4_ok = save_results(logger, train_data, annotated_ids, args.output)
        except Exception as e:
            logger.error(f"Échec de sauvegarde: {str(e)}")
            step4_ok = False
            raise

        # ===================== RÉSUMÉ FINAL =====================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("RÉSUMÉ DU PIPELINE")
        logger.info("="*70)
        logger.success(f"✓ [ÉTAPE 1] Chargement modèle     : RÉUSSI")
        logger.success(f"✓ [ÉTAPE 2] Lecture fichier      : RÉUSSI ({len(records)} records)")
        logger.success(f"✓ [ÉTAPE 3] Annotation textes    : RÉUSSI ({len(train_data)} annotés)")
        logger.success(f"✓ [ÉTAPE 4] Sauvegarde résultats : RÉUSSI")
        logger.info("="*70)
        logger.success(f"Pipeline complété en {duration:.2f}s")
        logger.success(f"Fichiers générés:")
        logger.success(f"  - {args.output}")
        logger.success(f"  - annotation_pipeline.log")
        logger.info("="*70)
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.error("\n" + "="*70)
        logger.error("ERREUR - Pipeline interrompu")
        logger.error("="*70)
        logger.error(f"Durée avant erreur: {duration:.2f}s")
        logger.error("="*70)
        sys.exit(1)


if __name__ == "__main__":
    main()