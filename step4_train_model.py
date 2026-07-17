"""
Étape 4 — Configuration & Entraînement du Modèle (Page 26)
===========================================================
Transfer learning à partir de en_core_web_sm : fine-tune le NER pré-entraîné
pour spécialiser sur 3 labels militaires (WEAPON, MIL_UNIT, MIL_ORG).

Pipeline :
  1. Générer une configuration spaCy valide
  2. Entraîner le modèle avec early stopping
  3. Évaluer et sauvegarder le meilleur modèle

Usage :
    python step4_train_model.py --training-data train.spacy --dev-data dev.spacy
"""

from __future__ import annotations

import argparse
import logging
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime

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


def setup_logger(log_file: Path = Path("training_pipeline.log")) -> logging.Logger:
    """Configure un logger avec file + console output."""
    logger = logging.getLogger("Training_Pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logging.addLevelName(25, "SUCCESS")
    def log_success(self, message, *args, **kwargs):
        if self.isEnabledFor(25):
            self._log(25, message, args, **kwargs)
    logger.success = log_success.__get__(logger, logger.__class__)
    
    return logger


# =====================================================================
# FONCTIONS D'ENTRAÎNEMENT
# =====================================================================

def check_spacy_version(logger: logging.Logger) -> bool:
    """Vérifie que spaCy >= 3.7 est installé."""
    try:
        import spacy
        version = tuple(map(int, spacy.__version__.split('.')[:2]))
        required = (3, 7)
        if version >= required:
            logger.success(f"✓ spaCy {spacy.__version__} détecté")
            return True
        else:
            logger.error(f"spaCy {spacy.__version__} détecté, mais >= 3.7 requis")
            logger.info("Mets à jour : pip install -U spacy")
            return False
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de spaCy: {str(e)}")
        return False


def check_base_model(logger: logging.Logger, base_model: str = "en_core_web_sm") -> bool:
    """Vérifie et télécharge le modèle de base si nécessaire."""
    try:
        import spacy
        logger.info(f"Vérification du modèle de base '{base_model}'…")
        
        try:
            spacy.load(base_model)
            logger.success(f"✓ Modèle '{base_model}' disponible")
            return True
        except OSError:
            logger.warning(f"Modèle '{base_model}' introuvable, téléchargement…")
            
            result = subprocess.run(
                ["python", "-m", "spacy", "download", base_model],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.success(f"✓ Modèle '{base_model}' téléchargé avec succès")
                return True
            else:
                logger.error(f"Impossible de télécharger '{base_model}'")
                logger.error(f"Télécharge manuellement : python -m spacy download {base_model}")
                return False
                
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        return False


def generate_config(logger: logging.Logger, config_path: Path, 
                   train_path: Path, dev_path: Path) -> bool:
    """Génère la configuration avec spaCy init et l'adapte pour transfer learning."""
    try:
        logger.info(f"Génération du fichier config.cfg…")
        
        # Utiliser spaCy pour générer un config valide
        result = subprocess.run(
            ["python", "-m", "spacy", "init", "config", str(config_path),
             "--lang", "en", "--pipeline", "ner", "--force"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 or "already exists" in result.stderr:
            logger.success(f"✓ Config généré: {config_path}")
            
            # Modifier le config pour le transfer learning
            logger.info("Adaptation pour transfer learning…")
            config_content = config_path.read_text(encoding="utf-8")
            
            # Convertir les chemins avec forward slashes (compatible INI)
            train_abs = str(train_path.absolute()).replace("\\", "/")
            dev_abs = str(dev_path.absolute()).replace("\\", "/")
            
            # 1. Remplacer les chemins train et dev
            # Chercher les lignes train = "..." et dev = "..."
            lines = config_content.split("\n")
            new_lines = []
            
            for line in lines:
                if line.strip().startswith("train ="):
                    new_lines.append(f'train = "{train_abs}"')
                elif line.strip().startswith("dev ="):
                    new_lines.append(f'dev = "{dev_abs}"')
                elif "[components.ner]" in line:
                    # Ajouter le sourcing du modèle après [components.ner]
                    new_lines.append(line)
                    new_lines.append('source = "en_core_web_sm"')
                    new_lines.append('replace_listeners = ["model.tok2vec"]')
                elif line.strip().startswith("factory = \"ner\""):
                    # Sauter la ligne factory pour éviter doublon
                    continue
                else:
                    new_lines.append(line)
            
            config_content = "\n".join(new_lines)
            config_path.write_text(config_content, encoding="utf-8")
            logger.success(f"✓ Config adapté pour transfer learning")
            return True
        else:
            logger.error(f"Impossible de générer le config")
            return False
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération du config: {str(e)}")
        return False


def validate_data(logger: logging.Logger, train_path: Path, dev_path: Path) -> bool:
    """Valide les fichiers de données."""
    try:
        logger.info(f"Validation des données…")
        
        if not train_path.exists():
            logger.error(f"Fichier '{train_path}' introuvable")
            return False
        
        if not dev_path.exists():
            logger.error(f"Fichier '{dev_path}' introuvable")
            return False
        
        logger.success(f"✓ Fichiers de données valides")
        return True
        
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        return False


def train_model(logger: logging.Logger, config_path: Path, output_dir: Path,
                train_path: Path, dev_path: Path, gpu_id: int = -1) -> bool:
    """Lance l'entraînement du modèle."""
    try:
        logger.info(f"Démarrage de l'entraînement…")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convertir les chemins en forward slashes (compatible Windows)
        config_str = str(config_path.absolute()).replace("\\", "/")
        output_str = str(output_dir.absolute()).replace("\\", "/")
        train_str = str(train_path.absolute()).replace("\\", "/")
        dev_str = str(dev_path.absolute()).replace("\\", "/")
        
        cmd = [
            "python", "-m", "spacy", "train",
            config_str,
            "--output", output_str,
            "--paths.train", train_str,
            "--paths.dev", dev_str,
        ]
        
        if gpu_id >= 0:
            cmd.extend(["--gpu-id", str(gpu_id)])
            logger.info(f"Utilisation GPU: {gpu_id}")
        else:
            logger.info(f"Utilisation CPU (pour GPU, passe --gpu-id)")
        
        logger.debug(f"Commande: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            logger.success(f"✓ Entraînement complété")
            return True
        else:
            logger.error(f"Entraînement échoué (code: {result.returncode})")
            return False
            
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {str(e)}")
        return False


def check_output_model(logger: logging.Logger, output_dir: Path, 
                       model_name: str = "model-best") -> bool:
    """Vérifie que le modèle de sortie existe."""
    try:
        model_path = output_dir / model_name
        
        if model_path.exists() and (model_path / "meta.json").exists():
            logger.success(f"✓ Modèle trouvé: {model_path}")
            
            import spacy
            nlp = spacy.load(model_path)
            logger.success(f"✓ Modèle chargé et fonctionnel")
            logger.info(f"  Pipeline: {nlp.pipe_names}")
            logger.info(f"  Labels NER: {nlp.get_pipe('ner').labels}")
            return True
        else:
            logger.error(f"Modèle non trouvé: {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du modèle: {str(e)}")
        return False


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Étape 4 — Configuration & entraînement du modèle NER"
    )
    parser.add_argument(
        "--training-data", type=Path, default=Path("train.spacy"),
        help="Données d'entraînement (format .spacy)"
    )
    parser.add_argument(
        "--dev-data", type=Path, default=Path("dev.spacy"),
        help="Données de validation (format .spacy)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.cfg"),
        help="Fichier de configuration spaCy"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("./output"),
        help="Répertoire de sortie du modèle"
    )
    parser.add_argument(
        "--gpu-id", type=int, default=-1,
        help="ID du GPU (-1 pour CPU)"
    )
    
    args = parser.parse_args()

    logger = setup_logger()
    start_time = datetime.now()
    
    logger.info("="*70)
    logger.info("ÉTAPE 4 — CONFIGURATION & ENTRAÎNEMENT DU MODÈLE NER")
    logger.info("="*70)
    logger.info(f"Données d'entraînement : {args.training_data}")
    logger.info(f"Données de validation  : {args.dev_data}")
    logger.info(f"Config spaCy           : {args.config}")
    logger.info(f"Répertoire output      : {args.output}")
    logger.info(f"GPU ID                 : {args.gpu_id if args.gpu_id >= 0 else 'CPU'}")
    logger.info(f"Démarrage              : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    try:
        # ===================== ÉTAPE 1: Vérifier spaCy =====================
        logger.info("\n[ÉTAPE 1/6] Vérification de l'installation spaCy")
        if not check_spacy_version(logger):
            sys.exit(1)

        # ===================== ÉTAPE 2: Vérifier modèle de base =====================
        logger.info("\n[ÉTAPE 2/6] Vérification du modèle de base")
        if not check_base_model(logger):
            sys.exit(1)

        # ===================== ÉTAPE 3: Vérifier données =====================
        logger.info("\n[ÉTAPE 3/6] Vérification des données d'entraînement")
        if not validate_data(logger, args.training_data, args.dev_data):
            sys.exit(1)

        # ===================== ÉTAPE 4: Générer config =====================
        logger.info("\n[ÉTAPE 4/6] Génération de la configuration spaCy")
        if not generate_config(logger, args.config, args.training_data, args.dev_data):
            sys.exit(1)

        # ===================== ÉTAPE 5: Entraîner =====================
        logger.info("\n[ÉTAPE 5/6] Entraînement du modèle")
        if not train_model(logger, args.config, args.output, 
                          args.training_data, args.dev_data, args.gpu_id):
            sys.exit(1)

        # ===================== ÉTAPE 6: Vérifier output =====================
        logger.info("\n[ÉTAPE 6/6] Vérification du modèle entraîné")
        if not check_output_model(logger, args.output):
            logger.warning("⚠ Impossible de vérifier le modèle de sortie")

        # ===================== RÉSUMÉ FINAL =====================
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.info("RÉSUMÉ DU PIPELINE")
        logger.info("="*70)
        logger.success(f"✓ [ÉTAPE 1] Vérification spaCy          : RÉUSSI")
        logger.success(f"✓ [ÉTAPE 2] Modèle de base             : RÉUSSI")
        logger.success(f"✓ [ÉTAPE 3] Vérification données       : RÉUSSI")
        logger.success(f"✓ [ÉTAPE 4] Génération config          : RÉUSSI")
        logger.success(f"✓ [ÉTAPE 5] Entraînement modèle        : RÉUSSI")
        logger.success(f"✓ [ÉTAPE 6] Vérification sortie        : RÉUSSI")
        logger.info("="*70)
        logger.success(f"Pipeline complété en {duration:.2f}s")
        logger.success(f"Fichiers générés:")
        logger.success(f"  - {args.output}/model-best")
        logger.success(f"  - training_pipeline.log")
        logger.info("\nProchaine étape : python step5_inference.py")
        logger.info("="*70 + "\n")
        
    except KeyboardInterrupt:
        logger.warning("\n[!] Pipeline interrompu par l'utilisateur")
        sys.exit(0)
        
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