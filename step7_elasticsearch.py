"""
Step 7 — Data Visualization avec Elasticsearch/Kibana (Pages 30-31)
====================================================================
Charge le dataset fusionné (avec entités) et l'indexe dans Elasticsearch
pour permettre la visualisation et l'analyse dans Kibana.

Ce script :
1. Connecte à Elasticsearch (local ou Docker)
2. Crée un index avec un mapping approprié pour les entités NER
3. Indexe tous les articles avec leurs entités
4. Génère des statistiques et prépare les visualisations Kibana

Prérequis :
    - Elasticsearch en cours d'exécution (localhost:9200 par défaut)
    - pip install elasticsearch

Usage :
    python step7_elasticsearch.py --input dataset_with_entities.json
    python step7_elasticsearch.py --input dataset_with_entities.json --host localhost --port 9200
    
Pour démarrer Elasticsearch avec Docker :
    docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" \
               -e "xpack.security.enabled=false" elasticsearch:8.11.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Generator, Any

try:
    from elasticsearch import Elasticsearch, helpers
    from elasticsearch.exceptions import ConnectionError, NotFoundError
except ImportError:
    print("❌ Le module 'elasticsearch' n'est pas installé.")
    print("   Installez-le avec : pip install elasticsearch")
    sys.exit(1)


# =====================================================================
# CONFIGURATION
# =====================================================================

# Nom de l'index Elasticsearch
INDEX_NAME = "tass_ner_entities"

# Mapping pour l'index (optimisé pour les entités NER)
INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "entity_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            # Métadonnées de l'article
            "article_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "standard"},
            "text": {"type": "text", "analyzer": "standard"},
            "url": {"type": "keyword"},
            "date": {"type": "date", "format": "strict_date_optional_time||epoch_millis||yyyy-MM-dd||dd.MM.yyyy"},
            "timestamp": {"type": "date"},
            
            # Entités extraites (nested pour requêtes complexes)
            "entities": {
                "type": "nested",
                "properties": {
                    "text": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "label": {"type": "keyword"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"}
                }
            },
            
            # Compteurs par type d'entité (pour agrégations rapides)
            "weapon_count": {"type": "integer"},
            "mil_unit_count": {"type": "integer"},
            "mil_org_count": {"type": "integer"},
            "total_entities": {"type": "integer"},
            
            # Listes dénormalisées (pour facettes/filtres rapides)
            "weapons": {"type": "keyword"},
            "mil_units": {"type": "keyword"},
            "mil_orgs": {"type": "keyword"},
            
            # Champ pour la recherche full-text combinée
            "all_entity_texts": {"type": "text", "analyzer": "entity_analyzer"}
        }
    }
}


# =====================================================================
# LOGGING
# =====================================================================

class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour une meilleure lisibilité."""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[0m',
        'SUCCESS': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{log_color}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logger(log_file: Path = Path("elasticsearch_pipeline.log")) -> logging.Logger:
    """Configure un logger avec file + console output."""
    logger = logging.getLogger("Elasticsearch_Pipeline")
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
# FONCTIONS ELASTICSEARCH
# =====================================================================

def connect_elasticsearch(logger: logging.Logger, host: str, port: int, 
                          timeout: int = 30) -> Elasticsearch:
    """Établit la connexion à Elasticsearch."""
    try:
        logger.info(f"Connexion à Elasticsearch ({host}:{port})...")
        
        es = Elasticsearch(
            [{"host": host, "port": port, "scheme": "http"}],
            request_timeout=timeout
        )
        
        # Vérifier la connexion
        if not es.ping():
            raise ConnectionError("Impossible de joindre Elasticsearch")
        
        info = es.info()
        version = info.get('version', {}).get('number', 'unknown')
        cluster = info.get('cluster_name', 'unknown')
        
        logger.success(f"✓ Connecté à Elasticsearch v{version} (cluster: {cluster})")
        return es
        
    except ConnectionError as e:
        logger.error(f"❌ Impossible de se connecter à Elasticsearch: {e}")
        logger.error("   Vérifiez qu'Elasticsearch est en cours d'exécution.")
        logger.error("   Pour démarrer avec Docker:")
        logger.error('   docker run -d --name elasticsearch -p 9200:9200 \\')
        logger.error('              -e "discovery.type=single-node" \\')
        logger.error('              -e "xpack.security.enabled=false" \\')
        logger.error('              elasticsearch:8.11.0')
        raise


def create_index(logger: logging.Logger, es: Elasticsearch, 
                 index_name: str, force_recreate: bool = False) -> bool:
    """Crée l'index avec le mapping approprié."""
    try:
        exists = es.indices.exists(index=index_name)
        
        if exists:
            if force_recreate:
                logger.warning(f"Suppression de l'index existant '{index_name}'...")
                es.indices.delete(index=index_name)
            else:
                logger.info(f"L'index '{index_name}' existe déjà")
                return True
        
        logger.info(f"Création de l'index '{index_name}'...")
        es.indices.create(index=index_name, body=INDEX_MAPPING)
        logger.success(f"✓ Index '{index_name}' créé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'index: {e}")
        raise


def load_dataset(logger: logging.Logger, input_path: Path) -> list[dict]:
    """Charge le dataset fusionné."""
    try:
        logger.info(f"Chargement du dataset: {input_path}")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Fichier introuvable: {input_path}")
        
        data = json.loads(input_path.read_text(encoding="utf-8"))
        logger.success(f"✓ {len(data)} articles chargés")
        return data
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement: {e}")
        raise


def parse_entities(raw_entities: list) -> list[dict]:
    """Parse les entités depuis différents formats possibles."""
    parsed = []
    
    for ent in raw_entities:
        if isinstance(ent, dict):
            # Format dict: {"text": ..., "label": ..., "start": ..., "end": ...}
            parsed.append({
                "text": ent.get("text", ""),
                "label": ent.get("label", "UNKNOWN"),
                "start": ent.get("start", 0),
                "end": ent.get("end", 0)
            })
        elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
            # Format tuple/list: [start, end, label] ou [start, end, label, text]
            start, end, label = ent[0], ent[1], ent[2]
            text = ent[3] if len(ent) > 3 else ""
            parsed.append({
                "text": text,
                "label": label,
                "start": start,
                "end": end
            })
    
    return parsed


def prepare_document(article: dict, text_content: str = None) -> dict:
    """Prépare un document pour l'indexation dans Elasticsearch."""
    
    # Parser les entités
    raw_entities = article.get("entities", [])
    entities = parse_entities(raw_entities)
    
    # Récupérer le texte de l'article
    text = text_content or article.get("text", "")
    
    # Extraire le texte des entités depuis l'article si nécessaire
    for ent in entities:
        if not ent["text"] and text and ent["start"] >= 0 and ent["end"] > ent["start"]:
            ent["text"] = text[ent["start"]:ent["end"]]
    
    # Compter par type
    type_counts = Counter(ent["label"] for ent in entities)
    
    # Listes par type (dénormalisées pour les facettes)
    weapons = list(set(ent["text"] for ent in entities if ent["label"] == "WEAPON" and ent["text"]))
    mil_units = list(set(ent["text"] for ent in entities if ent["label"] == "MIL_UNIT" and ent["text"]))
    mil_orgs = list(set(ent["text"] for ent in entities if ent["label"] == "MIL_ORG" and ent["text"]))
    
    # Texte combiné pour recherche full-text
    all_texts = " ".join(ent["text"] for ent in entities if ent["text"])
    
    # Parser la date si présente
    date_value = article.get("date")
    if date_value:
        # Essayer différents formats
        try:
            if isinstance(date_value, str):
                # Format ISO ou similaire
                date_value = date_value.split("T")[0] if "T" in date_value else date_value
        except:
            date_value = None
    
    return {
        "article_id": article.get("id", ""),
        "title": article.get("title", ""),
        "text": text[:10000] if text else "",  # Limiter la taille du texte
        "url": article.get("url", ""),
        "date": date_value,
        "timestamp": datetime.utcnow().isoformat(),
        "entities": entities,
        "weapon_count": type_counts.get("WEAPON", 0),
        "mil_unit_count": type_counts.get("MIL_UNIT", 0),
        "mil_org_count": type_counts.get("MIL_ORG", 0),
        "total_entities": len(entities),
        "weapons": weapons,
        "mil_units": mil_units,
        "mil_orgs": mil_orgs,
        "all_entity_texts": all_texts
    }


def generate_bulk_actions(articles: list[dict], index_name: str) -> Generator[dict, None, None]:
    """Génère les actions pour l'indexation bulk."""
    for article in articles:
        doc = prepare_document(article)
        yield {
            "_index": index_name,
            "_id": doc["article_id"] or None,
            "_source": doc
        }


def index_documents(logger: logging.Logger, es: Elasticsearch, 
                    articles: list[dict], index_name: str,
                    batch_size: int = 500) -> tuple[int, int]:
    """Indexe les documents dans Elasticsearch."""
    try:
        logger.info(f"Indexation de {len(articles)} articles...")
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        # Utiliser le bulk helper pour des performances optimales
        actions = generate_bulk_actions(articles, index_name)
        
        for ok, result in helpers.streaming_bulk(
            es, 
            actions,
            chunk_size=batch_size,
            raise_on_error=False,
            raise_on_exception=False
        ):
            if ok:
                success_count += 1
            else:
                error_count += 1
                logger.debug(f"Erreur d'indexation: {result}")
            
            if (success_count + error_count) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (success_count + error_count) / elapsed if elapsed > 0 else 0
                logger.info(f"  → Progression: {success_count + error_count}/{len(articles)} ({rate:.0f} docs/s)")
        
        # Rafraîchir l'index
        es.indices.refresh(index=index_name)
        
        elapsed = time.time() - start_time
        logger.success(f"✓ Indexation terminée: {success_count} réussis, {error_count} erreurs")
        logger.success(f"✓ Temps total: {elapsed:.1f}s ({success_count/elapsed:.0f} docs/s)")
        
        return success_count, error_count
        
    except Exception as e:
        logger.error(f"Erreur lors de l'indexation: {e}")
        raise


def compute_statistics(logger: logging.Logger, es: Elasticsearch, 
                       index_name: str) -> dict:
    """Calcule des statistiques sur les données indexées."""
    try:
        logger.info("Calcul des statistiques...")
        
        stats = {}
        
        # Nombre total de documents
        count_resp = es.count(index=index_name)
        stats["total_documents"] = count_resp["count"]
        
        # Agrégations sur les entités
        agg_query = {
            "size": 0,
            "aggs": {
                "total_entities": {
                    "sum": {"field": "total_entities"}
                },
                "total_weapons": {
                    "sum": {"field": "weapon_count"}
                },
                "total_mil_units": {
                    "sum": {"field": "mil_unit_count"}
                },
                "total_mil_orgs": {
                    "sum": {"field": "mil_org_count"}
                },
                "top_weapons": {
                    "terms": {"field": "weapons", "size": 20}
                },
                "top_mil_units": {
                    "terms": {"field": "mil_units", "size": 20}
                },
                "top_mil_orgs": {
                    "terms": {"field": "mil_orgs", "size": 20}
                },
                "docs_with_entities": {
                    "filter": {"range": {"total_entities": {"gt": 0}}}
                }
            }
        }
        
        agg_resp = es.search(index=index_name, body=agg_query)
        aggs = agg_resp["aggregations"]
        
        stats["total_entities"] = int(aggs["total_entities"]["value"])
        stats["total_weapons"] = int(aggs["total_weapons"]["value"])
        stats["total_mil_units"] = int(aggs["total_mil_units"]["value"])
        stats["total_mil_orgs"] = int(aggs["total_mil_orgs"]["value"])
        stats["docs_with_entities"] = aggs["docs_with_entities"]["doc_count"]
        
        stats["top_weapons"] = [(b["key"], b["doc_count"]) for b in aggs["top_weapons"]["buckets"]]
        stats["top_mil_units"] = [(b["key"], b["doc_count"]) for b in aggs["top_mil_units"]["buckets"]]
        stats["top_mil_orgs"] = [(b["key"], b["doc_count"]) for b in aggs["top_mil_orgs"]["buckets"]]
        
        logger.success("✓ Statistiques calculées")
        return stats
        
    except Exception as e:
        logger.error(f"Erreur lors du calcul des statistiques: {e}")
        return {}


def print_statistics(logger: logging.Logger, stats: dict) -> None:
    """Affiche les statistiques de manière formatée."""
    
    logger.info("\n" + "="*70)
    logger.info("STATISTIQUES ELASTICSEARCH")
    logger.info("="*70)
    
    logger.info(f"\n📊 Vue d'ensemble:")
    logger.info(f"   • Documents indexés     : {stats.get('total_documents', 0):,}")
    logger.info(f"   • Documents avec entités: {stats.get('docs_with_entities', 0):,}")
    logger.info(f"   • Total entités         : {stats.get('total_entities', 0):,}")
    
    logger.info(f"\n🏷️  Entités par type:")
    logger.info(f"   • WEAPON    : {stats.get('total_weapons', 0):,}")
    logger.info(f"   • MIL_UNIT  : {stats.get('total_mil_units', 0):,}")
    logger.info(f"   • MIL_ORG   : {stats.get('total_mil_orgs', 0):,}")
    
    if stats.get("top_weapons"):
        logger.info(f"\n🔫 Top 10 WEAPON:")
        for name, count in stats["top_weapons"][:10]:
            logger.info(f"   {count:5d} | {name}")
    
    if stats.get("top_mil_units"):
        logger.info(f"\n🎖️  Top 10 MIL_UNIT:")
        for name, count in stats["top_mil_units"][:10]:
            logger.info(f"   {count:5d} | {name}")
    
    if stats.get("top_mil_orgs"):
        logger.info(f"\n🏛️  Top 10 MIL_ORG:")
        for name, count in stats["top_mil_orgs"][:10]:
            logger.info(f"   {count:5d} | {name}")


def generate_kibana_instructions(logger: logging.Logger, index_name: str) -> None:
    """Génère les instructions pour configurer Kibana."""
    
    instructions = f"""
================================================================================
INSTRUCTIONS KIBANA
================================================================================

1. ACCÉDER À KIBANA
   Ouvrez http://localhost:5601 dans votre navigateur

2. CRÉER UN DATA VIEW (INDEX PATTERN)
   → Menu ☰ → Stack Management → Data Views
   → Create data view
   → Name: {index_name}
   → Index pattern: {index_name}
   → Timestamp field: timestamp (ou @timestamp)
   → Save

3. VISUALISATIONS SUGGÉRÉES

   a) PIE CHART - Distribution des types d'entités
      → Visualize → Create → Pie
      → Data view: {index_name}
      → Slice by: Split slices → Terms → Field: entities.label
      
   b) BAR CHART - Top 20 Armes mentionnées
      → Visualize → Create → Vertical Bar
      → Data view: {index_name}
      → Y-axis: Count
      → X-axis: Terms → Field: weapons → Size: 20
      
   c) BAR CHART - Top 20 Unités militaires
      → Visualize → Create → Vertical Bar
      → Data view: {index_name}
      → Y-axis: Count
      → X-axis: Terms → Field: mil_units → Size: 20
      
   d) LINE CHART - Évolution temporelle des mentions
      → Visualize → Create → Line
      → Data view: {index_name}
      → Y-axis: Sum → Field: total_entities
      → X-axis: Date Histogram → Field: date
      
   e) DATA TABLE - Articles avec le plus d'entités
      → Visualize → Create → Data Table
      → Data view: {index_name}
      → Metrics: Max → Field: total_entities
      → Buckets: Split rows → Terms → Field: title.keyword → Size: 50

4. CRÉER UN DASHBOARD
   → Menu ☰ → Dashboard → Create dashboard
   → Add → Sélectionnez vos visualisations
   → Save

5. REQUÊTES UTILES (Dev Tools → Console)

   # Rechercher des articles mentionnant un système d'arme spécifique
   GET {index_name}/_search
   {{
     "query": {{
       "match": {{ "weapons": "S-400" }}
     }}
   }}
   
   # Articles avec le plus d'entités WEAPON
   GET {index_name}/_search
   {{
     "size": 10,
     "sort": [{{ "weapon_count": "desc" }}],
     "_source": ["title", "weapon_count", "weapons"]
   }}
   
   # Agrégation des co-occurrences d'entités
   GET {index_name}/_search
   {{
     "size": 0,
     "aggs": {{
       "by_weapon": {{
         "terms": {{ "field": "weapons", "size": 10 }},
         "aggs": {{
           "with_units": {{
             "terms": {{ "field": "mil_units", "size": 5 }}
           }}
         }}
       }}
     }}
   }}

================================================================================
"""
    
    print(instructions)
    
    # Sauvegarder dans un fichier
    instructions_path = Path("kibana_instructions.txt")
    instructions_path.write_text(instructions, encoding="utf-8")
    logger.success(f"✓ Instructions Kibana sauvegardées dans {instructions_path}")


# =====================================================================
# PIPELINE PRINCIPAL
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Indexation des entités NER dans Elasticsearch pour visualisation Kibana"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("dataset_with_entities.json"),
        help="Fichier JSON contenant le dataset avec entités"
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Hôte Elasticsearch"
    )
    parser.add_argument(
        "--port", type=int, default=9200,
        help="Port Elasticsearch"
    )
    parser.add_argument(
        "--index", type=str, default=INDEX_NAME,
        help="Nom de l'index Elasticsearch"
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Supprimer et recréer l'index s'il existe"
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Taille des lots pour l'indexation bulk"
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger()
    start_time = datetime.now()
    
    logger.info("="*70)
    logger.info("STEP 7 — DATA VISUALIZATION AVEC ELASTICSEARCH")
    logger.info("="*70)
    logger.info(f"Input            : {args.input}")
    logger.info(f"Elasticsearch    : {args.host}:{args.port}")
    logger.info(f"Index            : {args.index}")
    logger.info(f"Batch size       : {args.batch_size}")
    logger.info(f"Démarrage        : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    try:
        # Étape 1: Charger les données
        logger.info("\n[ÉTAPE 1/5] Chargement du dataset")
        articles = load_dataset(logger, args.input)
        
        # Étape 2: Connexion à Elasticsearch
        logger.info("\n[ÉTAPE 2/5] Connexion à Elasticsearch")
        es = connect_elasticsearch(logger, args.host, args.port)
        
        # Étape 3: Créer l'index
        logger.info("\n[ÉTAPE 3/5] Création de l'index")
        create_index(logger, es, args.index, force_recreate=args.recreate)
        
        # Étape 4: Indexer les documents
        logger.info("\n[ÉTAPE 4/5] Indexation des documents")
        success, errors = index_documents(logger, es, articles, args.index, args.batch_size)
        
        # Étape 5: Statistiques
        logger.info("\n[ÉTAPE 5/5] Calcul des statistiques")
        stats = compute_statistics(logger, es, args.index)
        print_statistics(logger, stats)
        
        # Instructions Kibana
        logger.info("\n[BONUS] Génération des instructions Kibana")
        generate_kibana_instructions(logger, args.index)
        
        # Résumé final
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*70)
        logger.success("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info("="*70)
        logger.success(f"✓ {success} documents indexés dans '{args.index}'")
        logger.success(f"✓ {stats.get('total_entities', 0)} entités disponibles pour visualisation")
        logger.success(f"✓ Temps total: {duration:.1f}s")
        logger.info("="*70)
        logger.info("\n📊 Pour visualiser les données:")
        logger.info(f"   1. Ouvrez Kibana: http://localhost:5601")
        logger.info(f"   2. Créez un Data View avec l'index '{args.index}'")
        logger.info(f"   3. Consultez kibana_instructions.txt pour les visualisations")
        logger.info("="*70 + "\n")
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ Fichier introuvable: {e}")
        logger.error("   Assurez-vous d'avoir exécuté step6_data_merged.py d'abord")
        sys.exit(1)
        
    except ConnectionError:
        logger.error("\n❌ Impossible de se connecter à Elasticsearch")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n❌ Erreur inattendue: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
