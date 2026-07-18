# 🎯 AI Deployment - OSINT NER Pipeline

Pipeline complet d'extraction d'entités nommées (NER) à partir d'articles de presse TASS sur le conflit en Ukraine.

## 📋 Description du projet

Ce projet implémente un système OSINT (Open Source Intelligence) qui :
1. **Collecte** des articles de presse depuis TASS (agence de presse russe)
2. **Extrait** automatiquement les entités militaires mentionnées
3. **Visualise** les données dans Kibana pour l'analyse

### Entités extraites (NER Labels)

| Label | Description | Exemples |
|-------|-------------|----------|
| `WEAPON` | Systèmes d'armes, munitions, équipements | S-400, HIMARS, Kalibr, T-90 |
| `MIL_UNIT` | Unités militaires | 58th Army, 1st Tank Brigade |
| `MIL_ORG` | Organisations militaires | Russian Ministry of Defense, NATO |

## 🏗️ Architecture du Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Step 1    │───▶│   Step 2    │───▶│   Step 3    │───▶│   Step 4    │
│  Extract &  │    │     LLM     │    │  Convert &  │    │    Train    │
│    Clean    │    │  Annotate   │    │    Split    │    │    Model    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│   Step 7    │◀───│   Step 6    │◀───│   Step 5    │◀──────────┘
│ Elasticsearch│    │    Merge    │    │  Inference  │
│   Kibana    │    │    Data     │    │     NER     │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 📁 Structure des fichiers

| Fichier | Description |
|---------|-------------|
| `step1_extract_clean.py` | Extraction et nettoyage du corpus TASS |
| `step2_llm_annotate.py` | Annotation automatique via LLM |
| `step3_convert_split.py` | Conversion au format SpaCy et split train/dev |
| `step4_train_model.py` | Entraînement du modèle NER SpaCy |
| `step5_inference.py` | Inférence sur le corpus complet |
| `step6_data_merged.py` | Fusion des annotations avec le dataset |
| `step7_elasticsearch.py` | Indexation et visualisation Elasticsearch/Kibana |
| `docker-compose.yml` | Configuration Docker pour ES + Kibana |
| `diag.py` | Script de diagnostic des annotations |

## 🚀 Installation et utilisation

### Prérequis

- Python 3.9+
- Docker et Docker Compose
- ~4 GB de RAM pour Elasticsearch

### 1. Cloner le repository

```bash
git clone https://github.com/Almaire-Lab/ai_deployment.git
cd ai_deployment
```

### 2. Installer les dépendances Python

```bash
pip install spacy elasticsearch
python -m spacy download en_core_web_sm
```

### 3. Exécuter le pipeline complet

```bash
# Étape 1: Extraction et nettoyage
python step1_extract_clean.py --input data_set.json --output texts_clean.json

# Étape 2: Annotation LLM (nécessite API key)
python step2_llm_annotate.py --input texts_clean.json --output annotations.json

# Étape 3: Conversion pour SpaCy
python step3_convert_split.py --input annotations.json

# Étape 4: Entraînement du modèle
python step4_train_model.py --train train.spacy --dev dev.spacy

# Étape 5: Inférence
python step5_inference.py --model ./output/model-best --corpus texts_clean.json

# Étape 6: Fusion des données
python step6_data_merged.py

# Étape 7: Visualisation (voir section dédiée)
```

## 📊 Visualisation avec Elasticsearch/Kibana

### Démarrer les services

```bash
# Lancer Elasticsearch et Kibana
docker-compose up -d

# Attendre ~30 secondes que les services démarrent
# Vérifier qu'Elasticsearch est prêt
curl http://localhost:9200
```

### Indexer les données

```bash
pip install elasticsearch
python step7_elasticsearch.py --input dataset_with_entities.json
```

### Accéder à Kibana

🔗 **http://localhost:5601**

### Créer les visualisations

1. **Menu ☰ → Stack Management → Data Views**
2. Créer un Data View : `tass_ner_entities`
3. **Menu ☰ → Visualize Library** pour créer des graphiques :
   - **Pie Chart** : Distribution des types d'entités
   - **Bar Chart** : Top 20 armes / unités militaires
   - **Timeline** : Évolution des mentions dans le temps

### Arrêter les services

```bash
docker-compose down
```

## 📈 Exemples de résultats

Le pipeline extrait automatiquement des entités comme :

- 🔫 **WEAPON** : S-400, HIMARS, Kalibr cruise missiles, T-90 tanks
- 🎖️ **MIL_UNIT** : 58th Combined Arms Army, 1st Guards Tank Army
- 🏛️ **MIL_ORG** : Russian Ministry of Defense, Ukrainian Armed Forces

## 🛠️ Technologies utilisées

- **SpaCy** - Framework NLP pour le NER
- **Elasticsearch** - Moteur de recherche et d'indexation
- **Kibana** - Visualisation et tableaux de bord
- **Docker** - Conteneurisation des services
- **Python** - Langage principal du pipeline

## 📝 Licence

Ce projet est à usage éducatif dans le cadre d'un TP sur le déploiement d'IA.

---

*Projet réalisé dans le cadre du cours AI Deployment*