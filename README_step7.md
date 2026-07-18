# Step 7 — Data Visualization avec Elasticsearch/Kibana

Cette étape finale du pipeline OSINT NER permet de visualiser et analyser les entités extraites (WEAPON, MIL_UNIT, MIL_ORG) dans Kibana.

## Prérequis

1. **Python** avec le package Elasticsearch :
   ```bash
   pip install elasticsearch
   ```

2. **Docker** (recommandé) pour Elasticsearch et Kibana

## Démarrage rapide

### 1. Lancer Elasticsearch et Kibana

```bash
# Avec Docker Compose (recommandé)
docker-compose up -d

# Ou avec Docker directement
docker run -d --name elasticsearch -p 9200:9200 \
           -e "discovery.type=single-node" \
           -e "xpack.security.enabled=false" \
           elasticsearch:8.11.0

docker run -d --name kibana -p 5601:5601 \
           -e "ELASTICSEARCH_HOSTS=http://host.docker.internal:9200" \
           kibana:8.11.0
```

### 2. Vérifier qu'Elasticsearch est prêt

```bash
curl http://localhost:9200
```

### 3. Exécuter le script d'indexation

```bash
# Indexer les données dans Elasticsearch
python step7_elasticsearch.py --input dataset_with_entities.json

# Options disponibles
python step7_elasticsearch.py --help
```

### 4. Accéder à Kibana

Ouvrez http://localhost:5601 dans votre navigateur.

## Créer les visualisations dans Kibana

### 1. Créer un Data View

1. Menu ☰ → **Stack Management** → **Data Views**
2. Cliquer sur **Create data view**
3. Paramètres :
   - **Name** : `tass_ner_entities`
   - **Index pattern** : `tass_ner_entities`
   - **Timestamp field** : `timestamp`
4. Cliquer sur **Save data view to Kibana**

### 2. Visualisations suggérées

#### a) Pie Chart — Distribution des types d'entités

1. Menu ☰ → **Visualize Library** → **Create visualization**
2. Type : **Pie**
3. Data view : `tass_ner_entities`
4. Configuration :
   - **Slice by** : Split slices → Terms → `entities.label`

#### b) Bar Chart — Top 20 Armes

1. Type : **Vertical Bar**
2. Configuration :
   - **Y-axis** : Count
   - **X-axis** : Terms → `weapons` → Size: 20

#### c) Bar Chart — Top 20 Unités militaires

1. Type : **Vertical Bar**
2. Configuration :
   - **Y-axis** : Count
   - **X-axis** : Terms → `mil_units` → Size: 20

#### d) Timeline — Évolution des mentions

1. Type : **Line**
2. Configuration :
   - **Y-axis** : Sum → `total_entities`
   - **X-axis** : Date Histogram → `date`

### 3. Créer un Dashboard

1. Menu ☰ → **Dashboard** → **Create dashboard**
2. Cliquer sur **Add from library**
3. Sélectionner vos visualisations
4. Arranger et sauvegarder

## Requêtes utiles (Dev Tools)

Dans Kibana, ouvrez **Dev Tools** (Menu ☰ → Dev Tools) :

```json
# Rechercher des articles mentionnant un système d'arme
GET tass_ner_entities/_search
{
  "query": {
    "match": { "weapons": "S-400" }
  }
}

# Articles avec le plus d'entités WEAPON
GET tass_ner_entities/_search
{
  "size": 10,
  "sort": [{ "weapon_count": "desc" }],
  "_source": ["title", "weapon_count", "weapons"]
}

# Distribution des entités par type
GET tass_ner_entities/_search
{
  "size": 0,
  "aggs": {
    "by_type": {
      "nested": { "path": "entities" },
      "aggs": {
        "labels": {
          "terms": { "field": "entities.label" }
        }
      }
    }
  }
}

# Co-occurrences armes/unités
GET tass_ner_entities/_search
{
  "size": 0,
  "aggs": {
    "by_weapon": {
      "terms": { "field": "weapons", "size": 10 },
      "aggs": {
        "with_units": {
          "terms": { "field": "mil_units", "size": 5 }
        }
      }
    }
  }
}
```

## Structure des données indexées

Chaque document dans l'index contient :

| Champ | Type | Description |
|-------|------|-------------|
| `article_id` | keyword | ID unique de l'article |
| `title` | text | Titre de l'article |
| `text` | text | Contenu textuel (tronqué) |
| `url` | keyword | URL source |
| `date` | date | Date de publication |
| `entities` | nested | Liste des entités NER |
| `weapon_count` | integer | Nombre d'entités WEAPON |
| `mil_unit_count` | integer | Nombre d'entités MIL_UNIT |
| `mil_org_count` | integer | Nombre d'entités MIL_ORG |
| `weapons` | keyword[] | Liste des armes (dénormalisée) |
| `mil_units` | keyword[] | Liste des unités (dénormalisée) |
| `mil_orgs` | keyword[] | Liste des organisations (dénormalisée) |

## Arrêt des services

```bash
# Avec Docker Compose
docker-compose down

# Supprimer aussi les données
docker-compose down -v
```

## Troubleshooting

### Elasticsearch ne démarre pas

```bash
# Vérifier les logs
docker logs elasticsearch_ner

# Augmenter la mémoire virtuelle (Linux)
sudo sysctl -w vm.max_map_count=262144
```

### Erreur de connexion

```bash
# Vérifier que le port est accessible
curl -v http://localhost:9200
```

### Kibana ne se connecte pas

Attendez qu'Elasticsearch soit complètement démarré (healthcheck vert) avant de démarrer Kibana.
