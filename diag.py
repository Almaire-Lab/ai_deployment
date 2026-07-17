

# Créer et exécuter le diagnostic

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
possible_files = [ROOT / 'text_nered_unified.json', ROOT / 'text_nered.json', ROOT / 'annotations.json']

annotations_path = None
for path in possible_files:
    if path.exists():
        annotations_path = path
        break

if not annotations_path:
    print("❌ Aucun fichier d'annotations trouvé !")
    exit()

print(f"\n📋 Fichier : {annotations_path.name}")

with open(annotations_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Longueur : {len(data)}")

total_entities = 0
for item in data:
    if isinstance(item, list) and len(item) >= 2:
        if isinstance(item[1], dict) and 'entities' in item[1]:
            total_entities += len(item[1]['entities'])
    elif isinstance(item, dict):
        if 'entities' in item:
            total_entities += len(item['entities'])

print(f"Total d'entités : {total_entities}")

if total_entities > 0:
    print("✅ ENTITÉS PRÉSENTES")
else:
    print("❌ AUCUNE ENTITÉ")
