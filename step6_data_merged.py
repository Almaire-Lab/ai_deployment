"""
Step 6 Text Matching — Fusion intelligente en matchant par texte

Matche les annotations avec le dataset en comparant le texte,
avec une tolérance pour les petites différences.

🕐 AVEC LOGS DE TEMPS
"""

import json
from pathlib import Path
from collections import defaultdict
import difflib
import time


def format_duration(seconds):
    """Formate la durée en secondes de manière lisible"""
    if seconds < 1:
        return f"{seconds:.3f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def text_similarity(text1, text2, min_length=50):
    """
    Calcule la similarité entre deux textes.
    Retourne un score entre 0 et 1.
    """
    if not text1 or not text2:
        return 0
    
    # Normaliser : minuscules, espaces multiples
    text1_clean = ' '.join(text1.lower().split())
    text2_clean = ' '.join(text2.lower().split())
    
    # Si les textes sont courts, utiliser une comparaison exacte
    if len(text1_clean) < min_length or len(text2_clean) < min_length:
        if text1_clean == text2_clean:
            return 1.0
        else:
            return 0.0
    
    # Utiliser SequenceMatcher pour la similarité
    return difflib.SequenceMatcher(None, text1_clean, text2_clean).ratio()


def find_best_match(article_text, annotations_list, threshold=0.85):
    """
    Cherche la meilleure annotation qui matche ce texte.
    Retourne (annotation, score) ou (None, 0) si pas de match assez bon.
    """
    best_match = None
    best_score = 0
    
    for annotation in annotations_list:
        if isinstance(annotation, dict):
            annotation_text = annotation.get('text', '')
        elif isinstance(annotation, list) and len(annotation) >= 1:
            annotation_text = annotation[0]
        else:
            continue
        
        score = text_similarity(article_text, annotation_text)
        
        if score > best_score:
            best_score = score
            best_match = annotation
    
    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, 0


def main():
    
    start_time_total = time.time()
    
    print("\n" + "="*70)
    print("STEP 6 TEXT MATCHING — Fusion par texte")
    print("="*70)
    print(f"⏱️  Démarrage : {time.strftime('%H:%M:%S')}")
    
    ROOT = Path(__file__).resolve().parent
    dataset_path = ROOT / 'data_set.json'
    annotations_path = ROOT / 'annotations.json'
    output_path = ROOT / 'dataset_with_entities.json'
    
    # ====================================================================
    # Charger les données
    # ====================================================================
    
    print("\n1️⃣ Chargement...")
    start_time = time.time()
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")
    
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations introuvables : {annotations_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations_raw = json.load(f)
    
    elapsed = time.time() - start_time
    print(f"  ✓ Dataset : {len(dataset)} articles")
    print(f"  ✓ Annotations : {len(annotations_raw)} entrées")
    print(f"  ⏱️  Temps : {format_duration(elapsed)}")
    
    # ====================================================================
    # Fusionner : matching par texte
    # ====================================================================
    
    print("\n2️⃣ Fusion par matching de texte...")
    print("  (Cela peut prendre quelques secondes...)")
    
    start_time = time.time()
    
    merged = []
    matched = 0
    unmatched = 0
    match_scores = []
    
    for i, article in enumerate(dataset):
        article_text = article.get('text', '')
        
        # Copier tous les champs du dataset
        merged_article = article.copy()
        
        # Chercher la meilleure annotation qui matche ce texte
        annotation, score = find_best_match(article_text, annotations_raw, threshold=0.85)
        
        if annotation:
            # Match trouvé
            if isinstance(annotation, dict):
                merged_article['entities'] = annotation.get('entities', [])
            elif isinstance(annotation, list) and len(annotation) >= 2:
                if isinstance(annotation[1], dict):
                    merged_article['entities'] = annotation[1].get('entities', [])
                else:
                    merged_article['entities'] = []
            else:
                merged_article['entities'] = []
            
            matched += 1
            match_scores.append(score)
        else:
            # Pas de match
            merged_article['entities'] = []
            unmatched += 1
        
        merged.append(merged_article)
        
        # Afficher la progression
        if (i + 1) % 5000 == 0:
            elapsed_step = time.time() - start_time
            rate = (i + 1) / elapsed_step
            remaining = (len(dataset) - (i + 1)) / rate if rate > 0 else 0
            print(f"    → Traité {i + 1:5}/{len(dataset)} ({elapsed_step/60:5.1f}m écoulé, {format_duration(remaining)} restant, {rate:.0f} art/s)")
    
    elapsed = time.time() - start_time
    print(f"  ✓ Fusion terminée")
    print(f"    → Articles matchés : {matched}")
    print(f"    → Articles sans match : {unmatched}")
    
    if match_scores:
        avg_score = sum(match_scores) / len(match_scores)
        min_score = min(match_scores)
        max_score = max(match_scores)
        print(f"    → Score de similarité moyen : {avg_score:.3f}")
        print(f"    → Score min/max : {min_score:.3f} / {max_score:.3f}")
    
    print(f"  ⏱️  Temps de fusion : {format_duration(elapsed)}")
    
    # ====================================================================
    # Calculer les statistiques
    # ====================================================================
    
    print("\n3️⃣ Statistiques...")
    start_time = time.time()
    
    total_entities = sum(len(article.get('entities', [])) for article in merged)
    articles_with_entities = sum(1 for article in merged if article.get('entities'))
    
    print(f"  ✓ Total d'entités : {total_entities}")
    print(f"  ✓ Articles avec entités : {articles_with_entities}")
    if len(merged) > 0:
        print(f"  ✓ Moyenne par article : {total_entities / len(merged):.2f}")
    
    # Compter par type
    entity_types = defaultdict(int)
    for article in merged:
        for entity in article.get('entities', []):
            if len(entity) >= 3:
                entity_type = entity[2]
                entity_types[entity_type] += 1
    
    print(f"  ✓ Entités par type :")
    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
        print(f"      - {entity_type:12} : {count:6} entités")
    
    elapsed = time.time() - start_time
    print(f"  ⏱️  Temps de calcul : {format_duration(elapsed)}")
    
    # ====================================================================
    # Sauvegarder
    # ====================================================================
    
    print("\n4️⃣ Sauvegarde...")
    start_time = time.time()
    
    output_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    
    elapsed = time.time() - start_time
    print(f"  ✓ Fichier créé : {output_path.name}")
    print(f"  ⏱️  Temps d'écriture : {format_duration(elapsed)}")
    
    # ====================================================================
    # Exemple
    # ====================================================================
    
    print("\n5️⃣ Exemples (articles avec entités)...")
    
    examples = [a for a in merged if a.get('entities')][:3]
    
    for idx, example in enumerate(examples, 1):
        print(f"\n  [{idx}] ID : {example.get('id')}")
        print(f"      Titre : {example.get('title', 'N/A')[:60]}")
        print(f"      Entités : {len(example.get('entities', []))} détectées")
        
        if example.get('entities'):
            for i, entity in enumerate(example['entities'][:2], 1):
                if len(entity) >= 3:
                    start, end, label = entity[0], entity[1], entity[2]
                    text_slice = example.get('text', '')[start:end]
                    print(f"        {i}. [{start:4}, {end:4}] {label:12} : \"{text_slice}\"")
    
    # ====================================================================
    # Résumé final
    # ====================================================================
    
    total_elapsed = time.time() - start_time_total
    
    print("\n" + "="*70)
    print("✅ FUSION TERMINÉE AVEC SUCCÈS")
    print("="*70)
    print(f"📊 Résultat final : {output_path.name}")
    print(f"   • Articles totaux : {len(merged)}")
    print(f"   • Articles matchés : {matched}")
    print(f"   • Entités totales : {total_entities}")
    print(f"   • Articles avec entités : {articles_with_entities}")
    print(f"\n⏱️  TEMPS TOTAL : {format_duration(total_elapsed)}")
    print(f"   Fin : {time.strftime('%H:%M:%S')}")
    print(f"\n💾 Fichier prêt pour l'exploitation !")


if __name__ == "__main__":
    main()