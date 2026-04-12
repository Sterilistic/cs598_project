import os
import json
import csv
import re
import pandas as pd
import numpy as np
from operator import itemgetter
from collections import OrderedDict, defaultdict


def normalize_entity_text(text):
    """Normalize text so relation entities match alias keys more reliably."""
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    # Collapse repeated whitespace to a single space.
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation around the token while keeping medical separators inside terms.
    return text.strip(".,;:!?\"'()[]{}")


def build_entity_lookup(data):
    """Build lookup table from aliases and names keyed by normalized text."""
    save_entity_dict = {}
    for _, entity in data.items():
        aliases = entity.get('Aliases', [])
        for alias in aliases:
            alias_key = normalize_entity_text(alias)
            if alias_key:
                save_entity_dict[alias_key] = {
                    'entity_type': entity.get('entity_type', ''),
                    'count': entity.get('count', 0),
                    'name': entity.get('Name', ''),
                    'cui': entity.get('CUI', '')
                }

        name_key = normalize_entity_text(entity.get('Name', ''))
        if name_key and name_key not in save_entity_dict:
            save_entity_dict[name_key] = {
                'entity_type': entity.get('entity_type', ''),
                'count': entity.get('count', 0),
                'name': entity.get('Name', ''),
                'cui': entity.get('CUI', '')
            }
    return save_entity_dict


def candidate_entity_keys(text):
    """Return normalized candidate keys for robust entity lookup."""
    key = normalize_entity_text(text)
    if not key:
        return []

    candidates = [key]
    # Lightweight singular/plural fallback for common variations.
    if key.endswith('ies') and len(key) > 3:
        candidates.append(key[:-3] + 'y')
    if key.endswith('es') and len(key) > 2:
        candidates.append(key[:-2])
    if key.endswith('s') and len(key) > 1:
        candidates.append(key[:-1])
    return list(OrderedDict.fromkeys(candidates))

def has_measurement_units(text):
    # Use regex to match numbers followed by units, e.g., "8mm", "9cm", including ranges like "5-10mm"
    pattern = r'\d+(\.\d+)?\s*-?\s*(mm|cm|m|km|in|ft|yd|mi)'
    # Search for matching patterns in the text
    return bool(re.search(pattern, text))

def extract_size_relations(input_csv_file, input_json_file, save_csv_file):
    # Load entity data
    with open(input_json_file, 'r') as file:
        data = json.load(file)
    
    # Create a dictionary of entities with normalized aliases/names as keys.
    save_entity_dict = build_entity_lookup(data)
    
    # Load relation data
    relation_df = pd.read_csv(input_csv_file)
    
    # Process relations
    save_row = []
    miss_counter = defaultdict(int)
    for index, row in relation_df.iterrows():
        source_entity = normalize_entity_text(row['source_entity'])
        target_entity_raw = row['target_entity']

        if has_measurement_units(source_entity):
            resolved_target = None
            for key in candidate_entity_keys(target_entity_raw):
                if key in save_entity_dict:
                    resolved_target = save_entity_dict[key]
                    break

            if resolved_target is not None:
                save_row.append([
                    source_entity,
                    normalize_entity_text(target_entity_raw),
                    resolved_target['cui'],
                    resolved_target['entity_type'],
                    row['count']
                ])
            else:
                miss_counter[normalize_entity_text(target_entity_raw)] += 1

    if miss_counter:
        print(f"Entity lookup misses: {sum(miss_counter.values())}")
        for entity, cnt in sorted(miss_counter.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"Entity not found: {entity} (count={cnt})")
    
    # Save processed relations to CSV
    with open(save_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['source_entity', 'target_entity', 'target_cui', 'target_entity_type', 'count'])
        writer.writerows(save_row)
    
    # Group by target_cui and aggregate
    df = pd.read_csv(save_csv_file)
    result = df.groupby('target_cui').agg({
        'source_entity': 'first',   # Keep the first occurrence of source_entity
        'target_entity': 'first',   # Keep the first occurrence of target_entity
        'target_entity_type': 'first',  # Keep the first occurrence of target_entity_type
        'count': 'sum'  # Sum the counts
    }).reset_index()
    
    # Save the final result
    result.to_csv(save_csv_file, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract Size Relations')
    parser.add_argument('--entity_dir', type=str, default='./entities')
    parser.add_argument('--real_dir', type=str, default='./relation')
    args = parser.parse_args()
    
    input_csv_file = os.path.join(args.real_dir, 'all_relations.csv')
    save_csv_file = os.path.join(args.real_dir, 'size_relations.csv')
    isolated_json_file = os.path.join(args.entity_dir, 'isolated_entities_merge.json')
    
    extract_size_relations(input_csv_file, isolated_json_file, save_csv_file)

