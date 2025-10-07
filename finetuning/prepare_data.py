import json
from pathlib import Path
import random
from collections import defaultdict
import pandas as pd

def prepare_training_data(train_file, output_dir):
    # Load training data
    rows = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
    
    df = pd.DataFrame(rows)
    
    # Prepare training pairs
    training_pairs = []
    
    for _, row in df.iterrows():
        query = row['query']
        paragraphs = row['paragraphs']
        target_actions = row['target_actions']
        
        # Process each paragraph and its corresponding target action
        for para_id, target_id in zip(sorted(paragraphs.keys()), sorted(target_actions.keys())):
            passage = paragraphs[para_id]['passage']
            label = int(target_actions[target_id])  # Convert label to int
            
            if 0 <= label <= 4:  # Only include valid labels
                training_pairs.append({
                    'query': query,
                    'passage': passage,
                    'label': label
                })
    
    # Group by label and undersample majority class 0
    rng = random.Random(42)  # reproducible shuffling
    label_groups = defaultdict(list)
    for pair in training_pairs:
        label_groups[pair['label']].append(pair)

    majority_label = 0
    if label_groups.get(majority_label):
        other_counts = [len(examples) for lbl, examples in label_groups.items() if lbl != majority_label and len(examples) > 0]
        if other_counts:
            target_size = min(len(label_groups[majority_label]), max(other_counts))
            rng.shuffle(label_groups[majority_label])
            label_groups[majority_label] = label_groups[majority_label][:target_size]

    balanced_pairs = []
    for examples in label_groups.values():
        balanced_pairs.extend(examples)

    rng.shuffle(balanced_pairs)
    label_distribution = {label: len(examples) for label, examples in label_groups.items()}

    # Save the processed data
    output_file = Path(output_dir) / 'training_pairs.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in balanced_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"Processed {len(balanced_pairs)} training pairs (after undersampling label 0)")
    print(f"Label distribution: {label_distribution}")
    print(f"Data saved to {output_file}")

if __name__ == '__main__':
    train_file = Path('../hsrc/hsrc_train.jsonl')
    output_dir = Path('./data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prepare_training_data(train_file, output_dir)