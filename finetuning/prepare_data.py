import json
from pathlib import Path
import random
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
    
    # Shuffle the training pairs
    random.shuffle(training_pairs)
    
    # Save the processed data
    output_file = Path(output_dir) / 'training_pairs.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(training_pairs)} training pairs")
    print(f"Data saved to {output_file}")

if __name__ == '__main__':
    train_file = Path('../hsrc/hsrc_train.jsonl')
    output_dir = Path('./data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prepare_training_data(train_file, output_dir)