import json
from baseline_submission.model import preprocess, predict

def load_and_predict(jsonl_path):
    # Build corpus_dict: {paragraph_uuid: {"passage": passage, ...}}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # If the file is a list of queries with paragraphs, build corpus_dict from all paragraphs
    corpus_dict = {}
    for sample in data:
        paragraphs = sample.get('paragraphs', {})
        for para in paragraphs.values():
            para_id = para.get('uuid')
            if para_id:
                corpus_dict[para_id] = {'passage': para.get('passage', '')}

    preprocessed = preprocess(corpus_dict)

    results = []
    for sample in data:
        query_text = sample.get('query', '')
        query_id = sample.get('query_uuid', sample.get('id'))
        query_dict = {'query': query_text}
        prediction = predict(query_dict, preprocessed)
        results.append({'id': query_id, 'prediction': prediction})
    return results

if __name__ == "__main__":
    predictions = load_and_predict("hsrc/hsrc_train.jsonl")
    for item in predictions:
        print(item)
    print(1)