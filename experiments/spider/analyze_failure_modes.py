import json

# Get one more critical insight - why is constrained stuck?
with open('predictions/constrained_predictions.json') as f:
    const_preds = json.load(f)

# Check the max_new_tokens limit issue
print('=== CONSTRAINED REPETITION ANALYSIS ===')
print('Average length of constrained predictions:')
avg_len = sum(len(p) for p in const_preds) / len(const_preds)
print(f'Average: {avg_len:.1f} characters')
print(f'Min: {min(len(p) for p in const_preds)}')
print(f'Max: {max(len(p) for p in const_preds)}')

# Check hybrid empty query issue
with open('results/hybrid_8b_results.json') as f:
    hybrid_results = json.load(f)

empty_queries = [r for r in hybrid_results if r['predicted_sql'] == '']
print(f'\n=== HYBRID EMPTY QUERIES ===')
print(f'Total empty queries: {len(empty_queries)}')
if empty_queries:
    print('Sample questions that produce empty queries:')
    for r in empty_queries[:5]:
        print(f'  - {r["question"]}')

# Check text extraction errors in unconstrained
with open('results/unconstrained_8b_results.json') as f:
    uncon_results = json.load(f)

extraction_errors = [r for r in uncon_results if '```' in r['error_message']]
print(f'\n=== UNCONSTRAINED TEXT EXTRACTION ERRORS ===')
print(f'Total markdown extraction errors: {len(extraction_errors)}')

incomplete_errors = [r for r in uncon_results if not r['is_valid'] and 'Empty' not in r['error_message']]
print(f'Total syntax/incomplete errors: {len(incomplete_errors)}')
