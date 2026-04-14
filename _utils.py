import math

def filter_and_sort_predictions(predictions):
    """Filters out NaN values and sorts predictions by risk score (descending)."""
    valid_preds = []
    for i, prob in enumerate(predictions):
        prob_val = float(prob)
        if not math.isnan(prob_val):
            valid_preds.append((i, prob_val))
            
    sorted_preds = sorted(valid_preds, key=lambda x: x[1], reverse=True)
    return valid_preds, sorted_preds