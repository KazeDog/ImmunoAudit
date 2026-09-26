TASK3_POSITIVE_MARGIN_THRESHOLD = 0.12

def attach_task3_dual_score_decision(result: dict) -> dict:
    parsed = result.get('parsed_response', {})
    response_score = float(parsed.get('response_support_score', 0.5))
    resistance_score = float(parsed.get('resistance_support_score', 0.5))
    dominant_side = str(parsed.get('dominant_side', 'mixed'))
    margin = response_score - resistance_score
    if margin >= TASK3_POSITIVE_MARGIN_THRESHOLD:
        prediction = 'positive'
    else:
        prediction = 'negative'
    distance_to_threshold = abs(margin - TASK3_POSITIVE_MARGIN_THRESHOLD)
    final_confidence = max(0.5, min(0.95, 0.5 + distance_to_threshold * 0.9))
    parsed['prediction'] = prediction
    parsed['confidence'] = round(final_confidence, 4)
    parsed['score_margin'] = round(margin, 4)
    result['derived_decision'] = {'prediction': prediction, 'confidence': round(final_confidence, 4), 'score_margin': round(margin, 4), 'positive_margin_threshold': TASK3_POSITIVE_MARGIN_THRESHOLD, 'response_support_score': round(response_score, 4), 'resistance_support_score': round(resistance_score, 4), 'dominant_side': dominant_side}
    return result
