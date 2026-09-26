TASK4_POSITIVE_MARGIN_THRESHOLD = 0.0

def attach_task4_llm_decision(result: dict, margin_threshold: float=TASK4_POSITIVE_MARGIN_THRESHOLD) -> dict:
    parsed = result.get('parsed_response', {})
    toxicity_score = float(parsed.get('toxicity_support_score', 0.5))
    tolerance_score = float(parsed.get('tolerance_support_score', 0.5))
    dominant_side = str(parsed.get('dominant_side', 'mixed'))
    score_margin = toxicity_score - tolerance_score
    prediction = 'positive' if score_margin >= margin_threshold else 'negative'
    distance_to_threshold = abs(score_margin - margin_threshold)
    confidence = max(0.5, min(0.95, 0.5 + distance_to_threshold * 0.9))
    parsed['prediction'] = prediction
    parsed['confidence'] = round(confidence, 4)
    parsed['score_margin'] = round(score_margin, 4)
    result['derived_decision'] = {'prediction': prediction, 'confidence': round(confidence, 4), 'score_margin': round(score_margin, 4), 'calibrated_margin': round(score_margin, 4), 'summary_toxicity_index': round(float(parsed.get('summary_toxicity_index', toxicity_score)), 4), 'fused_toxicity_index': round(float(parsed.get('fused_toxicity_index', toxicity_score)), 4), 'positive_margin_threshold': round(margin_threshold, 4), 'toxicity_support_score': round(toxicity_score, 4), 'tolerance_support_score': round(tolerance_score, 4), 'dominant_side': dominant_side}
    return result

def attach_task4_rule_decision(rule_profile: dict, margin_threshold: float=TASK4_POSITIVE_MARGIN_THRESHOLD) -> dict:
    toxicity_score = float(rule_profile['toxicity_support_score'])
    tolerance_score = float(rule_profile['tolerance_support_score'])
    score_margin = toxicity_score - tolerance_score
    calibrated_margin = float(rule_profile['rule_centered_score'])
    prediction = str(rule_profile['prediction'])
    dominant_side = str(rule_profile['dominant_side'])
    confidence = max(0.5, min(0.95, 0.5 + abs(toxicity_score - 0.5) * 0.9))
    parsed = {'toxicity_support_score': round(toxicity_score, 4), 'tolerance_support_score': round(tolerance_score, 4), 'dominant_side': dominant_side, 'rationale': list(rule_profile.get('rationale', [])), 'prediction': prediction, 'confidence': round(confidence, 4), 'score_margin': round(score_margin, 4), 'calibrated_margin': round(calibrated_margin, 4), 'summary_toxicity_index': round(float(rule_profile['rule_raw_score']), 4), 'fused_toxicity_index': round(float(rule_profile['rule_raw_score']), 4)}
    return {'mode': 'rule_first_v3', 'parsed_response': parsed, 'derived_decision': {'prediction': prediction, 'confidence': round(confidence, 4), 'score_margin': round(score_margin, 4), 'calibrated_margin': round(calibrated_margin, 4), 'summary_toxicity_index': round(float(rule_profile['rule_raw_score']), 4), 'fused_toxicity_index': round(float(rule_profile['rule_raw_score']), 4), 'positive_margin_threshold': round(margin_threshold, 4), 'toxicity_support_score': round(toxicity_score, 4), 'tolerance_support_score': round(tolerance_score, 4), 'dominant_side': dominant_side, 'rule_threshold': round(float(rule_profile['rule_threshold']), 4)}, 'raw_response': {'usage': {'total_tokens': 0}}}
