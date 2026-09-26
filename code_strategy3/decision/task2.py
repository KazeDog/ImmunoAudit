def attach_task2_risk_decision(result: dict) -> dict:
    parsed = result.get('parsed_response', {})
    raw_score = float(parsed.get('risk_score', 0.5))
    risk_score = max(0.0, min(1.0, raw_score))
    derived_group = derive_risk_group(risk_score)
    confidence = max(0.5, min(0.95, 0.5 + abs(risk_score - 0.5) * 0.9))
    parsed['risk_score'] = round(risk_score, 4)
    parsed['risk_group'] = derived_group
    parsed['confidence'] = round(confidence, 4)
    result['derived_decision'] = {'risk_group': derived_group, 'risk_score': round(risk_score, 4), 'confidence': round(confidence, 4)}
    return result

def derive_risk_group(risk_score: float) -> str:
    if risk_score >= 0.67:
        return 'high'
    if risk_score <= 0.33:
        return 'low'
    return 'intermediate'
