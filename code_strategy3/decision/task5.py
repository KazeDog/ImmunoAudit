def attach_task5_risk_decision(result: dict, sample_summary: dict | None=None) -> dict:
    parsed = result.get('parsed_response', {})
    raw_score = float(parsed.get('risk_score', 0.5))
    risk_score = max(0.0, min(1.0, raw_score))
    recalibration = apply_task5_targeted_recalibration(risk_score, sample_summary)
    risk_score = recalibration['risk_score']
    derived_group = derive_risk_group(risk_score)
    confidence = max(0.5, min(0.95, 0.5 + abs(risk_score - 0.5) * 0.9))
    parsed['risk_score'] = round(risk_score, 4)
    parsed['risk_group'] = derived_group
    parsed['confidence'] = round(confidence, 4)
    if recalibration['applied']:
        parsed['recalibration'] = recalibration
    result['derived_decision'] = {'risk_group': derived_group, 'risk_score': round(risk_score, 4), 'confidence': round(confidence, 4), 'recalibration_version': recalibration['version'], 'recalibration_applied': recalibration['applied'], 'recalibration_tags': recalibration['tags']}
    return result

def apply_task5_targeted_recalibration(risk_score: float, sample_summary: dict | None) -> dict:
    version = 'v0.4.1'
    if not sample_summary:
        return {'version': version, 'applied': False, 'risk_score': round(risk_score, 4), 'tags': []}
    ecology = sample_summary.get('ecology_snapshot', {})
    calibration = sample_summary.get('calibration_snapshot', {})
    inflamed = float(ecology.get('inflamed_support_score', 0.0))
    suppressive = float(ecology.get('suppressive_pressure_score', 0.0))
    proliferative = float(ecology.get('proliferative_pressure_score', 0.0))
    net_risk = float(calibration.get('calibrated_net_risk', ecology.get('net_risk_score', 0.0)))
    strong_protective = int(calibration.get('strong_protective_signal_count', ecology.get('strong_protective_signal_count', 0)))
    strong_adverse = int(calibration.get('strong_adverse_signal_count', ecology.get('strong_adverse_signal_count', 0)))
    dominant_adverse_axes = int(calibration.get('dominant_adverse_axis_count', ecology.get('dominant_adverse_axis_count', 0)))
    adaptive_reserve = float(calibration.get('adaptive_reserve_score', ecology.get('adaptive_reserve_score', 0.0)))
    barrier_pressure = float(calibration.get('barrier_pressure_score', ecology.get('barrier_pressure_score', 0.0)))
    proliferative_escape = float(calibration.get('proliferative_escape_score', ecology.get('proliferative_escape_score', 0.0)))
    mixed_profile = bool(calibration.get('mixed_profile', False))
    near_zero = bool(calibration.get('near_zero_net_risk', False))
    low_risk_ready = bool(calibration.get('low_risk_ready', False))
    low_risk_blocked = bool(calibration.get('low_risk_blocked', False))
    favorable_immune_ready = bool(calibration.get('favorable_immune_ready', False))
    high_risk_ready = bool(calibration.get('high_risk_ready', False))
    excluded_inflamed = bool(calibration.get('excluded_inflamed_state', calibration.get('immune_excluded_state', False)))
    stromal_dominant = bool(calibration.get('stromal_dominant_state', False))
    myeloid_exhausted = bool(calibration.get('myeloid_exhausted_state', False))
    proliferative_escape_state = bool(calibration.get('proliferative_escape_state', False))
    score = float(risk_score)
    tags: list[str] = []
    low_candidate = not low_risk_blocked and (low_risk_ready or (favorable_immune_ready and inflamed >= 0.8 and (suppressive < 0.58) and (proliferative < 0.64) and (strong_adverse <= 1) and (net_risk <= -0.02)))
    if low_candidate and score > 0.33:
        target = 0.24 if net_risk <= -0.08 else 0.3
        score = min(score, target)
        tags.append('release_low_risk')
    false_high_guard = score >= 0.67 and inflamed >= 0.76 and (adaptive_reserve >= 0.68) and (strong_protective >= 3) and (net_risk < 0.16) and (barrier_pressure < 0.68) and (not high_risk_ready or mixed_profile or excluded_inflamed or (0.02 <= net_risk <= 0.12 and inflamed >= 0.82))
    if false_high_guard:
        target = 0.6 if near_zero else 0.63
        score = min(score, target)
        tags.append('suppress_false_high')
    true_high_promotion = score < 0.67 and (high_risk_ready and dominant_adverse_axes >= 2 and (barrier_pressure >= 0.66) and (net_risk >= 0.14) and (stromal_dominant or myeloid_exhausted or proliferative_escape_state) or (dominant_adverse_axes >= 3 and strong_adverse >= 2 and (proliferative_escape >= 0.7) and (barrier_pressure >= 0.68) and (net_risk >= 0.16) and (adaptive_reserve < 0.68)))
    if true_high_promotion:
        target = 0.72 if net_risk >= 0.2 else 0.68
        score = max(score, target)
        tags.append('promote_true_high')
    moderate_adverse_lift = not true_high_promotion and (dominant_adverse_axes >= 2 or excluded_inflamed or proliferative_escape_state) and (net_risk >= 0.06) and (score < 0.64)
    if moderate_adverse_lift:
        target = 0.64 if net_risk >= 0.12 else 0.6
        score = max(score, target)
        tags.append('lift_moderate_adverse')
    if not low_candidate and (not true_high_promotion) and (not moderate_adverse_lift):
        if near_zero:
            score = min(max(score, 0.5), 0.58)
            tags.append('compress_near_zero')
        elif favorable_immune_ready and adaptive_reserve >= 0.68 and (score > 0.5):
            score = min(score, 0.48)
            tags.append('spread_favorable_intermediate')
        elif -0.02 <= net_risk <= 0.06 and score > 0.64:
            score = 0.62
            tags.append('soften_borderline_high')
        elif 0.02 <= net_risk <= 0.08:
            score = min(max(score, 0.52), 0.58)
            tags.append('spread_intermediate_mid')
        elif 0.08 < net_risk < 0.14 and score > 0.66:
            score = 0.64
            tags.append('cap_borderline_high')
    score = max(0.0, min(1.0, round(score, 4)))
    return {'version': version, 'applied': bool(tags), 'risk_score': score, 'tags': tags}

def derive_risk_group(risk_score: float) -> str:
    if risk_score >= 0.67:
        return 'high'
    if risk_score <= 0.33:
        return 'low'
    return 'intermediate'
