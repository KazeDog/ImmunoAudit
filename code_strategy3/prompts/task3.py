import json
RESPONSE_SUPPORTING_PROGRAMS = {'interferon_gamma_program', 'antigen_presentation_program', 'cytotoxic_lymphocyte_program', 'chemokine_inflamed_program', 'b_cell_plasma_program'}
RESISTANCE_SUPPORTING_PROGRAMS = {'tgfb_stromal_program', 't_cell_exhaustion_program', 'myeloid_inflammation_program', 'epithelial_keratin_program'}
TASK3_SYSTEM_PROMPT = '\nYou are a biomedical decision-support model for immunotherapy response prediction.\nUse only the supplied sample summary.\nDo not use cohort identity, benchmark names, or formatting cues to infer labels.\nScore response-supporting evidence and resistance-supporting evidence separately on a 0 to 1 scale.\nDo not treat proliferation alone as sufficient evidence for response.\nWhen evidence is mixed, keep both scores close rather than forcing a one-sided answer.\nIf at least two strong immune-response programs co-occur, do not let one isolated resistance program automatically dominate.\nReturn only JSON with dual scores, dominant_side, and 3 short rationale items.\n'.strip()

def build_task3_messages(sample_summary: dict) -> tuple[str, str]:
    compact_summary = build_compact_task3_summary(sample_summary)
    user_prompt = '\n'.join(['Assess this sample using dual scoring rather than direct one-step classification.', 'Program activity reflects relative within-sample enrichment of curated immune and tumor biology gene sets.', 'Ranked genes and themes are neutral descriptors and do not encode the ground-truth label.', 'Weigh response-supporting evidence against resistance-supporting evidence.', 'Output response_support_score and resistance_support_score on a 0 to 1 scale.', 'If evidence is mixed, use closer scores instead of collapsing to one side.', 'Do not let a single resistance program automatically outweigh multiple strong immune-response programs.', json.dumps(compact_summary, ensure_ascii=False, separators=(',', ':'))])
    return (TASK3_SYSTEM_PROMPT, user_prompt)

def build_compact_task3_summary(sample_summary: dict) -> dict:
    top_genes = []
    for feature in sample_summary.get('top_ranked_features', [])[:6]:
        item = {'gene_symbol': feature.get('gene_symbol'), 'value': feature.get('value')}
        if feature.get('biological_tags'):
            item['biological_tags'] = feature['biological_tags'][:2]
        if item['gene_symbol'] is None:
            item['feature_id'] = feature.get('feature_id')
        top_genes.append(item)
    top_themes = [{'theme': theme.get('theme'), 'matched_feature_count': theme.get('matched_feature_count')} for theme in sample_summary.get('biological_theme_snapshot', [])[:4]]
    top_programs = [{'program': program.get('program'), 'activity_level': program.get('activity_level'), 'relative_activity_score': program.get('relative_activity_score'), 'representative_genes': program.get('representative_genes', [])[:4]} for program in sample_summary.get('program_activity_snapshot', [])[:4]]
    evidence_balance = build_program_evidence_balance(sample_summary.get('program_activity_snapshot', []))
    return {'sample_id': sample_summary.get('sample_id'), 'top_ranked_genes': top_genes, 'dominant_biological_themes': top_themes, 'dominant_programs': top_programs, 'evidence_balance': evidence_balance}

def build_program_evidence_balance(programs: list[dict]) -> dict:
    supportive = []
    resistance = []
    context = []
    for program in programs:
        compact = {'program': program.get('program'), 'activity_level': program.get('activity_level'), 'relative_activity_score': program.get('relative_activity_score')}
        program_name = program.get('program')
        if program_name in RESPONSE_SUPPORTING_PROGRAMS:
            supportive.append(compact)
        elif program_name in RESISTANCE_SUPPORTING_PROGRAMS:
            resistance.append(compact)
        elif program_name == 'cell_cycle_program':
            context.append(compact)
    supportive.sort(key=lambda item: (-float(item['relative_activity_score']), item['program']))
    resistance.sort(key=lambda item: (-float(item['relative_activity_score']), item['program']))
    context.sort(key=lambda item: (-float(item['relative_activity_score']), item['program']))
    response_strength = round(sum((float(item['relative_activity_score']) for item in supportive[:3])), 4)
    resistance_strength = round(sum((float(item['relative_activity_score']) for item in resistance[:3])), 4)
    return {'response_supporting_programs': supportive[:3], 'resistance_supporting_programs': resistance[:3], 'context_programs': context[:2], 'response_supporting_count': len(supportive), 'resistance_supporting_count': len(resistance), 'response_strength_top3': response_strength, 'resistance_strength_top3': resistance_strength, 'net_response_minus_resistance': round(response_strength - resistance_strength, 4)}
