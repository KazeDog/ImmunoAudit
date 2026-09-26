import json
RESPONSE_SUPPORTING_PROGRAMS = {'interferon_gamma_program', 'antigen_presentation_program', 'cytotoxic_lymphocyte_program', 'chemokine_inflamed_program', 'b_cell_plasma_program'}
RESISTANCE_SUPPORTING_PROGRAMS = {'tgfb_stromal_program', 't_cell_exhaustion_program', 'myeloid_inflammation_program'}
TASK1_SYSTEM_PROMPT = '\nYou are a biomedical decision-support model for immunotherapy response prediction.\nUse only the supplied sample summary.\nDo not infer the label from sample identifiers, row order, or formatting cues.\nScore response-supporting evidence and resistance-supporting evidence separately on a 0 to 1 scale.\nPrediction positive means responder-associated biology.\nPrediction negative means non-responder-associated biology.\nTreat antigen_presentation_program as a common background immune-visibility signal rather than a decisive responder marker.\nUse cytotoxic, interferon, and especially B-cell/plasma evidence as more specific response-supporting signals.\nTreat myeloid inflammation, cell-cycle/proliferative activity, and T-cell exhaustion as major resistance-supporting signals.\nIf strong cytotoxic activity co-occurs with strong exhaustion and myeloid/cell-cycle evidence, keep the sample mixed or resistance-leaning instead of optimistic.\nUse wider score separation only when one side clearly dominates across multiple orthogonal programs.\nIf the calibration snapshot shows calibrated_net_evidence near zero, prefer mixed or resistance-leaning judgments rather than optimistic response-leaning ones.\nReturn only JSON with response_support_score, resistance_support_score, dominant_side, and 3 short rationale items.\n'.strip()

def build_task1_messages(sample_summary: dict) -> tuple[str, str]:
    compact_summary = build_compact_task1_summary(sample_summary)
    user_prompt = '\n'.join(['Assess this Task 1 sample using dual scoring rather than direct one-step classification.', 'Top ranked genes are descriptive features from one patient-level expression profile and do not encode the ground-truth label.', 'Program activity reflects within-sample relative enrichment of curated immune and resistance-associated gene programs.', 'Weigh response-supporting evidence against resistance-supporting evidence.', 'Output response_support_score and resistance_support_score on a 0 to 1 scale.', 'Use the calibration_snapshot as an explicit decision aid rather than as background prose.', 'Do not treat antigen_presentation_program as sufficient positive evidence on its own because it is common across many samples.', 'Upweight b_cell_plasma_program when present; it is a more response-specific positive modifier than antigen presentation alone.', 'Upweight myeloid_inflammation_program and cell_cycle_program as resistance drivers.', 'If response_support_score and resistance_support_score are both high, only make the sample response-dominant when cytotoxic/interferon/B-cell evidence clearly exceeds exhaustion/myeloid/cell-cycle evidence on several axes.', 'If calibration_snapshot.calibrated_net_evidence is near zero or negative, do not output a clearly response-dominant score pair.', 'Only use dominant_side=response when response_support_score exceeds resistance_support_score by a meaningful margin and resistance_high_axes_count is not comparable to response_high_axes_count.', json.dumps(compact_summary, ensure_ascii=False, separators=(',', ':'))])
    return (TASK1_SYSTEM_PROMPT, user_prompt)

def build_compact_task1_summary(sample_summary: dict) -> dict:
    top_genes = []
    for gene in sample_summary.get('top_ranked_genes', [])[:8]:
        item = {'gene_symbol': gene.get('gene_symbol'), 'value': gene.get('value')}
        if gene.get('biological_tags'):
            item['biological_tags'] = gene['biological_tags'][:2]
        top_genes.append(item)
    top_themes = [{'theme': theme.get('theme'), 'matched_gene_count': theme.get('matched_gene_count')} for theme in sample_summary.get('biological_theme_snapshot', [])[:4]]
    top_programs = [{'program': program.get('program'), 'activity_level': program.get('activity_level'), 'relative_activity_score': program.get('relative_activity_score'), 'representative_genes': program.get('representative_genes', [])[:4]} for program in sample_summary.get('program_activity_snapshot', [])[:5]]
    return {'top_ranked_genes': top_genes, 'dominant_biological_themes': top_themes, 'dominant_programs': top_programs, 'evidence_balance': build_program_evidence_balance(sample_summary.get('program_activity_snapshot', [])), 'calibration_snapshot': sample_summary.get('calibration_snapshot', {})}

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
