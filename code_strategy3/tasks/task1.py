from submission_paths import path as _submission_path
from pathlib import Path
import pandas as pd
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
TASK1_DATA_PATH = WORKSPACE_DIR / 'processed_data_strategy1' / 'task1_X_final.csv'
TASK1_LABEL_PATH = WORKSPACE_DIR / 'processed_data_strategy1' / 'task1_y_final.csv'
GENERIC_GENE_PREFIXES = ('RPL', 'RPS', 'MT-')
GENERIC_GENE_NAMES = {'ACTB', 'ACTG1', 'B2M', 'EEF1A1', 'GAPDH', 'HSP90AA1', 'HSP90AB1', 'MALAT1', 'TMSB10', 'TMSB4X', 'UBB', 'UBC'}
BIOLOGICAL_THEMES = {'interferon_response': {'STAT1', 'STAT2', 'IRF1', 'IRF7', 'IFI6', 'IFI27', 'IFI35', 'IFI44', 'IFI44L', 'IFIT1', 'IFIT2', 'IFIT3', 'IFITM1', 'ISG15', 'MX1', 'OAS1', 'OAS2', 'RSAD2'}, 'antigen_presentation': {'B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRA', 'HLA-DRB1', 'HLA-E', 'PSMB8', 'PSMB9', 'TAP1', 'TAP2'}, 'cytotoxic_effector': {'CD8A', 'CD8B', 'EOMES', 'GNLY', 'GZMA', 'GZMB', 'IFNG', 'NKG7', 'PRF1'}, 't_cell_activation': {'CD2', 'CD3D', 'CD3E', 'CD27', 'CD28', 'CXCL9', 'CXCL10', 'CXCL11', 'ICOS', 'PDCD1', 'TIGIT'}, 'b_cell_plasma': {'BANK1', 'BLK', 'CD19', 'CD22', 'CD79A', 'CD79B', 'MZB1', 'MS4A1', 'SDC1'}, 'myeloid_inflammation': {'CCR1', 'CCR2', 'CD14', 'CXCL8', 'FCGR3A', 'IL1B', 'LILRB1', 'S100A8', 'S100A9', 'TYMP'}, 'stroma_emt': {'ACTA2', 'COL1A1', 'COL1A2', 'COL3A1', 'COL5A1', 'FAP', 'FN1', 'MMP2', 'TAGLN', 'VIM'}, 'cell_cycle': {'AURKA', 'BIRC5', 'CCNB1', 'CDC20', 'CDK1', 'CENPF', 'MKI67', 'TOP2A', 'UBE2C'}}
PROGRAM_GENESETS = {'interferon_gamma_program': {'B2M', 'CXCL9', 'CXCL10', 'CXCL11', 'GBP1', 'HLA-A', 'HLA-B', 'HLA-C', 'IDO1', 'IFI27', 'IFIT1', 'IFIT2', 'IFIT3', 'IRF1', 'ISG15', 'PSMB8', 'PSMB9', 'STAT1', 'TAP1'}, 'antigen_presentation_program': {'B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DRA', 'HLA-DRB1', 'PSMB8', 'PSMB9', 'TAP1', 'TAP2'}, 'cytotoxic_lymphocyte_program': {'CD2', 'CD3D', 'CD3E', 'CD8A', 'CD8B', 'EOMES', 'GNLY', 'GZMA', 'GZMB', 'GZMK', 'IFNG', 'NKG7', 'PRF1'}, 'chemokine_inflamed_program': {'CCL4', 'CCL5', 'CXCL9', 'CXCL10', 'CXCL11', 'CXCR3', 'STAT1'}, 'b_cell_plasma_program': {'BANK1', 'BLK', 'CD19', 'CD22', 'CD79A', 'CD79B', 'MZB1', 'MS4A1', 'SDC1'}, 'myeloid_inflammation_program': {'CCR1', 'CCR2', 'CD14', 'CXCL8', 'FCGR3A', 'IL1B', 'LILRB1', 'S100A8', 'S100A9', 'TYMP'}, 'tgfb_stromal_program': {'ACTA2', 'COL1A1', 'COL1A2', 'COL3A1', 'COL5A1', 'FAP', 'FN1', 'MMP2', 'TAGLN', 'TGFB1', 'VIM'}, 't_cell_exhaustion_program': {'CTLA4', 'ENTPD1', 'HAVCR2', 'LAG3', 'PDCD1', 'TIGIT', 'TOX'}, 'cell_cycle_program': {'AURKA', 'BIRC5', 'CCNB1', 'CDC20', 'CDC6', 'CDK1', 'CENPF', 'MKI67', 'PCNA', 'TOP2A', 'UBE2C'}}

def load_task1_processed_dataset() -> pd.DataFrame:
    if not TASK1_DATA_PATH.exists():
        raise FileNotFoundError(f'Task 1 processed dataset not found: {TASK1_DATA_PATH}')
    return pd.read_csv(TASK1_DATA_PATH, index_col=0)

def load_task1_labels() -> pd.Series:
    if not TASK1_LABEL_PATH.exists():
        raise FileNotFoundError(f'Task 1 labels not found: {TASK1_LABEL_PATH}')
    labels = pd.read_csv(TASK1_LABEL_PATH, index_col=0)['response'].astype(int)
    labels.index = labels.index.astype(str)
    return labels

def build_task1_sample_summary(df: pd.DataFrame, sample_id: str | None=None, top_k_features: int=12) -> dict:
    if df.empty:
        raise ValueError('Task 1 dataframe is empty.')
    if sample_id is None:
        sample_id = str(df.index[0])
    row = df.loc[sample_id].astype(float)
    gene_profile = build_gene_expression_profile(row)
    top_genes = build_top_gene_records(gene_profile, top_k_features=top_k_features)
    program_activity = summarize_program_activity(gene_profile)
    return {'task_name': 'task1', 'sample_id': str(sample_id), 'n_features': int(df.shape[1]), 'summary_type': 'patient_level_gene_expression_snapshot', 'feature_ranking_rule': 'genes sorted by observed value in descending order after filtering ubiquitous markers from the top-gene display', 'top_ranked_genes': top_genes, 'biological_theme_snapshot': summarize_biological_themes(top_genes), 'program_activity_snapshot': program_activity, 'calibration_snapshot': build_calibration_snapshot(program_activity)}

def build_gene_expression_profile(row: pd.Series) -> pd.Series:
    gene_values = {normalize_gene_symbol(gene_name): float(value) for gene_name, value in row.items()}
    profile = pd.Series(gene_values, dtype=float)
    return profile.sort_values(ascending=False)

def build_top_gene_records(gene_profile: pd.Series, top_k_features: int) -> list[dict]:
    informative = [gene for gene in gene_profile.index if not is_generic_gene(gene)]
    selected = informative[:top_k_features]
    if len(selected) < top_k_features:
        for gene in gene_profile.index:
            if gene not in selected:
                selected.append(gene)
            if len(selected) >= top_k_features:
                break
    records = []
    for gene in selected:
        record = {'gene_symbol': gene, 'value': round(float(gene_profile.loc[gene]), 4)}
        tags = infer_feature_themes(gene)
        if tags:
            record['biological_tags'] = tags
        records.append(record)
    return records

def is_generic_gene(gene_symbol: str) -> bool:
    if gene_symbol in GENERIC_GENE_NAMES:
        return True
    return any((gene_symbol.startswith(prefix) for prefix in GENERIC_GENE_PREFIXES))

def summarize_biological_themes(feature_records: list[dict]) -> list[dict]:
    theme_counts: dict[str, int] = {}
    theme_genes: dict[str, list[str]] = {}
    for record in feature_records:
        gene_symbol = record.get('gene_symbol')
        for theme in record.get('biological_tags', []):
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
            if gene_symbol:
                theme_genes.setdefault(theme, []).append(str(gene_symbol))
    ranked = sorted(theme_counts.items(), key=lambda item: (-item[1], item[0]))
    summaries = []
    for theme, count in ranked[:5]:
        summaries.append({'theme': theme, 'matched_gene_count': count, 'example_genes': sorted(set(theme_genes.get(theme, [])))[:5]})
    return summaries

def summarize_program_activity(gene_profile: pd.Series, top_n: int=6) -> list[dict]:
    percentile_profile = gene_profile.rank(pct=True, method='average')
    program_summaries = []
    for program_name, genes in PROGRAM_GENESETS.items():
        overlap = sorted(set(percentile_profile.index).intersection(genes))
        if len(overlap) < 2:
            continue
        program_score = float(percentile_profile.loc[overlap].mean())
        if program_score < 0.55:
            continue
        representative_genes = gene_profile.loc[overlap].sort_values(ascending=False).head(5).index.tolist()
        program_summaries.append({'program': program_name, 'relative_activity_score': round(program_score, 4), 'activity_level': describe_program_activity(program_score), 'matched_gene_count': len(overlap), 'representative_genes': representative_genes})
    program_summaries.sort(key=lambda item: (-item['relative_activity_score'], -item['matched_gene_count'], item['program']))
    return program_summaries[:top_n]

def build_calibration_snapshot(program_activity: list[dict]) -> dict:
    scores = {item['program']: float(item['relative_activity_score']) for item in program_activity}
    response_axes = {'cytotoxic_lymphocyte_program': scores.get('cytotoxic_lymphocyte_program', 0.0), 'interferon_gamma_program': scores.get('interferon_gamma_program', 0.0), 'b_cell_plasma_program': scores.get('b_cell_plasma_program', 0.0), 'chemokine_inflamed_program': scores.get('chemokine_inflamed_program', 0.0), 'antigen_presentation_program': scores.get('antigen_presentation_program', 0.0)}
    resistance_axes = {'t_cell_exhaustion_program': scores.get('t_cell_exhaustion_program', 0.0), 'myeloid_inflammation_program': scores.get('myeloid_inflammation_program', 0.0), 'cell_cycle_program': scores.get('cell_cycle_program', 0.0), 'tgfb_stromal_program': scores.get('tgfb_stromal_program', 0.0)}
    response_high_axes = sum((1 for key in ['cytotoxic_lymphocyte_program', 'interferon_gamma_program', 'b_cell_plasma_program', 'chemokine_inflamed_program'] if response_axes.get(key, 0.0) >= 0.85))
    resistance_high_axes = sum((1 for key in resistance_axes if resistance_axes.get(key, 0.0) >= 0.85))
    calibrated_response_score = 0.32 * response_axes['cytotoxic_lymphocyte_program'] + 0.18 * response_axes['interferon_gamma_program'] + 0.28 * response_axes['b_cell_plasma_program'] + 0.07 * response_axes['chemokine_inflamed_program'] + 0.05 * response_axes['antigen_presentation_program']
    calibrated_resistance_score = 0.34 * resistance_axes['t_cell_exhaustion_program'] + 0.28 * resistance_axes['myeloid_inflammation_program'] + 0.28 * resistance_axes['cell_cycle_program'] + 0.1 * resistance_axes['tgfb_stromal_program']
    net_evidence = calibrated_response_score - calibrated_resistance_score
    return {'response_axes': round_nested_floats(response_axes), 'resistance_axes': round_nested_floats(resistance_axes), 'response_high_axes_count': int(response_high_axes), 'resistance_high_axes_count': int(resistance_high_axes), 'calibrated_response_score': round(calibrated_response_score, 4), 'calibrated_resistance_score': round(calibrated_resistance_score, 4), 'calibrated_net_evidence': round(net_evidence, 4), 'calibration_notes': ['antigen_presentation_program is common and should not dominate the final decision by itself', 'b_cell_plasma_program is a stronger response-specific modifier than antigen presentation alone', 'high myeloid_inflammation_program and high cell_cycle_program should materially increase resistance', 'high t_cell_exhaustion_program should offset cytotoxic activation when both are strong']}

def infer_feature_themes(gene_symbol: str) -> list[str]:
    matches = []
    for theme, genes in BIOLOGICAL_THEMES.items():
        if gene_symbol in genes:
            matches.append(theme)
    return matches

def describe_program_activity(score: float) -> str:
    if score >= 0.85:
        return 'very_high'
    if score >= 0.72:
        return 'high'
    if score >= 0.6:
        return 'moderate'
    return 'mild'

def normalize_gene_symbol(gene_symbol: str) -> str:
    return str(gene_symbol).strip().upper()

def round_nested_floats(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}
