from submission_paths import path as _submission_path
from pathlib import Path
import pandas as pd
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
TASK2_DATA_PATH = WORKSPACE_DIR / 'processed_data_strategy1' / 'task2_X_final.csv'
TASK2_LABEL_PATH = WORKSPACE_DIR / 'processed_data_strategy1' / 'task2_y_final.csv'
BIOLOGICAL_THEMES = {'immune_escape': {'B2M', 'HLA-A', 'HLA-B', 'IFNGR1', 'JAK1', 'JAK2', 'JAK3'}, 'checkpoint_axis': {'CD274', 'CTLA4', 'PDCD1', 'ICOSLG', 'IL10'}, 'mapk_signaling': {'ALK', 'ARAF', 'BRAF', 'EGFR', 'ERBB2', 'ERBB3', 'ERBB4', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'HRAS', 'KRAS', 'MAP2K1', 'MAP2K2', 'MET', 'NF1', 'NRAS', 'RET', 'ROS1'}, 'pi3k_akt_mtor': {'AKT1', 'AKT2', 'AKT3', 'INPP4A', 'INPP4B', 'INPPL1', 'MTOR', 'PDPK1', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG', 'PIK3R1', 'PIK3R2', 'PIK3R3', 'PTEN', 'TSC1', 'TSC2'}, 'cell_cycle_p53': {'AURKA', 'AURKB', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CDK4', 'CDK6', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C', 'CHEK1', 'CHEK2', 'MDM2', 'MDM4', 'MYC', 'RB1', 'TP53'}, 'dna_repair': {'ATM', 'ATR', 'BAP1', 'BARD1', 'BRCA1', 'BRCA2', 'BRIP1', 'CHEK1', 'CHEK2', 'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5', 'FANCA', 'FANCC', 'MLH1', 'MSH2', 'MSH3', 'MSH6', 'MUTYH', 'NBN', 'PALB2', 'PMS1', 'PMS2', 'POLE', 'POLD1', 'RAD51', 'SETD2'}, 'chromatin_remodeling': {'ARID1A', 'ARID1B', 'ARID2', 'ASXL1', 'ASXL2', 'ATRX', 'BAP1', 'CREBBP', 'DNMT3A', 'EP300', 'EZH2', 'KDM5C', 'KDM6A', 'KMT2A', 'KMT2B', 'KMT2C', 'KMT2D', 'PBRM1', 'SETD2', 'SMARCA4'}, 'wnt_beta_catenin': {'APC', 'AXIN1', 'AXIN2', 'CTNNB1', 'FAT1', 'RNF43'}, 'rtk_growth_factor': {'CSF1R', 'EGFR', 'EPHA3', 'EPHA5', 'EPHA7', 'EPHB1', 'ERBB2', 'ERBB3', 'ERBB4', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'FLT1', 'FLT3', 'FLT4', 'HGF', 'IGF1R', 'INSR', 'KDR', 'KIT', 'MET', 'PDGFRA', 'PDGFRB', 'RET', 'ROS1'}}
PROGRAM_GENESETS = {'antigen_presentation_escape_program': {'B2M', 'HLA-A', 'HLA-B', 'IFNGR1', 'JAK1', 'JAK2', 'JAK3'}, 'stk11_keap1_resistance_program': {'KEAP1', 'NFE2L2', 'STK11'}, 'mapk_signaling_program': {'BRAF', 'EGFR', 'ERBB2', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'HRAS', 'KRAS', 'MAP2K1', 'MAP2K2', 'MET', 'NF1', 'NRAS', 'RET', 'ROS1'}, 'pi3k_akt_mtor_program': {'AKT1', 'AKT2', 'AKT3', 'MTOR', 'PDPK1', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG', 'PIK3R1', 'PIK3R2', 'PIK3R3', 'PTEN', 'TSC1', 'TSC2'}, 'cell_cycle_p53_program': {'AURKA', 'AURKB', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CDK4', 'CDK6', 'CDKN2A', 'MDM2', 'MDM4', 'MYC', 'RB1', 'TP53'}, 'dna_repair_instability_program': {'ATM', 'ATR', 'BRCA1', 'BRCA2', 'BRIP1', 'CHEK1', 'CHEK2', 'FANCA', 'FANCC', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'POLE', 'POLD1'}, 'chromatin_remodeling_program': {'ARID1A', 'ARID1B', 'ARID2', 'ATRX', 'CREBBP', 'EP300', 'KDM5C', 'KDM6A', 'KMT2C', 'KMT2D', 'PBRM1', 'SETD2', 'SMARCA4', 'VHL'}, 'wnt_beta_catenin_program': {'APC', 'AXIN1', 'AXIN2', 'CTNNB1', 'FAT1', 'RNF43'}}
ADVERSE_RISK_GENES = {'B2M', 'CTNNB1', 'EGFR', 'HLA-A', 'HLA-B', 'IFNGR1', 'JAK1', 'JAK2', 'KEAP1', 'KRAS', 'MDM2', 'MET', 'MYC', 'PIK3CA', 'PTEN', 'RB1', 'STK11', 'TP53'}
FAVORABLE_CONTEXT_GENES = {'ARID2', 'ATM', 'BRCA1', 'BRCA2', 'MLH1', 'MSH2', 'MSH6', 'PBRM1', 'POLE', 'POLD1', 'PMS2', 'SETD2', 'VHL'}

def load_task2_processed_dataset() -> pd.DataFrame:
    if not TASK2_DATA_PATH.exists():
        raise FileNotFoundError(f'Task 2 processed dataset not found: {TASK2_DATA_PATH}')
    df = pd.read_csv(TASK2_DATA_PATH, index_col=0)
    df.index = df.index.astype(str)
    return df

def load_task2_labels() -> pd.DataFrame:
    if not TASK2_LABEL_PATH.exists():
        raise FileNotFoundError(f'Task 2 labels not found: {TASK2_LABEL_PATH}')
    labels = pd.read_csv(TASK2_LABEL_PATH, index_col=0)
    labels.index = labels.index.astype(str)
    labels['duration'] = labels['duration'].astype(float)
    labels['event'] = labels['event'].astype(int)
    return labels

def build_task2_sample_summary(df: pd.DataFrame, sample_id: str | None=None, top_k_features: int=12) -> dict:
    if df.empty:
        raise ValueError('Task 2 dataframe is empty.')
    if sample_id is None:
        sample_id = str(df.index[0])
    row = df.loc[sample_id].astype(float)
    altered_profile = build_altered_gene_profile(row)
    prevalence = get_task2_feature_prevalence(df)
    burden_snapshot = build_burden_snapshot(df, sample_id, row)
    top_genes = build_top_gene_records(altered_profile, prevalence, top_k_features=top_k_features)
    program_activity = summarize_program_activity(altered_profile)
    immunotherapy_relevance = build_immunotherapy_relevance_snapshot(altered_profile, burden_snapshot, program_activity)
    calibration_snapshot = build_calibration_snapshot(burden_snapshot=burden_snapshot, program_activity=program_activity, immunotherapy_relevance=immunotherapy_relevance)
    return {'task_name': 'task2', 'sample_id': str(sample_id), 'n_features': int(df.shape[1]), 'summary_type': 'patient_level_genomic_alteration_snapshot', 'feature_ranking_rule': 'altered genes prioritized by immunotherapy relevance, pathway centrality, and rarity within the dataset', 'altered_gene_count': int(altered_profile.shape[0]), 'top_altered_genes': top_genes, 'biological_theme_snapshot': summarize_biological_themes(top_genes), 'program_activity_snapshot': program_activity, 'burden_snapshot': burden_snapshot, 'immunotherapy_relevance_snapshot': immunotherapy_relevance, 'calibration_snapshot': calibration_snapshot}

def build_altered_gene_profile(row: pd.Series) -> pd.Series:
    altered = {normalize_gene_symbol(gene_name): float(value) for gene_name, value in row.items() if float(value) > 0}
    if not altered:
        return pd.Series(dtype=float)
    profile = pd.Series(altered, dtype=float)
    return profile.sort_index()

def build_top_gene_records(altered_profile: pd.Series, prevalence: pd.Series, top_k_features: int) -> list[dict]:
    if altered_profile.empty:
        return []
    ranked_genes = sorted(altered_profile.index.tolist(), key=lambda gene: (-compute_gene_priority(gene), float(prevalence.get(gene, 1.0)), gene))
    records = []
    for gene in ranked_genes[:top_k_features]:
        tags = infer_feature_themes(gene)
        record = {'gene_symbol': gene, 'value': round(float(altered_profile.loc[gene]), 4), 'dataset_prevalence': round(float(prevalence.get(gene, 0.0)), 4)}
        if tags:
            record['biological_tags'] = tags
        records.append(record)
    return records

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

def summarize_program_activity(altered_profile: pd.Series, top_n: int=6) -> list[dict]:
    if altered_profile.empty:
        return []
    altered_genes = set(altered_profile.index.tolist())
    program_summaries = []
    for program_name, genes in PROGRAM_GENESETS.items():
        overlap = sorted(altered_genes.intersection(genes))
        if not overlap:
            continue
        denominator = min(len(genes), 6)
        program_score = min(1.0, len(overlap) / max(1, denominator))
        program_summaries.append({'program': program_name, 'relative_activity_score': round(float(program_score), 4), 'activity_level': describe_program_activity(program_score), 'matched_gene_count': len(overlap), 'representative_genes': overlap[:5]})
    program_summaries.sort(key=lambda item: (-item['relative_activity_score'], -item['matched_gene_count'], item['program']))
    return program_summaries[:top_n]

def build_burden_snapshot(df: pd.DataFrame, sample_id: str, row: pd.Series) -> dict:
    burden_counts = get_task2_active_feature_counts(df)
    active_count = int((row > 0).sum())
    percentile = float(burden_counts.rank(pct=True, method='average').loc[str(sample_id)])
    return {'altered_gene_count': active_count, 'burden_percentile_within_dataset': round(percentile, 4), 'burden_level': describe_burden_level(percentile)}

def build_immunotherapy_relevance_snapshot(altered_profile: pd.Series, burden_snapshot: dict, program_activity: list[dict]) -> dict:
    altered_genes = set(altered_profile.index.tolist())
    program_scores = {item['program']: float(item['relative_activity_score']) for item in program_activity}
    adverse_markers = sorted(altered_genes.intersection(ADVERSE_RISK_GENES))
    favorable_markers = sorted(altered_genes.intersection(FAVORABLE_CONTEXT_GENES))
    return {'adverse_risk_markers': adverse_markers[:8], 'favorable_context_markers': favorable_markers[:8], 'antigen_presentation_escape_hits': sorted(altered_genes.intersection(PROGRAM_GENESETS['antigen_presentation_escape_program']))[:6], 'dna_repair_instability_hits': sorted(altered_genes.intersection(PROGRAM_GENESETS['dna_repair_instability_program']))[:6], 'mutation_burden_proxy': {'altered_gene_count': int(burden_snapshot['altered_gene_count']), 'burden_level': burden_snapshot['burden_level']}, 'risk_signal_strength': round(program_scores.get('antigen_presentation_escape_program', 0.0) + program_scores.get('stk11_keap1_resistance_program', 0.0) + program_scores.get('cell_cycle_p53_program', 0.0) + program_scores.get('mapk_signaling_program', 0.0) + program_scores.get('pi3k_akt_mtor_program', 0.0) + program_scores.get('wnt_beta_catenin_program', 0.0) + 0.12 * len(adverse_markers), 4), 'benefit_signal_strength': round(program_scores.get('dna_repair_instability_program', 0.0) + program_scores.get('chromatin_remodeling_program', 0.0) + 0.1 * len(favorable_markers) + 0.2 * float(burden_snapshot['burden_percentile_within_dataset']), 4)}

def build_calibration_snapshot(*, burden_snapshot: dict, program_activity: list[dict], immunotherapy_relevance: dict) -> dict:
    program_scores = {item['program']: float(item['relative_activity_score']) for item in program_activity}
    adverse_markers = immunotherapy_relevance.get('adverse_risk_markers', [])
    favorable_markers = immunotherapy_relevance.get('favorable_context_markers', [])
    burden_percentile = float(burden_snapshot.get('burden_percentile_within_dataset', 0.0))
    altered_gene_count = int(burden_snapshot.get('altered_gene_count', 0))
    burden_support = calibrate_burden_support(burden_percentile)
    favorable_marker_support = min(1.0, len(favorable_markers) / 3.0)
    adverse_marker_support = min(1.0, len(adverse_markers) / 3.0)
    adverse_axes = {'antigen_presentation_escape_program': program_scores.get('antigen_presentation_escape_program', 0.0), 'stk11_keap1_resistance_program': program_scores.get('stk11_keap1_resistance_program', 0.0), 'cell_cycle_p53_program': program_scores.get('cell_cycle_p53_program', 0.0), 'mapk_signaling_program': program_scores.get('mapk_signaling_program', 0.0), 'pi3k_akt_mtor_program': program_scores.get('pi3k_akt_mtor_program', 0.0), 'wnt_beta_catenin_program': program_scores.get('wnt_beta_catenin_program', 0.0), 'adverse_marker_load': adverse_marker_support}
    favorable_axes = {'dna_repair_instability_program': program_scores.get('dna_repair_instability_program', 0.0), 'chromatin_remodeling_program': program_scores.get('chromatin_remodeling_program', 0.0), 'burden_support': burden_support, 'favorable_marker_load': favorable_marker_support}
    adverse_high_axes = sum((1 for value in adverse_axes.values() if float(value) >= 0.55))
    favorable_high_axes = sum((1 for value in favorable_axes.values() if float(value) >= 0.55))
    dominant_adverse_axes = [axis_name for axis_name in ['antigen_presentation_escape_program', 'stk11_keap1_resistance_program', 'cell_cycle_p53_program', 'mapk_signaling_program', 'pi3k_akt_mtor_program', 'wnt_beta_catenin_program'] if adverse_axes[axis_name] >= 0.3]
    sparse_genomic_profile = altered_gene_count <= 4 and max(adverse_axes.values(), default=0.0) < 0.3 and (favorable_axes['dna_repair_instability_program'] < 0.3) and (favorable_axes['chromatin_remodeling_program'] < 0.55) and (len(favorable_markers) <= 1)
    repair_supported_favorable = favorable_axes['dna_repair_instability_program'] >= 0.3 and burden_support >= 0.6
    ultra_favorable_context = favorable_axes['dna_repair_instability_program'] >= 0.5 or (repair_supported_favorable and favorable_axes['chromatin_remodeling_program'] >= 0.5) or (favorable_axes['dna_repair_instability_program'] >= 0.3 and favorable_axes['favorable_marker_load'] >= 0.67 and (burden_support >= 0.6))
    strong_favorable_context = ultra_favorable_context or (favorable_axes['chromatin_remodeling_program'] >= 0.5 and favorable_axes['dna_repair_instability_program'] >= 0.15 and (burden_support >= 0.6))
    burden_only_favorable_context = burden_support >= 0.72 and favorable_axes['dna_repair_instability_program'] < 0.3 and (favorable_axes['favorable_marker_load'] < 0.67)
    calibrated_risk_score = 0.24 * adverse_axes['antigen_presentation_escape_program'] + 0.15 * adverse_axes['stk11_keap1_resistance_program'] + 0.16 * adverse_axes['cell_cycle_p53_program'] + 0.15 * adverse_axes['mapk_signaling_program'] + 0.1 * adverse_axes['pi3k_akt_mtor_program'] + 0.12 * adverse_axes['wnt_beta_catenin_program'] + 0.08 * adverse_axes['adverse_marker_load']
    calibrated_benefit_score = 0.3 * favorable_axes['dna_repair_instability_program'] + 0.22 * favorable_axes['chromatin_remodeling_program'] + 0.3 * favorable_axes['burden_support'] + 0.18 * favorable_axes['favorable_marker_load']
    calibrated_net_risk = calibrated_risk_score - calibrated_benefit_score
    high_risk_ready = not strong_favorable_context and (not burden_only_favorable_context) and (calibrated_net_risk >= 0.03) and (adverse_axes['stk11_keap1_resistance_program'] >= 0.3 and adverse_axes['cell_cycle_p53_program'] >= 0.3 or (adverse_axes['antigen_presentation_escape_program'] >= 0.3 and (adverse_axes['cell_cycle_p53_program'] >= 0.16 or adverse_axes['mapk_signaling_program'] >= 0.16 or adverse_axes['pi3k_akt_mtor_program'] >= 0.16)) or (len(dominant_adverse_axes) >= 2 and adverse_axes['adverse_marker_load'] >= 0.67) or (calibrated_net_risk >= 0.08 and adverse_axes['adverse_marker_load'] >= 1.0 and (not sparse_genomic_profile)))
    return {'adverse_axes': round_nested_floats(adverse_axes), 'favorable_axes': round_nested_floats(favorable_axes), 'adverse_high_axes_count': int(adverse_high_axes), 'favorable_high_axes_count': int(favorable_high_axes), 'dominant_adverse_axes': dominant_adverse_axes, 'dominant_adverse_axes_count': int(len(dominant_adverse_axes)), 'sparse_genomic_profile': bool(sparse_genomic_profile), 'repair_supported_favorable': bool(repair_supported_favorable), 'strong_favorable_context': bool(strong_favorable_context), 'ultra_favorable_context': bool(ultra_favorable_context), 'burden_only_favorable_context': bool(burden_only_favorable_context), 'high_risk_ready': bool(high_risk_ready), 'calibrated_risk_score': round(calibrated_risk_score, 4), 'calibrated_benefit_score': round(calibrated_benefit_score, 4), 'calibrated_net_risk': round(calibrated_net_risk, 4), 'calibration_notes': ['sparse low-burden profiles without explicit resistance should default to intermediate rather than low risk', 'low risk should usually require explicit DNA-repair support and clearly negative net risk rather than burden alone', 'strong adverse drivers should be softened when high burden and repair-favorable context co-occur', 'isolated chromatin remodeling without repair support is not enough for low risk', 'very high burden without repair support should usually remain intermediate', 'high risk is reasonable when multi-axis adverse context is explicit, net risk is positive, and strong favorable rescue is absent']}

def infer_feature_themes(gene_symbol: str) -> list[str]:
    matches = []
    for theme, genes in BIOLOGICAL_THEMES.items():
        if gene_symbol in genes:
            matches.append(theme)
    return matches

def compute_gene_priority(gene_symbol: str) -> int:
    tags = infer_feature_themes(gene_symbol)
    priority = len(tags)
    if gene_symbol in ADVERSE_RISK_GENES:
        priority += 4
    if gene_symbol in FAVORABLE_CONTEXT_GENES:
        priority += 3
    if gene_symbol in {'POLE', 'POLD1', 'PBRM1', 'STK11', 'KEAP1', 'TP53'}:
        priority += 2
    return priority

def get_task2_feature_prevalence(df: pd.DataFrame) -> pd.Series:
    cached = df.attrs.get('task2_feature_prevalence')
    if isinstance(cached, pd.Series):
        return cached
    prevalence = df.astype(float).mean(axis=0)
    prevalence.index = prevalence.index.astype(str).map(normalize_gene_symbol)
    df.attrs['task2_feature_prevalence'] = prevalence
    return prevalence

def get_task2_active_feature_counts(df: pd.DataFrame) -> pd.Series:
    cached = df.attrs.get('task2_active_feature_counts')
    if isinstance(cached, pd.Series):
        return cached
    counts = (df.astype(float) > 0).sum(axis=1).astype(int)
    counts.index = counts.index.astype(str)
    df.attrs['task2_active_feature_counts'] = counts
    return counts

def describe_program_activity(score: float) -> str:
    if score >= 0.8:
        return 'very_high'
    if score >= 0.5:
        return 'high'
    if score >= 0.3:
        return 'moderate'
    return 'mild'

def describe_burden_level(percentile: float) -> str:
    if percentile >= 0.9:
        return 'very_high'
    if percentile >= 0.7:
        return 'high'
    if percentile >= 0.35:
        return 'intermediate'
    return 'low'

def normalize_gene_symbol(gene_symbol: str) -> str:
    return str(gene_symbol).strip().upper()

def calibrate_burden_support(percentile: float) -> float:
    if percentile >= 0.9:
        return 0.95
    if percentile >= 0.7:
        return 0.72
    if percentile >= 0.5:
        return 0.45
    if percentile >= 0.35:
        return 0.25
    return 0.05

def round_nested_floats(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}
