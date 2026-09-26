from submission_paths import path as _submission_path
from pathlib import Path
import sys
import math
import pandas as pd
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
PROCESSED_DIR = WORKSPACE_DIR / 'processed_data_strategy1'
TASK4_X_PATH = PROCESSED_DIR / 'task4_X_final.csv'
TASK4_Y_PATH = PROCESSED_DIR / 'task4_y_final.csv'
AROMATIC_AA = set('FYW')
BASIC_AA = set('KRH')
ACIDIC_AA = set('DE')
POLAR_AA = set('STNQ')
GLY_PRO_AA = set('GP')
CYSTEINE_AA = set('C')
HYDROPHOBIC_AA = set('AVILM')

def load_task4_processed_dataset() -> pd.DataFrame:
    return pd.read_csv(TASK4_X_PATH, index_col=0)

def load_task4_labels() -> pd.Series:
    y = pd.read_csv(TASK4_Y_PATH, index_col=0)
    return y['toxicity'].astype(int)

def _motif_letters(motif: str) -> str:
    return ''.join((char for char in str(motif).upper() if char.isalpha()))

def _format_motif(motif: str) -> str:
    return str(motif).replace(' ', '·')

def infer_task4_motif_tags(motif: str) -> list[str]:
    raw = str(motif).upper()
    letters = _motif_letters(raw)
    tags = []
    if '*' in raw or ' ' in raw:
        tags.append('boundary')
    if any((char in AROMATIC_AA for char in letters)):
        tags.append('aromatic')
    if any((char in BASIC_AA for char in letters)):
        tags.append('basic')
    if any((char in ACIDIC_AA for char in letters)):
        tags.append('acidic')
    if any((char in POLAR_AA for char in letters)):
        tags.append('polar')
    if any((char in GLY_PRO_AA for char in letters)):
        tags.append('glycine_proline')
    if any((char in CYSTEINE_AA for char in letters)):
        tags.append('cysteine')
    if any((char in HYDROPHOBIC_AA for char in letters)):
        tags.append('hydrophobic')
    if letters and len(set(letters)) == 1:
        tags.append('repeat')
    if not tags:
        tags.append('neutral')
    return tags

def _describe_signal(value: float) -> str:
    if value >= 0.66:
        return 'high'
    if value >= 0.4:
        return 'moderate'
    return 'low'

def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0

def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
TASK4_V3_RULE_FEATURE_WEIGHTS = {'basic_fraction': 2.0, 'glycine_proline_fraction': 1.5, 'acidic_fraction': 1.2, 'hydrophobic_fraction': 1.0, 'polar_fraction': -1.2, 'regulatory_motif_fraction': -1.0, 'top1_share': -1.2}
TASK4_V3_RULE_POSITIVE_QUANTILE = 0.3
TASK4_V3_RULE_LOGIT_SCALE = 2.0

def build_task4_sample_summary(df: pd.DataFrame, sample_id: str | None=None, top_k_features: int=12) -> dict:
    if sample_id is None:
        sample_id = str(df.index[0])
    sample_id = str(sample_id)
    row = df.loc[sample_id].astype(float)
    prevalence = (df > 0).mean(axis=0)
    total_count = float(row.sum())
    active = row[row > 0].sort_values(ascending=False)
    nonzero_count = int(active.shape[0])
    top_active = active.head(top_k_features)
    top1_share = _safe_fraction(float(active.head(1).sum()), total_count)
    top3_share = _safe_fraction(float(active.head(3).sum()), total_count)
    top5_share = _safe_fraction(float(active.head(5).sum()), total_count)
    total_count_percentile = float((df.sum(axis=1) <= row.sum()).mean())
    richness_percentile = float(((df > 0).sum(axis=1) <= nonzero_count).mean())
    probabilities = active / total_count if total_count else active
    entropy = float(-(probabilities * probabilities.map(lambda value: math.log(value) if value > 0 else 0.0)).sum()) if total_count else 0.0
    normalized_entropy = float(entropy / math.log(nonzero_count)) if nonzero_count > 1 else 0.0
    motif_records = []
    theme_weight_totals = {'boundary': 0.0, 'aromatic': 0.0, 'basic': 0.0, 'acidic': 0.0, 'polar': 0.0, 'glycine_proline': 0.0, 'cysteine': 0.0, 'hydrophobic': 0.0, 'repeat': 0.0}
    rarity_values = []
    complexity_values = []
    rare_reactive_values = []
    common_reactive_values = []
    reactive_weight_total = 0.0
    regulatory_weight_total = 0.0
    for motif, count in active.items():
        tags = infer_task4_motif_tags(str(motif))
        weight = float(count)
        rarity = 1.0 - float(prevalence.loc[motif])
        rarity_values.append(weight * rarity)
        letters = _motif_letters(str(motif))
        complexity_values.append(len(set(letters)) / len(letters) if letters else 0.0)
        reactive_flag = any((tag in {'aromatic', 'basic', 'glycine_proline', 'cysteine', 'boundary', 'repeat'} for tag in tags))
        regulatory_flag = any((tag in {'polar', 'acidic'} for tag in tags))
        if reactive_flag:
            reactive_weight_total += weight
            rare_reactive_values.append(weight * rarity)
            common_reactive_values.append(weight * (1.0 - rarity))
        if regulatory_flag:
            regulatory_weight_total += weight
        for tag in tags:
            if tag in theme_weight_totals:
                theme_weight_totals[tag] += weight
    for motif, count in top_active.items():
        motif_records.append({'motif': str(motif), 'display_motif': _format_motif(str(motif)), 'count': int(count), 'dataset_prevalence': round(float(prevalence.loc[motif]), 4), 'rarity_score': round(1.0 - float(prevalence.loc[motif]), 4), 'tags': infer_task4_motif_tags(str(motif))})
    total_theme_weight = max(total_count, 1.0)
    theme_burdens = {f'{tag}_fraction': round(_safe_fraction(value, total_theme_weight), 4) for tag, value in theme_weight_totals.items()}
    chemistry_snapshot = {**theme_burdens, 'top1_share': round(top1_share, 4), 'top3_share': round(top3_share, 4), 'top5_share': round(top5_share, 4), 'normalized_entropy': round(normalized_entropy, 4), 'rarity_weighted_mean': round(_safe_fraction(sum(rarity_values), total_theme_weight), 4), 'motif_complexity_mean': round(_mean(complexity_values), 4), 'reactive_motif_fraction': round(_safe_fraction(reactive_weight_total, total_theme_weight), 4), 'regulatory_motif_fraction': round(_safe_fraction(regulatory_weight_total, total_theme_weight), 4), 'rare_reactive_fraction': round(_safe_fraction(sum(rare_reactive_values), total_theme_weight), 4), 'common_reactive_fraction': round(_safe_fraction(sum(common_reactive_values), total_theme_weight), 4)}
    autoreactivity_pressure = 0.28 * chemistry_snapshot['aromatic_fraction'] + 0.22 * chemistry_snapshot['basic_fraction'] + 0.16 * chemistry_snapshot['glycine_proline_fraction'] + 0.1 * chemistry_snapshot['cysteine_fraction'] + 0.12 * chemistry_snapshot['top5_share'] + 0.12 * chemistry_snapshot['boundary_fraction']
    tolerance_support = 0.24 * chemistry_snapshot['polar_fraction'] + 0.2 * chemistry_snapshot['acidic_fraction'] + 0.22 * chemistry_snapshot['normalized_entropy'] + 0.16 * (1.0 - chemistry_snapshot['top5_share']) + 0.1 * chemistry_snapshot['motif_complexity_mean'] + 0.08 * chemistry_snapshot['rarity_weighted_mean']
    clonal_focus = 0.65 * chemistry_snapshot['top5_share'] + 0.35 * (1.0 - chemistry_snapshot['normalized_entropy'])
    biochemical_instability = 0.32 * chemistry_snapshot['boundary_fraction'] + 0.24 * chemistry_snapshot['repeat_fraction'] + 0.22 * chemistry_snapshot['cysteine_fraction'] + 0.22 * chemistry_snapshot['glycine_proline_fraction']
    dominance_skew = 0.52 * chemistry_snapshot['top1_share'] + 0.3 * chemistry_snapshot['top3_share'] + 0.18 * chemistry_snapshot['top5_share'] - 0.34 * chemistry_snapshot['normalized_entropy']
    burden_score = 0.4 * total_count_percentile + 0.14 * (1.0 - richness_percentile) + 0.24 * chemistry_snapshot['top3_share'] + 0.22 * chemistry_snapshot['top5_share']
    reactive_motif_score = 0.32 * chemistry_snapshot['reactive_motif_fraction'] + 0.28 * chemistry_snapshot['aromatic_fraction'] + 0.18 * chemistry_snapshot['basic_fraction'] + 0.12 * chemistry_snapshot['cysteine_fraction'] + 0.1 * chemistry_snapshot['boundary_fraction']
    rare_reactive_burden = 0.38 * chemistry_snapshot['rare_reactive_fraction'] + 0.18 * chemistry_snapshot['rarity_weighted_mean'] + 0.26 * dominance_skew + 0.18 * burden_score
    common_motif_penalty = 0.54 * chemistry_snapshot['common_reactive_fraction'] + 0.26 * max(0.0, chemistry_snapshot['reactive_motif_fraction'] - chemistry_snapshot['rare_reactive_fraction']) + 0.2 * max(0.0, chemistry_snapshot['top5_share'] - chemistry_snapshot['top3_share'])
    regulatory_diversity_score = 0.24 * chemistry_snapshot['regulatory_motif_fraction'] + 0.24 * chemistry_snapshot['polar_fraction'] + 0.12 * chemistry_snapshot['acidic_fraction'] + 0.24 * chemistry_snapshot['normalized_entropy'] + 0.16 * chemistry_snapshot['motif_complexity_mean']
    summary_toxicity_index = 0.26 * dominance_skew + 0.2 * burden_score + 0.14 * autoreactivity_pressure + 0.12 * clonal_focus + 0.12 * biochemical_instability + 0.08 * reactive_motif_score + 0.08 * rare_reactive_burden - 0.18 * tolerance_support - 0.12 * regulatory_diversity_score - 0.06 * common_motif_penalty
    net_toxicity = 0.3 * summary_toxicity_index + 0.24 * autoreactivity_pressure + 0.14 * clonal_focus + 0.1 * biochemical_instability - 0.22 * tolerance_support
    calibration_snapshot = {'autoreactivity_pressure_score': round(autoreactivity_pressure, 4), 'autoreactivity_pressure_level': _describe_signal(autoreactivity_pressure), 'tolerance_support_score': round(tolerance_support, 4), 'tolerance_support_level': _describe_signal(tolerance_support), 'clonal_focus_score': round(clonal_focus, 4), 'clonal_focus_level': _describe_signal(clonal_focus), 'biochemical_instability_score': round(biochemical_instability, 4), 'biochemical_instability_level': _describe_signal(biochemical_instability), 'dominance_skew_score': round(dominance_skew, 4), 'dominance_skew_level': _describe_signal(dominance_skew), 'burden_score': round(burden_score, 4), 'burden_level': _describe_signal(burden_score), 'reactive_motif_score': round(reactive_motif_score, 4), 'reactive_motif_level': _describe_signal(reactive_motif_score), 'rare_reactive_burden': round(rare_reactive_burden, 4), 'rare_reactive_burden_level': _describe_signal(rare_reactive_burden), 'common_motif_penalty': round(common_motif_penalty, 4), 'regulatory_diversity_score': round(regulatory_diversity_score, 4), 'regulatory_diversity_level': _describe_signal(regulatory_diversity_score), 'summary_toxicity_index': round(summary_toxicity_index, 4), 'net_toxicity_score': round(net_toxicity, 4), 'near_zero_net_toxicity': bool(-0.015 <= net_toxicity <= 0.06), 'low_toxicity_ready': bool(regulatory_diversity_score >= 0.36 and common_motif_penalty <= 0.08 and (rare_reactive_burden < 0.1) and (dominance_skew < 0.05) and (net_toxicity <= 0.0)), 'high_toxicity_ready': bool(dominance_skew >= 0.055 and burden_score >= 0.28 and (summary_toxicity_index >= 0.02) or (autoreactivity_pressure >= 0.295 and clonal_focus >= 0.155 and (summary_toxicity_index >= 0.015))), 'mixed_profile': bool(-0.005 <= summary_toxicity_index <= 0.02 or (-0.005 <= net_toxicity <= 0.04 and 0.075 <= rare_reactive_burden <= 0.11) or (reactive_motif_score >= 0.22 and regulatory_diversity_score >= 0.34))}
    repertoire_snapshot = {'total_3mer_count': int(total_count), 'active_3mer_count': nonzero_count, 'total_count_percentile': round(total_count_percentile, 4), 'richness_percentile': round(richness_percentile, 4), 'top1_share': round(top1_share, 4), 'top3_share': round(top3_share, 4), 'top5_share': round(top5_share, 4), 'normalized_entropy': round(normalized_entropy, 4)}
    return {'task_name': 'task4', 'sample_id': sample_id, 'repertoire_snapshot': repertoire_snapshot, 'chemistry_snapshot': chemistry_snapshot, 'calibration_snapshot': calibration_snapshot, 'rule_snapshot': build_task4_v3_rule_inputs({'sample_id': sample_id, 'repertoire_snapshot': repertoire_snapshot, 'chemistry_snapshot': chemistry_snapshot, 'calibration_snapshot': calibration_snapshot, 'top_enriched_motifs': motif_records}), 'top_enriched_motifs': motif_records}

def build_task4_v3_rule_inputs(sample_summary: dict) -> dict:
    chemistry = sample_summary.get('chemistry_snapshot', {})
    return {'basic_fraction': float(chemistry.get('basic_fraction', 0.0)), 'glycine_proline_fraction': float(chemistry.get('glycine_proline_fraction', 0.0)), 'acidic_fraction': float(chemistry.get('acidic_fraction', 0.0)), 'hydrophobic_fraction': float(chemistry.get('hydrophobic_fraction', 0.0)), 'polar_fraction': float(chemistry.get('polar_fraction', 0.0)), 'regulatory_motif_fraction': float(chemistry.get('regulatory_motif_fraction', 0.0)), 'top1_share': float(chemistry.get('top1_share', sample_summary.get('repertoire_snapshot', {}).get('top1_share', 0.0)))}

def _rule_reason(feature_name: str, direction: str, value: float, sample_summary: dict) -> str:
    top_motifs = sample_summary.get('top_enriched_motifs', [])[:2]
    motif_text = ', '.join((str(item.get('display_motif', item.get('motif', ''))) for item in top_motifs if item.get('display_motif') or item.get('motif')))
    motif_suffix = f' Top motifs include {motif_text}.' if motif_text else ''
    if feature_name == 'basic_fraction':
        return f'{direction.capitalize()} basic-residue burden ({value:.3f}) shifts the rule-based score toward toxicity.{motif_suffix}'
    if feature_name == 'glycine_proline_fraction':
        return f'{direction.capitalize()} glycine/proline burden ({value:.3f}) supports a structurally reactive repertoire.{motif_suffix}'
    if feature_name == 'acidic_fraction':
        return f'{direction.capitalize()} acidic motif fraction ({value:.3f}) contributes to the rule-based toxicity axis.{motif_suffix}'
    if feature_name == 'hydrophobic_fraction':
        return f'{direction.capitalize()} hydrophobic motif burden ({value:.3f}) increases the repertoire risk score.{motif_suffix}'
    if feature_name == 'polar_fraction':
        return f'{direction.capitalize()} polar motif fraction ({value:.3f}) acts as a tolerance-leaning counterweight in the rule set.'
    if feature_name == 'regulatory_motif_fraction':
        return f'{direction.capitalize()} regulatory motif fraction ({value:.3f}) pushes the rule score away from toxicity.'
    return f'{direction.capitalize()} top1 clonotype share ({value:.3f}) is a major rule-based discriminator for this sample.'

def _build_task4_v3_rationale(sample_summary: dict, contribution_map: dict[str, float], prediction: str) -> list[str]:
    rule_inputs = build_task4_v3_rule_inputs(sample_summary)
    positive_features = sorted(contribution_map.items(), key=lambda item: item[1], reverse=True)
    negative_features = sorted(contribution_map.items(), key=lambda item: item[1])
    rationale = []
    for feature_name, contribution in positive_features[:2]:
        if contribution > 0:
            rationale.append(_rule_reason(feature_name, 'higher', rule_inputs[feature_name], sample_summary))
    for feature_name, contribution in negative_features[:2]:
        if contribution < 0:
            rationale.append(_rule_reason(feature_name, 'higher', rule_inputs[feature_name], sample_summary))
            break
    calibration = sample_summary.get('calibration_snapshot', {})
    if prediction == 'positive':
        rationale.append(f"Rule-first toxicity index is elevated with summary_toxicity_index {float(calibration.get('summary_toxicity_index', 0.0)):.3f} and autoreactivity pressure {float(calibration.get('autoreactivity_pressure_score', 0.0)):.3f}.")
    else:
        rationale.append(f"Rule-first toxicity index remains limited with regulatory_diversity_score {float(calibration.get('regulatory_diversity_score', 0.0)):.3f} and tolerance_support_score {float(calibration.get('tolerance_support_score', 0.0)):.3f}.")
    unique_rationale = []
    for item in rationale:
        if item not in unique_rationale:
            unique_rationale.append(item)
    return unique_rationale[:3]

def compute_task4_v3_rule_profiles(sample_summaries: list[dict]) -> dict[str, dict]:
    feature_table = pd.DataFrame([build_task4_v3_rule_inputs(sample_summary) for sample_summary in sample_summaries], index=[str(sample_summary['sample_id']) for sample_summary in sample_summaries])
    z_table = (feature_table - feature_table.mean(axis=0)) / feature_table.std(axis=0).replace(0, 1.0)
    z_table = z_table.fillna(0.0)
    raw_scores = pd.Series(0.0, index=feature_table.index, dtype=float)
    contribution_frames = {}
    for feature_name, weight in TASK4_V3_RULE_FEATURE_WEIGHTS.items():
        contribution_frames[feature_name] = z_table[feature_name] * float(weight)
        raw_scores = raw_scores + contribution_frames[feature_name]
    positive_threshold = float(raw_scores.quantile(TASK4_V3_RULE_POSITIVE_QUANTILE))
    centered_scores = raw_scores - positive_threshold
    toxicity_scores = 1.0 / (1.0 + (-centered_scores / TASK4_V3_RULE_LOGIT_SCALE).map(math.exp))
    toxicity_scores = toxicity_scores.clip(lower=0.05, upper=0.95)
    tolerance_scores = 1.0 - toxicity_scores
    rule_profiles = {}
    for sample_summary in sample_summaries:
        sample_id = str(sample_summary['sample_id'])
        raw_score = float(raw_scores.loc[sample_id])
        toxicity_score = float(toxicity_scores.loc[sample_id])
        tolerance_score = float(tolerance_scores.loc[sample_id])
        prediction = 'positive' if raw_score >= positive_threshold else 'negative'
        dominant_side = 'toxicity' if toxicity_score >= 0.55 else 'tolerance' if toxicity_score <= 0.45 else 'mixed'
        contribution_map = {feature_name: float(series.loc[sample_id]) for feature_name, series in contribution_frames.items()}
        rationale = _build_task4_v3_rationale(sample_summary, contribution_map=contribution_map, prediction=prediction)
        rule_profiles[sample_id] = {'rule_raw_score': round(raw_score, 4), 'rule_threshold': round(positive_threshold, 4), 'rule_centered_score': round(float(centered_scores.loc[sample_id]), 4), 'toxicity_support_score': round(toxicity_score, 4), 'tolerance_support_score': round(tolerance_score, 4), 'prediction': prediction, 'dominant_side': dominant_side, 'contributions': {feature_name: round(value, 4) for feature_name, value in contribution_map.items()}, 'rationale': rationale}
    return rule_profiles
