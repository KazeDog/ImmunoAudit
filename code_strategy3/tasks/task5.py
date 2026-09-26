from submission_paths import path as _submission_path
from pathlib import Path
import pandas as pd
from .task3 import build_feature_record, build_gene_expression_profile, load_task3_feature_annotations, summarize_biological_themes, summarize_program_activity
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
DEFAULT_TASK5_DATASET = 'task5_Gide_PRJEB23709'
TASK5_DATASET_SOURCES = {'task5_Gide_PRJEB23709': {'data_path': WORKSPACE_DIR / 'processed_data_strategy1' / 'task5_Gide_PRJEB23709_X_final.csv', 'label_path': WORKSPACE_DIR / 'processed_data_strategy1' / 'task5_Gide_PRJEB23709_y_final.csv', 'annotation_dataset': 'task3_Gide_PRJEB23709'}, 'task5_Liu_phs000452': {'data_path': WORKSPACE_DIR / 'processed_data_strategy1' / 'task5_Liu_phs000452_X_final.csv', 'label_path': WORKSPACE_DIR / 'processed_data_strategy1' / 'task5_Liu_phs000452_y_final.csv', 'annotation_dataset': 'task3_Liu_phs000452'}, 'task5_Riaz_GSE91061': {'data_path': WORKSPACE_DIR / 'processed_data_strategy1' / 'task5_Riaz_GSE91061_X_final.csv', 'label_path': WORKSPACE_DIR / 'processed_data_strategy1' / 'task5_Riaz_GSE91061_y_final.csv', 'annotation_dataset': 'task3_Riaz_GSE91061'}}
PROTECTIVE_PROGRAMS = {'interferon_gamma_program', 'antigen_presentation_program', 'cytotoxic_lymphocyte_program', 'chemokine_inflamed_program'}
ADVERSE_PROGRAMS = {'myeloid_inflammation_program', 'tgfb_stromal_program', 't_cell_exhaustion_program', 'cell_cycle_program'}

def load_task5_processed_dataset(dataset_name: str=DEFAULT_TASK5_DATASET) -> pd.DataFrame:
    config = TASK5_DATASET_SOURCES.get(dataset_name)
    if config is None:
        raise KeyError(f'Unknown Task 5 dataset: {dataset_name}')
    data_path = config['data_path']
    if not data_path.exists():
        raise FileNotFoundError(f'Task 5 processed dataset not found: {data_path}')
    df = pd.read_csv(data_path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df.attrs['dataset_name'] = dataset_name
    return df

def load_task5_labels(dataset_name: str=DEFAULT_TASK5_DATASET) -> pd.DataFrame:
    config = TASK5_DATASET_SOURCES.get(dataset_name)
    if config is None:
        raise KeyError(f'Unknown Task 5 dataset: {dataset_name}')
    label_path = config['label_path']
    if not label_path.exists():
        raise FileNotFoundError(f'Task 5 labels not found: {label_path}')
    labels = pd.read_csv(label_path, index_col=0)
    labels.index = labels.index.astype(str)
    labels['duration'] = labels['duration'].astype(float)
    labels['event'] = labels['event'].astype(int)
    return labels

def build_task5_sample_summary(df: pd.DataFrame, sample_id: str | None=None, top_k_features: int=12) -> dict:
    if df.empty:
        raise ValueError('Task 5 dataframe is empty.')
    if sample_id is None:
        sample_id = str(df.index[0])
    dataset_name = infer_task5_dataset_name(df)
    annotation_dataset = TASK5_DATASET_SOURCES[dataset_name]['annotation_dataset']
    row = df.loc[sample_id].astype(float)
    ranked = row.sort_values(ascending=False)
    feature_annotations = load_task3_feature_annotations(annotation_dataset)
    top_ranked = [build_feature_record(feature_id=str(feature), value=float(value), feature_annotations=feature_annotations) for feature, value in ranked.head(top_k_features).items()]
    gene_profile = build_gene_expression_profile(row, feature_annotations)
    program_activity = summarize_program_activity(gene_profile, top_n=8)
    ecology_snapshot = build_task5_ecology_snapshot(program_activity)
    calibration_snapshot = build_task5_calibration_snapshot(ecology_snapshot)
    return {'task_name': 'task5', 'dataset_name': dataset_name, 'sample_id': str(sample_id), 'n_features': int(df.shape[1]), 'summary_type': 'bulk_expression_survival_snapshot', 'feature_ranking_rule': 'features sorted by observed value in descending order', 'top_ranked_features': top_ranked, 'biological_theme_snapshot': summarize_biological_themes(top_ranked), 'program_activity_snapshot': program_activity, 'ecology_snapshot': ecology_snapshot, 'calibration_snapshot': calibration_snapshot}

def infer_task5_dataset_name(df: pd.DataFrame) -> str:
    name = getattr(df, 'attrs', {}).get('dataset_name')
    if isinstance(name, str) and name:
        return name
    return DEFAULT_TASK5_DATASET

def build_task5_ecology_snapshot(program_activity: list[dict]) -> dict:
    program_scores = {item['program']: float(item['relative_activity_score']) for item in program_activity}
    inflamed_support = average_program_score(program_scores, ['interferon_gamma_program', 'antigen_presentation_program', 'cytotoxic_lymphocyte_program', 'chemokine_inflamed_program'])
    suppressive_pressure = average_program_score(program_scores, ['myeloid_inflammation_program', 'tgfb_stromal_program', 't_cell_exhaustion_program'])
    proliferative_pressure = float(program_scores.get('cell_cycle_program', 0.0))
    antigen_presentation_score = float(program_scores.get('antigen_presentation_program', 0.0))
    interferon_score = float(program_scores.get('interferon_gamma_program', 0.0))
    cytotoxic_score = float(program_scores.get('cytotoxic_lymphocyte_program', 0.0))
    chemokine_score = float(program_scores.get('chemokine_inflamed_program', 0.0))
    exhaustion_score = float(program_scores.get('t_cell_exhaustion_program', 0.0))
    myeloid_score = float(program_scores.get('myeloid_inflammation_program', 0.0))
    stromal_score = float(program_scores.get('tgfb_stromal_program', 0.0))
    protective_signal_count = sum((1 for item in program_activity if item['program'] in PROTECTIVE_PROGRAMS and float(item['relative_activity_score']) >= 0.8))
    adverse_signal_count = sum((1 for item in program_activity if item['program'] in ADVERSE_PROGRAMS and float(item['relative_activity_score']) >= 0.78))
    immune_favorability_score = 0.36 * antigen_presentation_score + 0.28 * cytotoxic_score + 0.2 * interferon_score + 0.16 * chemokine_score
    barrier_pressure_score = 0.44 * stromal_score + 0.34 * myeloid_score + 0.22 * exhaustion_score
    adaptive_reserve_score = 0.5 * antigen_presentation_score + 0.3 * cytotoxic_score + 0.2 * chemokine_score - 0.18 * exhaustion_score
    proliferative_escape_score = 0.7 * proliferative_pressure + 0.3 * max(0.0, barrier_pressure_score - immune_favorability_score)
    favorability_barrier_gap = immune_favorability_score - barrier_pressure_score
    reserve_escape_gap = adaptive_reserve_score - proliferative_escape_score
    immune_balance = inflamed_support - suppressive_pressure
    immune_exclusion_gap = suppressive_pressure - inflamed_support
    net_risk = 0.34 * barrier_pressure_score + 0.26 * proliferative_pressure + 0.14 * exhaustion_score + 0.08 * max(0.0, proliferative_escape_score - 0.62) - 0.24 * adaptive_reserve_score - 0.16 * immune_favorability_score
    ranked_programs = sorted(({'program': item['program'], 'relative_activity_score': round(float(item['relative_activity_score']), 4)} for item in program_activity), key=lambda item: (-item['relative_activity_score'], item['program']))
    dominant_protective = [item for item in ranked_programs if item['program'] in PROTECTIVE_PROGRAMS][:4]
    dominant_adverse = [item for item in ranked_programs if item['program'] in ADVERSE_PROGRAMS][:4]
    return {'inflamed_support_score': round(inflamed_support, 4), 'inflamed_support_level': describe_signal_level(inflamed_support), 'immune_activation_score': round(inflamed_support, 4), 'immune_activation_level': describe_signal_level(inflamed_support), 'exclusion_pressure_score': round(suppressive_pressure, 4), 'exclusion_pressure_level': describe_signal_level(suppressive_pressure), 'suppressive_pressure_score': round(suppressive_pressure, 4), 'suppressive_pressure_level': describe_signal_level(suppressive_pressure), 'proliferative_pressure_score': round(proliferative_pressure, 4), 'proliferative_pressure_level': describe_signal_level(proliferative_pressure), 'antigen_presentation_score': round(antigen_presentation_score, 4), 'interferon_gamma_score': round(interferon_score, 4), 'cytotoxic_score': round(cytotoxic_score, 4), 'chemokine_inflamed_score': round(chemokine_score, 4), 't_cell_exhaustion_score': round(exhaustion_score, 4), 'myeloid_inflammation_score': round(myeloid_score, 4), 'tgfb_stromal_score': round(stromal_score, 4), 'immune_favorability_score': round(immune_favorability_score, 4), 'barrier_pressure_score': round(barrier_pressure_score, 4), 'adaptive_reserve_score': round(adaptive_reserve_score, 4), 'proliferative_escape_score': round(proliferative_escape_score, 4), 'favorability_barrier_gap': round(favorability_barrier_gap, 4), 'reserve_escape_gap': round(reserve_escape_gap, 4), 'immune_balance_score': round(immune_balance, 4), 'immune_exclusion_gap': round(immune_exclusion_gap, 4), 'net_risk_score': round(net_risk, 4), 'net_risk_direction': describe_net_risk(net_risk), 'dominant_protective_axis_count': sum((1 for program in PROTECTIVE_PROGRAMS if float(program_scores.get(program, 0.0)) >= 0.72)), 'dominant_adverse_axis_count': sum((1 for program in ADVERSE_PROGRAMS if float(program_scores.get(program, 0.0)) >= 0.72)), 'strong_protective_signal_count': protective_signal_count, 'strong_adverse_signal_count': adverse_signal_count, 'dominant_protective_programs': dominant_protective, 'dominant_adverse_programs': dominant_adverse}

def build_task5_calibration_snapshot(ecology_snapshot: dict) -> dict:
    inflamed = float(ecology_snapshot['immune_activation_score'])
    suppressive = float(ecology_snapshot['exclusion_pressure_score'])
    proliferative = float(ecology_snapshot['proliferative_pressure_score'])
    net_risk = float(ecology_snapshot['net_risk_score'])
    immune_balance = float(ecology_snapshot['immune_balance_score'])
    immune_exclusion_gap = float(ecology_snapshot['immune_exclusion_gap'])
    antigen_presentation = float(ecology_snapshot['antigen_presentation_score'])
    adaptive_reserve = float(ecology_snapshot['adaptive_reserve_score'])
    barrier_pressure = float(ecology_snapshot['barrier_pressure_score'])
    immune_favorability = float(ecology_snapshot['immune_favorability_score'])
    proliferative_escape = float(ecology_snapshot['proliferative_escape_score'])
    favorability_gap = float(ecology_snapshot['favorability_barrier_gap'])
    reserve_escape_gap = float(ecology_snapshot['reserve_escape_gap'])
    stromal_score = float(ecology_snapshot['tgfb_stromal_score'])
    myeloid_score = float(ecology_snapshot['myeloid_inflammation_score'])
    exhaustion_score = float(ecology_snapshot['t_cell_exhaustion_score'])
    strong_protective = int(ecology_snapshot['strong_protective_signal_count'])
    strong_adverse = int(ecology_snapshot['strong_adverse_signal_count'])
    dominant_adverse_axes = int(ecology_snapshot['dominant_adverse_axis_count'])
    excluded_inflamed_state = bool(immune_favorability >= 0.72 and barrier_pressure >= 0.74 and (favorability_gap <= 0.04) and (adaptive_reserve >= 0.66))
    stromal_dominant_state = bool(stromal_score >= 0.8 and barrier_pressure >= 0.74 and (favorability_gap < 0.0) and (adaptive_reserve < 0.74))
    myeloid_exhausted_state = bool(myeloid_score >= 0.74 and exhaustion_score >= 0.68 and (adaptive_reserve < 0.72) and (immune_balance <= 0.1))
    proliferative_escape_state = bool(proliferative >= 0.72 and proliferative_escape >= 0.68 and (reserve_escape_gap <= 0.06) and (immune_balance <= 0.08))
    favorable_immune_ready = bool(immune_favorability >= 0.74 and adaptive_reserve >= 0.74 and (antigen_presentation >= 0.76) and (favorability_gap >= -0.02) and (reserve_escape_gap >= 0.12) and (proliferative < 0.62) and (dominant_adverse_axes <= 1) and (not excluded_inflamed_state))
    low_risk_blocked = bool(excluded_inflamed_state or stromal_dominant_state or proliferative_escape_state or (strong_adverse >= 2 and suppressive >= 0.58 and (favorability_gap < 0.02)))
    low_risk_ready = bool(favorable_immune_ready and inflamed >= 0.72 and (suppressive < 0.6) and (proliferative < 0.58) and (strong_adverse <= 1) and (net_risk <= 0.0) and (not low_risk_blocked))
    high_risk_ready = bool(stromal_dominant_state and proliferative >= 0.66 and (net_risk >= 0.12) or (myeloid_exhausted_state and net_risk >= 0.1 and (adaptive_reserve < 0.68)) or (proliferative_escape_state and barrier_pressure >= 0.62 and (net_risk >= 0.12)) or (dominant_adverse_axes >= 3 and barrier_pressure >= 0.68 and (favorability_gap < 0.0) and (net_risk >= 0.14)))
    mixed_profile = bool((not low_risk_ready and (not high_risk_ready)) and (excluded_inflamed_state and adaptive_reserve >= 0.7 and (net_risk <= 0.18) or (-0.02 <= favorability_gap <= 0.08 and 0.02 <= net_risk <= 0.14) or (strong_protective >= 2 and strong_adverse >= 2 and (0.0 <= net_risk <= 0.18)) or (0.08 <= reserve_escape_gap <= 0.2 and 0.0 <= net_risk <= 0.12)))
    return {'inflamed_support_score': round(inflamed, 4), 'immune_activation_score': round(inflamed, 4), 'exclusion_pressure_score': round(suppressive, 4), 'suppressive_pressure_score': round(suppressive, 4), 'proliferative_pressure_score': round(proliferative, 4), 'immune_favorability_score': round(immune_favorability, 4), 'barrier_pressure_score': round(barrier_pressure, 4), 'adaptive_reserve_score': round(adaptive_reserve, 4), 'proliferative_escape_score': round(proliferative_escape, 4), 'favorability_barrier_gap': round(favorability_gap, 4), 'reserve_escape_gap': round(reserve_escape_gap, 4), 'immune_balance_score': round(immune_balance, 4), 'immune_exclusion_gap': round(immune_exclusion_gap, 4), 'calibrated_net_risk': round(net_risk, 4), 'near_zero_net_risk': bool(-0.03 <= net_risk <= 0.08), 'immune_excluded_state': excluded_inflamed_state, 'excluded_inflamed_state': excluded_inflamed_state, 'stromal_dominant_state': stromal_dominant_state, 'myeloid_exhausted_state': myeloid_exhausted_state, 'proliferative_escape_state': proliferative_escape_state, 'favorable_immune_ready': favorable_immune_ready, 'low_risk_blocked': low_risk_blocked, 'low_risk_ready': low_risk_ready, 'high_risk_ready': high_risk_ready, 'mixed_profile': mixed_profile, 'dominant_adverse_axis_count': dominant_adverse_axes, 'strong_adverse_signal_count': strong_adverse, 'strong_protective_signal_count': strong_protective}

def average_program_score(program_scores: dict[str, float], programs: list[str]) -> float:
    if not programs:
        return 0.0
    return float(sum((float(program_scores.get(program, 0.0)) for program in programs)) / len(programs))

def describe_signal_level(score: float) -> str:
    if score >= 0.8:
        return 'very_high'
    if score >= 0.68:
        return 'high'
    if score >= 0.55:
        return 'moderate'
    if score >= 0.4:
        return 'mild'
    return 'low'

def describe_net_risk(score: float) -> str:
    if score >= 0.18:
        return 'strongly_adverse'
    if score >= 0.06:
        return 'adverse_leaning'
    if score <= -0.12:
        return 'strongly_favorable'
    if score <= -0.03:
        return 'favorable_leaning'
    return 'balanced'
