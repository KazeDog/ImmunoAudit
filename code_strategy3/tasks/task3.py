from submission_paths import path as _submission_path
import csv
from functools import lru_cache
from pathlib import Path
import pandas as pd
WORKSPACE_DIR = Path(str(_submission_path('project', '')))
RAW_TASK3_DIR = Path(str(_submission_path('data', '3-')))
DEFAULT_TASK3_DATASET = 'task3_Gide_PRJEB23709'
TASK3_DATASET_SOURCES = {'task3_Gide_PRJEB23709': {'kind': 'standard', 'expression_path': RAW_TASK3_DIR / 'Gide_PRJEB23709' / 'Melanoma-PRJEB23709.Response.Expression Data.tsv'}, 'task3_Kim_PRJEB25780': {'kind': 'standard', 'expression_path': RAW_TASK3_DIR / 'Kim_PRJEB25780' / 'STAD-PRJEB25780.Response.Expression Data.tsv'}, 'task3_Liu_phs000452': {'kind': 'standard', 'expression_path': RAW_TASK3_DIR / 'Liu_phs000452' / 'Melanoma-phs000452.Response.Expression Data.tsv'}, 'task3_Riaz_GSE91061': {'kind': 'standard', 'expression_path': RAW_TASK3_DIR / 'Riaz_GSE91061' / 'Melanoma-GSE91061.Response.Expression Data.tsv'}, 'task3_IMvigor210': {'kind': 'imvigor210', 'feature_path': RAW_TASK3_DIR / 'IMvigor210' / 'fData_IMvigor210.csv'}}
BIOLOGICAL_THEMES = {'interferon_response': {'STAT1', 'STAT2', 'IRF1', 'IRF7', 'IFI6', 'IFI27', 'IFI35', 'IFI44', 'IFI44L', 'IFIH1', 'IFIT1', 'IFIT2', 'IFIT3', 'IFITM1', 'ISG15', 'MX1', 'MX2', 'OAS1', 'OAS2', 'OAS3', 'RSAD2'}, 'antigen_presentation': {'B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DRA', 'HLA-DRB1', 'HLA-E', 'PSMB8', 'PSMB9', 'TAP1', 'TAP2'}, 'cytotoxic_effector': {'CD8A', 'CD8B', 'EOMES', 'GNLY', 'GZMA', 'GZMB', 'GZMH', 'IFNG', 'NKG7', 'PRF1'}, 't_cell_activation': {'CD2', 'CD3D', 'CD3E', 'CD27', 'CD28', 'CXCL9', 'CXCL10', 'CXCL11', 'ICOS', 'LAG3', 'PDCD1', 'TIGIT'}, 'b_cell_signature': {'BANK1', 'BLK', 'CD19', 'CD22', 'CD79A', 'CD79B', 'MS4A1', 'MZB1'}, 'myeloid_inflammation': {'CCR1', 'CCR2', 'CD14', 'CXCL8', 'FCGR3A', 'IL1B', 'LILRB1', 'S100A8', 'S100A9', 'TYMP'}, 'stroma_emt': {'ACTA2', 'COL1A1', 'COL1A2', 'COL3A1', 'COL5A1', 'FN1', 'MMP2', 'TAGLN', 'VIM'}, 'cell_cycle': {'AURKA', 'BIRC5', 'CCNB1', 'CDC20', 'CDC6', 'CDK1', 'CENPF', 'MKI67', 'PCNA', 'TOP2A', 'UBE2C'}}
PROGRAM_GENESETS = {'interferon_gamma_program': {'B2M', 'CXCL9', 'CXCL10', 'CXCL11', 'GBP1', 'HLA-A', 'HLA-B', 'HLA-C', 'IDO1', 'IFI27', 'IFIT1', 'IFIT2', 'IFIT3', 'IRF1', 'ISG15', 'PSMB8', 'PSMB9', 'STAT1', 'TAP1'}, 'antigen_presentation_program': {'B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DRA', 'HLA-DRB1', 'PSMB8', 'PSMB9', 'TAP1', 'TAP2'}, 'cytotoxic_lymphocyte_program': {'CD2', 'CD3D', 'CD3E', 'CD8A', 'CD8B', 'EOMES', 'GNLY', 'GZMA', 'GZMB', 'GZMK', 'IFNG', 'NKG7', 'PRF1'}, 't_cell_exhaustion_program': {'CTLA4', 'ENTPD1', 'HAVCR2', 'LAG3', 'PDCD1', 'TIGIT', 'TOX'}, 'chemokine_inflamed_program': {'CCL4', 'CCL5', 'CXCL9', 'CXCL10', 'CXCL11', 'CXCR3', 'STAT1'}, 'b_cell_plasma_program': {'BANK1', 'BLK', 'CD19', 'CD22', 'CD79A', 'CD79B', 'MZB1', 'MS4A1', 'SDC1'}, 'myeloid_inflammation_program': {'CCR1', 'CCR2', 'CD14', 'CXCL8', 'FCGR3A', 'IL1B', 'LILRB1', 'S100A8', 'S100A9', 'TYMP'}, 'tgfb_stromal_program': {'ACTA2', 'COL1A1', 'COL1A2', 'COL3A1', 'COL5A1', 'FN1', 'FAP', 'MMP2', 'TAGLN', 'TGFB1', 'VIM'}, 'epithelial_keratin_program': {'EPCAM', 'KRT1', 'KRT5', 'KRT8', 'KRT10', 'KRT14', 'KRT18', 'KRT19'}, 'cell_cycle_program': {'AURKA', 'BIRC5', 'CCNB1', 'CDC20', 'CDC6', 'CDK1', 'CENPF', 'MKI67', 'PCNA', 'TOP2A', 'UBE2C'}}

def load_task3_processed_dataset(dataset_name: str=DEFAULT_TASK3_DATASET) -> pd.DataFrame:
    data_path = WORKSPACE_DIR / 'processed_data_strategy1' / f'{dataset_name}_X_final.csv'
    if not data_path.exists():
        raise FileNotFoundError(f'Task 3 processed dataset not found: {data_path}')
    return pd.read_csv(data_path, index_col=0)

def build_task3_sample_summary(df: pd.DataFrame, sample_id: str | None=None, top_k_features: int=12) -> dict:
    if df.empty:
        raise ValueError('Task 3 dataframe is empty.')
    if sample_id is None:
        sample_id = str(df.index[0])
    dataset_name = infer_dataset_name(df)
    row = df.loc[sample_id].astype(float)
    ranked = row.sort_values(ascending=False)
    feature_annotations = load_task3_feature_annotations(dataset_name)
    top_ranked = [build_feature_record(feature_id=str(feature), value=float(value), feature_annotations=feature_annotations) for feature, value in ranked.head(top_k_features).items()]
    gene_profile = build_gene_expression_profile(row, feature_annotations)
    return {'task_name': 'task3', 'dataset_name': dataset_name, 'sample_id': str(sample_id), 'n_features': int(df.shape[1]), 'summary_type': 'feature_magnitude_snapshot', 'feature_ranking_rule': 'features sorted by observed value in descending order', 'top_ranked_features': top_ranked, 'biological_theme_snapshot': summarize_biological_themes(top_ranked), 'program_activity_snapshot': summarize_program_activity(gene_profile)}

def infer_dataset_name(df: pd.DataFrame) -> str:
    name = getattr(df, 'attrs', {}).get('dataset_name')
    if isinstance(name, str) and name:
        return name
    return DEFAULT_TASK3_DATASET

@lru_cache(maxsize=None)
def load_task3_feature_annotations(dataset_name: str) -> dict[str, dict[str, str | int | None]]:
    config = TASK3_DATASET_SOURCES.get(dataset_name)
    if config is None:
        return {}
    if config['kind'] == 'standard':
        gene_symbols = load_standard_task3_gene_symbols(config['expression_path'])
        entrez_ids = [None] * len(gene_symbols)
    elif config['kind'] == 'imvigor210':
        feature_df = pd.read_csv(config['feature_path'])
        symbol_col = 'symbol' if 'symbol' in feature_df.columns else 'Symbol'
        entrez_col = 'entrez_id' if 'entrez_id' in feature_df.columns else None
        gene_symbols = feature_df[symbol_col].astype(str).tolist()
        entrez_ids = feature_df[entrez_col].tolist() if entrez_col else [None] * len(gene_symbols)
    else:
        return {}
    annotations = {}
    for idx, gene_symbol in enumerate(gene_symbols, start=1):
        feature_id = str(idx)
        normalized_gene = normalize_gene_symbol(gene_symbol)
        annotations[feature_id] = {'feature_id': feature_id, 'gene_symbol': gene_symbol, 'normalized_gene_symbol': normalized_gene, 'entrez_id': None if entrez_ids[idx - 1] is None or pd.isna(entrez_ids[idx - 1]) else int(entrez_ids[idx - 1])}
    return annotations

def build_feature_record(feature_id: str, value: float, feature_annotations: dict[str, dict[str, str | int | None]]) -> dict:
    annotation = feature_annotations.get(feature_id, {})
    if not annotation:
        normalized_feature_id = normalize_feature_id(feature_id)
        annotation = feature_annotations.get(normalized_feature_id, {})
    gene_symbol = annotation.get('gene_symbol')
    normalized_gene_symbol = annotation.get('normalized_gene_symbol')
    themes = infer_feature_themes(str(normalized_gene_symbol or ''))
    record = {'feature_id': feature_id, 'gene_symbol': gene_symbol if gene_symbol else None, 'value': round(value, 4)}
    entrez_id = annotation.get('entrez_id')
    if entrez_id is not None:
        record['entrez_id'] = int(entrez_id)
    if themes:
        record['biological_tags'] = themes
    return record

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
        summaries.append({'theme': theme, 'matched_feature_count': count, 'example_genes': sorted(set(theme_genes.get(theme, [])))[:5]})
    return summaries

def build_gene_expression_profile(row: pd.Series, feature_annotations: dict[str, dict[str, str | int | None]]) -> pd.Series:
    gene_values: dict[str, float] = {}
    for feature_id, value in row.items():
        annotation = feature_annotations.get(str(feature_id), {})
        if not annotation:
            annotation = feature_annotations.get(normalize_feature_id(str(feature_id)), {})
        gene_symbol = annotation.get('normalized_gene_symbol')
        if not gene_symbol:
            continue
        numeric_value = float(value)
        if gene_symbol in gene_values:
            gene_values[gene_symbol] = max(gene_values[gene_symbol], numeric_value)
        else:
            gene_values[gene_symbol] = numeric_value
    if not gene_values:
        return pd.Series(dtype=float)
    profile = pd.Series(gene_values, dtype=float)
    return profile.sort_values(ascending=False)

def summarize_program_activity(gene_profile: pd.Series, top_n: int=6) -> list[dict]:
    if gene_profile.empty:
        return []
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

def infer_feature_themes(gene_symbol: str) -> list[str]:
    matches = []
    for theme, genes in BIOLOGICAL_THEMES.items():
        if gene_symbol in genes:
            matches.append(theme)
    prefix_themes = [('SNORA', 'small_nucleolar_rna'), ('SNORD', 'small_nucleolar_rna'), ('SCARNA', 'small_cajal_body_rna'), ('MT-', 'mitochondrial_transcript'), ('KRT', 'epithelial_keratinization'), ('ACT', 'cytoskeleton_actin'), ('RPL', 'ribosomal_large_subunit'), ('RPS', 'ribosomal_small_subunit')]
    for prefix, theme in prefix_themes:
        if gene_symbol.startswith(prefix) and theme not in matches:
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

def load_standard_task3_gene_symbols(expression_path: Path) -> list[str]:
    gene_symbols = []
    with expression_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                gene_symbols.append(str(row[1]))
    return gene_symbols

def normalize_feature_id(feature_id: str) -> str:
    feature_id = str(feature_id)
    if feature_id.startswith('gene_'):
        return feature_id.split('gene_', 1)[1]
    return feature_id
