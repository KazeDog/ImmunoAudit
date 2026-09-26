"""Portable-path and corrected-input checks using synthetic fixtures only."""
import importlib.util
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from submission_paths import PACKAGE_ROOT


def load(name, relative):
    spec=importlib.util.spec_from_file_location(name,PACKAGE_ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def test_current_bcr_imports_without_archives():
    core=load('run_repair_submission_test','analysis/bcr/run_repair.py')
    assert core.p1.__file__.endswith('analysis/bcr/bcr_inputs.py')
    assert core.REPEAT_KEYS[1]=='strategy1|BCR_3mer|XGBoost'


def test_corrected_case_source():
    mod=load('case_submission_test','analysis/disagreement/scripts/run_disagreement_case_analysis.py')
    assert mod.input_paths(Path('work')).task4_repeated_samples==Path('work/analysis/bcr/bcr_case_input.csv')


def test_embedding_row_alignment(tmp_path):
    from task4_grouped_core import load_saved_embeddings
    np.save(tmp_path/'emb.npy',np.array([[10,20],[30,40]]))
    pd.DataFrame({'sample_id':['sample_b','sample_a']}).to_csv(tmp_path/'order.csv',index=False)
    result=load_saved_embeddings(tmp_path/'emb.npy',tmp_path/'order.csv',['sample_a','sample_b'])
    np.testing.assert_array_equal(result,[[30,40],[10,20]])


def test_cli_paths_propagate_to_analysis_and_llm_config(tmp_path):
    result=subprocess.run([sys.executable,str(PACKAGE_ROOT/'run.py'),
        '--project-root',str(tmp_path),'--data-root',str(tmp_path/'raw'),
        '--models-root',str(tmp_path/'weights'),'--script','tests/path_probe.py'],
        check=True,capture_output=True,text=True)
    value=json.loads(result.stdout)
    assert value['project']==str(tmp_path)
    assert value['llm_workspace']==str(tmp_path)
    assert value['data']==str(tmp_path/'raw')
    assert value['bcr_clinical']==str(tmp_path/'raw/4-/task5_clinical_labels.csv')
    assert value['models']==str(tmp_path/'weights')


def test_bcr_uses_corrected_matrix_without_old_feature_file(tmp_path, monkeypatch):
    import bcr_inputs as mod
    n=256
    ids=np.arange(n)%113
    mapping=pd.DataFrame({'sample_id':[f'synthetic_{i}' for i in range(n)],
                         'patient_id':ids,'collection_time':['BL']*113+['POST']*143,
                         'toxicity_corrected_grade_parser':np.zeros(n,int)})
    folds=pd.DataFrame({'sample_id':mapping.sample_id,'fold':ids%5+1})
    reads=[]
    def read(path,*args,**kwargs):
        reads.append(path)
        if path==mod.MAPPING_PATH:return mapping.copy()
        if path==mod.FOLD_PATH:return folds.copy()
        if path==mod.CLINICAL_PATH:return pd.DataFrame()
        raise AssertionError('Unexpected input dependency')
    monkeypatch.setattr(mod.pd,'read_csv',read)
    monkeypatch.setattr(mod,'load_saved_embeddings',lambda *a:np.zeros((n,2)))
    monkeypatch.setattr(mod,'derive_patient_label_audit',lambda _:pd.DataFrame({
        'patient_id':np.arange(113),'corrected_grade_label':np.zeros(113,int),
        'post_ici_incident_label':np.zeros(113,int)}))
    corrected=np.full((n,8000),1/8000)
    np.save(tmp_path/'corrected.npy',corrected)
    _,matrices,_=mod.load_inputs(tmp_path/'corrected.npy')
    np.testing.assert_array_equal(matrices['BCR_3mer'],corrected)
    assert len(reads)==3


def test_current_survival_ranks_are_fold_affine_invariant():
    mod=load('rank_submission_test','analysis/rank_sensitivity/analyse_rank_scales.py')
    rng=np.random.default_rng(11)
    blocks=[rng.normal(size=(15,3)) for _ in range(5)]
    before=mod.summarize(blocks)
    after=mod.summarize([b*(i+1)+100*i for i,b in enumerate(blocks)])
    np.testing.assert_allclose(before[0],after[0])
    np.testing.assert_allclose(before[3],after[3])
