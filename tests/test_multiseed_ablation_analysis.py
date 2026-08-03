import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.analyze_multiseed_ablation import (ALL_METRICS, PRIMARY_METRICS, REQUIRED_METRICS, SEEDS, build_parser, canonical_config, load_run, normalize_predictions, paired_statistics, prediction_check, range_error_bars, run_analysis, summary_tables, validate_runs)


def config(seed, mode, **overrides):
    value={"experiment":f"A4_{mode}_{seed}","input_mode":mode,"seed":seed,"backbone":"densenet121","pretrained":True,"preprocessing":"imagenet","input_size":224,"augmentation":"historical","rotation_degrees":7,"horizontal_flip":True,"loss":"weighted_bce","optimizer":"adamw","learning_rate":1e-5,"backbone_learning_rate":1e-5,"head_learning_rate":1e-4,"weight_decay":1e-5,"scheduler":"plateau","batch_size":8,"epochs":20,"early_stopping_patience":5}; value.update(overrides); return value

def make_run(root, seed, mode, values=None, predictions=None, **changes):
    run=root/f"{mode}_{seed}"; run.mkdir(); metrics={m:0.7+i*.001 for i,m in enumerate(REQUIRED_METRICS)}; metrics.update(values or {})
    (run/"validation_metrics.json").write_text(json.dumps(metrics)); (run/"config.json").write_text(json.dumps(config(seed,mode,**changes))); (run/"run_summary.json").write_text(json.dumps({"best_epoch":4}))
    if predictions is not None: predictions.to_csv(run/"validation_predictions.csv",index=False)
    return run

def paths(tmp_path):
    return {(model,seed):make_run(tmp_path,seed,mode, {"auroc":.7+seed/100000}) for model,mode in [("original","original"),("hard_masked","hard_masked")] for seed in SEEDS}

def test_actual_metric_schema_and_seed(tmp_path):
    run=make_run(tmp_path,42,"original"); loaded=load_run(run,"original",42); assert loaded["metrics"]["auroc"] == .7 and loaded["seed"]==42

def test_missing_metric_rejected(tmp_path):
    run=make_run(tmp_path,42,"original"); (run/"validation_metrics.json").write_text("{}")
    with pytest.raises(ValueError,match="missing required validation metrics"): load_run(run,"original",42)

def test_legacy_a4_and_comparability(tmp_path):
    legacy=config(42,"hard_masked"); legacy.pop("input_mode"); assert canonical_config(legacy,tmp_path/"config.json")["input_mode"]=="hard_masked"
    runs=[load_run(make_run(tmp_path,s,m), model,s) for model,m in [("original","original"),("hard_masked","hard_masked")] for s in SEEDS]; validate_runs(runs)
    runs[-1]["config"]["batch_size"]=99
    with pytest.raises(ValueError,match="batch_size"): validate_runs(runs)

def test_summary_sd_and_student_t_ci_and_effect():
    frame=pd.DataFrame([{"model":"original",**{m:float(i) for m in ALL_METRICS}} for i in [1,2,3]])
    frame=pd.concat([frame,pd.DataFrame([{"model":"hard_masked",**{m:float(i) for m in ALL_METRICS}} for i in [2,3,4]])])
    summary,_=summary_tables(frame); assert summary.iloc[0].sample_sd==pytest.approx(1)
    result=paired_statistics(np.array([1.,2.,3.])); assert result["ci_lower"]==pytest.approx(2-4.3026527299/np.sqrt(3)); assert result["cohens_dz"]==pytest.approx(2)
    assert result["paired_t_p_value"] is not None and result["wilcoxon_p_value"] is not None

def test_paired_zero_variance_safe():
    value=paired_statistics(np.array([.1,.1,.1])); assert value["paired_t_p_value"] is None and value["cohens_dz"] is None
    assert value["hedges_gz"] is None and value["paired_t_status"]=="not_evaluable_zero_variance"
    assert value["ci_lower"]==pytest.approx(.1) and value["ci_upper"]==pytest.approx(.1) and value["ci_status"]=="degenerate_zero_variance"

def test_all_zero_and_near_zero_variance_are_not_evaluable():
    for differences in (np.zeros(3), np.array([.1, .1 + 1e-14, .1 - 1e-14])):
        result=paired_statistics(differences)
        assert result["paired_t_p_value"] is None and result["cohens_dz"] is None
        assert result["paired_t_status"]=="not_evaluable_zero_variance"

def test_range_error_bars_clip_floating_point_roundoff():
    means=np.array([.3, .7], dtype=np.float64); lower=np.array([.1, .7 + 1e-16]); upper=np.array([.5, .9])
    yerr=range_error_bars(means,lower,upper)
    assert yerr.shape==(2,2) and np.isfinite(yerr).all() and (yerr>=0).all()
    assert yerr[0,1] == 0.0

def test_prediction_schemas_alignment_and_missing(tmp_path):
    legacy=pd.DataFrame({"patient_id":[1,2],"study_id":[3,4],"image_path":["a","b"],"label":[1,0],"probability":[.6,.2],"split":["validation"]*2})
    modern=legacy.rename(columns={"label":"original_label"})
    modern["binary_target"]=modern.original_label; modern["logit"]=[.4,-.4]; modern["predicted_class"]=[1,0]; modern["model_name"]="A4"; modern["run_id"]="run"
    p0=tmp_path/"normalization.csv"; legacy.to_csv(p0,index=False); assert len(normalize_predictions(p0))==2
    p1=tmp_path/"a.csv";p2=tmp_path/"b.csv";legacy.to_csv(p1,index=False);modern.to_csv(p2,index=False); check=prediction_check(p1,p2,.5); assert check["available"] and check["paired_case_count"]==2
    assert not prediction_check(p1,tmp_path/"none.csv",.5)["available"]

def test_pipeline_publication_archive_and_cli(tmp_path):
    result=run_analysis(paths(tmp_path),tmp_path/"analysis"); table=pd.read_csv(result/"publication_table_multiseed.csv"); assert "±" in table.original_mean_sd.iloc[0]
    differences=pd.read_csv(result/"paired_seed_differences.csv"); assert differences.difference_masked_minus_original.eq(0).all()
    with zipfile.ZipFile(tmp_path/"analysis_results.zip") as archive: assert not any("checkpoint" in x for x in archive.namelist())
    with pytest.raises(SystemExit): build_parser().parse_args([])
