from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.analysis.segmentation_quality_features import binary_entropy, segmentation_quality_features
import scripts.analysis.audit_segmentation_reliability as reliability_audit
from scripts.analysis.audit_segmentation_reliability import _device, masking_benefit_targets, preflight, report_and_gate, validate_methodological_comparability
from scripts.classification.checkpoint_compatibility import resolve_methodological_metadata


def _checkpoint(path, configuration):
    model=nn.Linear(1,1)
    torch.save({"configuration":configuration,"model_state_dict":model.state_dict()},path)
    return path


def _load_checkpoint(path, monkeypatch):
    monkeypatch.setattr(reliability_audit,"create_model",lambda *args,**kwargs:nn.Linear(1,1))
    return reliability_audit._checkpoint(path)[1]


def _manifest(tmp_path):
    path=tmp_path/"chexpert_splits.csv"
    path.write_text("split,patient_id,study_id,image_path,pneumonia_label\ntrain,p1,s1,x.png,1\nvalidation,p2,s2,y.png,0\n")
    return path

def test_entropy_confident_and_uncertain_masks():
    assert binary_entropy(np.array([.5]))[0] > binary_entropy(np.array([.01]))[0]
def test_empty_mask_handling():
    f=segmentation_quality_features(np.zeros((4,4))); assert f["empty_mask"] and f["foreground_pixel_count"]==0 and np.isnan(f["mask_centroid_x_normalized"])
def test_one_sided_mask_handling():
    a=np.zeros((4,4));a[:,0]=1; f=segmentation_quality_features(a); assert f["single_side_mask"] and np.isnan(f["left_right_component_balance"])
def test_connected_components_and_border_touch():
    a=np.zeros((5,5));a[0,0]=a[3,3]=1;f=segmentation_quality_features(a);assert f["connected_component_count"]==2 and f["touches_image_border"] and f["border_touch_fraction"]==.5
def test_symmetry_and_roughness():
    a=np.zeros((5,6));a[1:4,1:2]=a[1:4,4:5]=1;f=segmentation_quality_features(a);assert f["left_right_area_symmetry"]==1 and f["boundary_roughness"]>0
def test_normal_feature_values_are_finite():
    f=segmentation_quality_features(np.full((4,4),.8)); assert all(np.isfinite(v) for k,v in f.items() if isinstance(v,(float,int)) and k not in {"left_right_centroid_vertical_difference"})
def test_deterministic_output():
    a=np.arange(16).reshape(4,4)/15;assert segmentation_quality_features(a)==segmentation_quality_features(a)
def test_cuda_unavailable_error(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available",lambda:False)
    with pytest.raises(RuntimeError):_device("cuda")
def test_benefit_target_calculation():
    x=masking_benefit_targets(1.,.2,False,True); assert x["masked_benefit_bce"]==.8 and x["masked_benefit_correctness"]==1 and x["masked_helped"]
def test_test_only_manifest_rejected(tmp_path):
    p=tmp_path/"m.csv";pd.DataFrame([{"split":"test","patient_id":"a","study_id":"s","image_path":"x","pneumonia_label":1}]).to_csv(p,index=False)
    a=type("A",(),{"device":"cpu","splits_csv":p,"segmentation_checkpoint":p,"hard_masked_checkpoint":p,"original_checkpoint":p,"image_root":tmp_path,"max_samples":None})()
    with pytest.raises(ValueError):preflight(a)
def test_patient_overlap_rejected(tmp_path):
    p=tmp_path/"m.csv";pd.DataFrame([{"split":s,"patient_id":"a","study_id":s,"image_path":"x","pneumonia_label":1} for s in ("train","validation")]).to_csv(p,index=False)
    a=type("A",(),{"device":"cpu","splits_csv":p,"segmentation_checkpoint":p,"hard_masked_checkpoint":p,"original_checkpoint":p,"image_root":tmp_path,"max_samples":None})()
    with pytest.raises(ValueError):preflight(a)
def test_go_no_go_synthetic_data():
    rows=[]
    for split in ["train"]*20+["validation"]*20:
      for i in range(1): rows.append({"split":split,"label":i%2,"empty_mask":False,"single_side_mask":False,"touches_image_border":False,"hard_masked_correct":i%2==0,"original_correct":False,"hard_masked_outcome":"FN" if i%2 else "TP","high_confidence_hard_masked_error":i%2==1,"masked_helped":i%2==0,"masked_hurt":False,"masked_benefit_bce":float(i%2),"masked_benefit_correctness":i%2,**{k:float(i%2) for k in __import__("scripts.analysis.segmentation_quality_features",fromlist=["QUALITY_FEATURE_COLUMNS"]).QUALITY_FEATURE_COLUMNS}})
    r=report_and_gate(pd.DataFrame(rows),pd.DataFrame(rows));assert "go_no_go" in r


def test_legacy_a4_optimisation_checkpoint_without_input_mode_is_hard_masked(tmp_path, monkeypatch):
    raw={"experiment":"A4_regularised_optimisation","backbone":"densenet121","pretrained":True,"input_size":320,"preprocessing":"imagenet","augmentation":"historical","loss":"weighted_bce","optimizer":"adamw","learning_rate":1e-5,"weight_decay":1e-5,"epochs":20}
    config=_load_checkpoint(_checkpoint(tmp_path/"a4.pt",raw),monkeypatch)
    assert config["input_mode"]=="hard_masked" and config["classifier_image_size"]==320
    assert config["mask_threshold"]==.5 and config["lung_crop_padding"]==0
    for key,value in raw.items(): assert config[key] == value


def test_modern_a4_original_control_resolves_as_original(tmp_path, monkeypatch):
    raw={"experiment":"A4_original_control","input_mode":"original","backbone":"densenet121","pretrained":True,"input_size":224,"loss":"weighted_bce","optimizer":"adamw"}
    assert _load_checkpoint(_checkpoint(tmp_path/"original.pt",raw),monkeypatch)["input_mode"]=="original"


def test_old_nested_original_checkpoint_remains_supported(tmp_path, monkeypatch):
    config=_load_checkpoint(_checkpoint(tmp_path/"original.pt",{"model":{"name":"densenet121","image_size":256,"preprocessing":"imagenet"}}),monkeypatch)
    assert config["input_mode"]=="original" and config["classifier_image_size"]==256


def test_a4_and_original_control_are_methodologically_comparable(tmp_path):
    manifest=_manifest(tmp_path)
    shared={"backbone":"densenet121","pretrained":True,"preprocessing":"imagenet","classifier_image_size":224,"augmentation":"historical","loss":"weighted_bce","optimizer":"adamw","learning_rate":1e-5,"weight_decay":1e-5,"epochs":20}
    validate_methodological_comparability({**shared,"input_mode":"hard_masked"},{**shared,"input_mode":"original","dataset_split_path":"/kaggle/input/assets/chexpert_splits.csv","label_policy":"ignore"},supplied_splits_csv=manifest)


def test_comparability_reports_all_hyperparameter_mismatches(tmp_path):
    manifest=_manifest(tmp_path)
    with pytest.raises(ValueError) as error:
        validate_methodological_comparability({"input_mode":"hard_masked","backbone":"densenet121","learning_rate":1e-5,"epochs":20},{"input_mode":"original","backbone":"resnet50","learning_rate":1e-4,"epochs":10},supplied_splits_csv=manifest)
    message=str(error.value)
    assert "backbone:" in message and "learning_rate:" in message and "epochs:" in message


def test_ambiguous_checkpoint_schema_fails_clearly(tmp_path, monkeypatch):
    with pytest.raises(ValueError,match=r"Ambiguous checkpoint configuration schema.*Detected keys: epoch"):
        _load_checkpoint(_checkpoint(tmp_path/"ambiguous.pt",{"epoch":3}),monkeypatch)


def test_missing_legacy_metadata_resolves_from_supplied_manifest(tmp_path):
    manifest=_manifest(tmp_path)
    resolved=resolve_methodological_metadata({"experiment":"A4","backbone":"densenet121","pretrained":True,"input_size":224,"loss":"weighted_bce","optimizer":"adamw"},supplied_splits_csv=manifest)
    assert resolved["dataset_split_path"].startswith("sha256:") and resolved["label_policy"]=="ignore"


def test_windows_and_kaggle_manifest_paths_do_not_mismatch(tmp_path):
    manifest=_manifest(tmp_path)
    shared={"backbone":"densenet121","input_mode":"hard_masked"}
    validate_methodological_comparability({**shared,"dataset_split_path":r"C:\data\chexpert_splits.csv"},{**shared,"input_mode":"original","dataset_split_path":"/kaggle/input/assets/chexpert_splits.csv"},supplied_splits_csv=manifest)


def test_saved_manifest_content_identity_mismatch_fails(tmp_path):
    manifest=_manifest(tmp_path); other=tmp_path/"other"; other.mkdir()
    saved=other/manifest.name; saved.write_text("different manifest")
    with pytest.raises(ValueError,match="content does not match"):
        resolve_methodological_metadata({"input_mode":"original","dataset_split_path":str(saved)},supplied_splits_csv=manifest)


def test_explicit_conflicting_label_policy_fails(tmp_path):
    manifest=_manifest(tmp_path)
    with pytest.raises(ValueError,match="label_policy"):
        validate_methodological_comparability({"input_mode":"hard_masked"},{"input_mode":"original","label_policy":"u_zero"},supplied_splits_csv=manifest)


def _criterion_rows():
    columns=__import__("scripts.analysis.segmentation_quality_features",fromlist=["QUALITY_FEATURE_COLUMNS"]).QUALITY_FEATURE_COLUMNS
    rows=[]
    for split in ("train","validation"):
        for i in range(8):
            helped=i%2==1
            rows.append({"split":split,"hard_masked_correct":helped,"original_correct":False,"masked_helped":helped,**{key:float(i) for key in columns}})
    return pd.DataFrame(rows)


def test_masked_helped_target_is_binary_and_normalizes_boolean_object_values():
    target=reliability_audit.normalize_masked_helped_target(pd.Series([True,False,"true","false",1,0,None]))
    assert target.dropna().tolist()==[1,0,1,0,1,0]


def test_continuous_masked_helped_values_are_rejected():
    with pytest.raises(ValueError,match="continuous"):
        reliability_audit.normalize_masked_helped_target(pd.Series([0.25]))


def test_criterion4_uses_binary_target_and_handles_single_class():
    rows=_criterion_rows(); success,_,metadata=reliability_audit._criterion4(rows)
    assert metadata["criterion4_status"]=="evaluated" and isinstance(success,bool)
    rows.loc[0,"hard_masked_correct"]=np.nan
    _,_,metadata=reliability_audit._criterion4(rows)
    assert metadata["criterion4_training_count"]==7
    rows=_criterion_rows(); rows.loc[rows.split.eq("validation"),"hard_masked_correct"]=True
    success,info,metadata=reliability_audit._criterion4(rows)
    assert not success and "AUROC is undefined" in info["reason"]
    rows=_criterion_rows()
    rows["masked_helped"]=True; rows["hard_masked_correct"]=True
    success,info,metadata=reliability_audit._criterion4(rows)
    assert not success and info["status"]=="not_evaluable" and "single class" in metadata["criterion4_reason"]


def test_criterion4_imputes_from_training_data_only(monkeypatch):
    rows=_criterion_rows(); columns=__import__("scripts.analysis.segmentation_quality_features",fromlist=["QUALITY_FEATURE_COLUMNS"]).QUALITY_FEATURE_COLUMNS
    rows.loc[rows.split.eq("train"),columns[0]]=np.nan
    rows.loc[rows.split.eq("validation"),columns[0]]=999.0
    captured=[]; original_fit=__import__("sklearn.impute",fromlist=["SimpleImputer"]).SimpleImputer.fit
    def fit(self,x,*args,**kwargs):
        captured.append(np.asarray(x)); return original_fit(self,x,*args,**kwargs)
    monkeypatch.setattr("sklearn.impute.SimpleImputer.fit",fit)
    reliability_audit._criterion4(rows)
    assert captured and not np.isin(999.0,captured[0]).any()


def test_existing_csv_finalization_skips_inference_and_rejects_test_rows(tmp_path,monkeypatch):
    columns=__import__("scripts.analysis.segmentation_quality_features",fromlist=["QUALITY_FEATURE_COLUMNS"]).QUALITY_FEATURE_COLUMNS
    rows=[]
    for split in ("train","validation"):
        for i in range(8):
            helped=i%2==1
            rows.append({"split":split,"label":i%2,"hard_masked_correct":helped,"original_correct":False,"hard_masked_outcome":"TP" if helped else "FN","high_confidence_hard_masked_error":False,"masked_benefit_bce":float(i),"masked_benefit_correctness":int(helped),"masked_helped":helped,"masked_hurt":False,"both_correct":False,"both_wrong":False,"hard_masked_probability":.5,**{key:float(i) for key in columns}})
    csv=tmp_path/"audit.csv"; pd.DataFrame(rows).to_csv(csv,index=False)
    monkeypatch.setattr(reliability_audit,"_checkpoint",lambda *args:pytest.fail("finalization must not load checkpoints"))
    reliability_audit.audit(from_existing_audit_csv=csv,output_directory=tmp_path/"out",restart=True,seed=42)
    assert (tmp_path/"out"/"audit_report.json").is_file()
    bad=pd.read_csv(csv); bad.loc[0,"split"]="test"; bad.to_csv(csv,index=False)
    with pytest.raises(ValueError,match="test rows"):
        reliability_audit.audit(from_existing_audit_csv=csv,output_directory=tmp_path/"out2",restart=True,seed=42)
