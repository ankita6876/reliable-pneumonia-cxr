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


def _checkpoint(path, configuration):
    model=nn.Linear(1,1)
    torch.save({"configuration":configuration,"model_state_dict":model.state_dict()},path)
    return path


def _load_checkpoint(path, monkeypatch):
    monkeypatch.setattr(reliability_audit,"create_model",lambda *args,**kwargs:nn.Linear(1,1))
    return reliability_audit._checkpoint(path)[1]

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


def test_a4_and_original_control_are_methodologically_comparable():
    shared={"backbone":"densenet121","pretrained":True,"preprocessing":"imagenet","classifier_image_size":224,"augmentation":"historical","loss":"weighted_bce","optimizer":"adamw","learning_rate":1e-5,"weight_decay":1e-5,"epochs":20}
    validate_methodological_comparability({**shared,"input_mode":"hard_masked"},{**shared,"input_mode":"original"})


def test_comparability_reports_all_hyperparameter_mismatches():
    with pytest.raises(ValueError) as error:
        validate_methodological_comparability({"input_mode":"hard_masked","backbone":"densenet121","learning_rate":1e-5,"epochs":20},{"input_mode":"original","backbone":"resnet50","learning_rate":1e-4,"epochs":10})
    message=str(error.value)
    assert "backbone:" in message and "learning_rate:" in message and "epochs:" in message


def test_ambiguous_checkpoint_schema_fails_clearly(tmp_path, monkeypatch):
    with pytest.raises(ValueError,match=r"Ambiguous checkpoint configuration schema.*Detected keys: epoch"):
        _load_checkpoint(_checkpoint(tmp_path/"ambiguous.pt",{"epoch":3}),monkeypatch)
