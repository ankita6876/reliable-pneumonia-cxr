"""Development-only read-only audit linking lung-mask quality to classifier failures."""
from __future__ import annotations
import argparse, hashlib, json, math, platform, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from pneumonia_ai.classification.segmentation_guided import InputMode, prepare_classifier_image
from pneumonia_ai.models.factory import create_model
from pneumonia_ai.segmentation.cache import MaskCache
from pneumonia_ai.segmentation.inference import FrozenLungSegmenter
from scripts.classification.checkpoint_compatibility import (classify_checkpoint_configuration, load_adjacent_experiment_configuration, normalize_checkpoint_configuration, normalize_nested_original_baseline_configuration, resolve_methodological_metadata)
from scripts.train_baseline import _transforms
from scripts.analysis.segmentation_quality_features import QUALITY_FEATURE_COLUMNS, segmentation_quality_features


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("splits-csv","image-root","segmentation-checkpoint","hard-masked-checkpoint","original-checkpoint"): p.add_argument("--"+n,type=Path)
    p.add_argument("--output-directory",type=Path,required=True); p.add_argument("--from-existing-audit-csv",type=Path)
    p.add_argument("--mask-cache",type=Path); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--num-workers",type=int,default=0); p.add_argument("--seed",type=int,default=42); p.add_argument("--max-samples",type=int); p.add_argument("--restart",action="store_true")
    args=p.parse_args()
    if args.from_existing_audit_csv is None:
        missing=[name for name in ("splits_csv","image_root","segmentation_checkpoint","hard_masked_checkpoint","original_checkpoint") if getattr(args,name) is None]
        if missing: p.error("the following arguments are required unless --from-existing-audit-csv is supplied: "+", ".join("--"+name.replace("_","-") for name in missing))
    return args

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available(): raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    return torch.device(name)

def preflight(args: argparse.Namespace) -> pd.DataFrame:
    _device(args.device)
    for p in (args.splits_csv,args.segmentation_checkpoint,args.hard_masked_checkpoint,args.original_checkpoint):
        if not p.is_file(): raise FileNotFoundError(f"Required input does not exist: {p}")
    if not args.image_root.is_dir(): raise NotADirectoryError(f"Image root does not exist: {args.image_root}")
    df=pd.read_csv(args.splits_csv)
    required={"split","patient_id","study_id","image_path","pneumonia_label"}
    if not required.issubset(df): raise ValueError("Split manifest is missing required columns: "+", ".join(sorted(required-set(df))))
    dev=df.loc[df.split.isin(["train","validation"])].copy()
    if dev.empty: raise ValueError("Manifest contains no development rows; test-only manifests are rejected.")
    if set(dev.split)!={"train","validation"}: raise ValueError("Audit requires both train and validation development rows.")
    overlap=set(dev.loc[dev.split=="train","patient_id"].astype(str)) & set(dev.loc[dev.split=="validation","patient_id"].astype(str))
    if overlap: raise ValueError("Patient-level split overlap detected between train and validation.")
    if args.max_samples: dev=dev.sort_values(["split","patient_id","study_id","image_path"],kind="stable").head(args.max_samples)
    return dev.sort_values(["split","patient_id","study_id","image_path"],kind="stable").reset_index(drop=True)

def _checkpoint(path: Path) -> tuple[dict[str,Any],dict[str,Any],torch.nn.Module]:
    state=torch.load(path,map_location="cpu",weights_only=False); raw=state.get("configuration", state.get("config",{}))
    if not isinstance(raw,Mapping): raise ValueError(f"Checkpoint lacks a configuration: {path}")
    adjacent=load_adjacent_experiment_configuration(path)
    schema=classify_checkpoint_configuration(raw)
    if schema in {"optimisation","legacy_optimisation"}:
        if schema == "legacy_optimisation": raw={**raw,"input_mode":InputMode.HARD_MASKED.value}
        config=normalize_checkpoint_configuration(raw,adjacent)
        if adjacent: config.update({k:v for k,v in adjacent.items() if k not in config})
    else:
        config=normalize_nested_original_baseline_configuration(raw)
    config.setdefault("backbone","densenet121"); model=create_model(str(config["backbone"]),pretrained=False)
    key="model_state_dict" if "model_state_dict" in state else "model_state"
    if key not in state: raise ValueError(f"Checkpoint lacks model weights: {path}")
    model.load_state_dict(state[key],strict=True); return state,config,model

COMPARABILITY_FIELDS = ("backbone", "pretrained", "preprocessing", "input_size", "classifier_image_size", "augmentation", "horizontal_flip", "rotation_degrees", "loss", "optimizer", "learning_rate", "backbone_learning_rate", "head_learning_rate", "weight_decay", "scheduler", "batch_size", "epochs", "early_stopping_patience", "seed", "dataset_split_path", "label_policy")

def validate_methodological_comparability(hard: Mapping[str,Any], original: Mapping[str,Any], *, supplied_splits_csv: Path) -> None:
    """Require paired A4/control checkpoints to differ only in input treatment."""
    hard=resolve_methodological_metadata(hard,supplied_splits_csv=supplied_splits_csv)
    original=resolve_methodological_metadata(original,supplied_splits_csv=supplied_splits_csv)
    mismatches=[]
    for field in COMPARABILITY_FIELDS:
        # Optimisation checkpoints store these fields directly; legacy aliases are only for clear errors.
        left,right=hard.get(field),original.get(field)
        if left != right: mismatches.append(f"{field}: hard_masked={left!r}, original={right!r}")
    if hard.get("input_mode") != "hard_masked": mismatches.append(f"hard_masked input_mode={hard.get('input_mode')!r}")
    if original.get("input_mode") != "original": mismatches.append(f"original input_mode={original.get('input_mode')!r}")
    if mismatches: raise ValueError("Checkpoint pairs are not methodologically comparable; mismatched fields: " + "; ".join(mismatches))

def _transform(config: Mapping[str,Any]):
    size=int(config["classifier_image_size"]); preprocessing=str(config.get("preprocessing","imagenet"))
    return _transforms(size,preprocessing,augmentation="none",horizontal_flip=False,rotation_degrees=0)[1]

def _prediction(model: torch.nn.Module, image: Image.Image, transform: Any, device: torch.device) -> tuple[float,float]:
    with torch.inference_mode():
        logit=float(model(transform(image).unsqueeze(0).to(device)).view(-1).cpu()[0])
    return logit, float(1/(1+math.exp(-logit)))

def _outcomes(prefix: str, row: dict[str,Any], target: int) -> None:
    p=row[prefix+"probability"]; pred=int(p>=.5); correct=pred==target
    row.update({prefix+"predicted_class":pred,prefix+"confidence":max(p,1-p),prefix+"binary_target":target,prefix+"correct":correct,prefix+"bce_loss":-(target*math.log(max(p,1e-7))+(1-target)*math.log(max(1-p,1e-7))),prefix+"outcome":("TP" if pred and target else "TN" if not pred and not target else "FP" if pred else "FN")})

def masking_benefit_targets(original_bce: float, hard_masked_bce: float, original_correct: bool, hard_masked_correct: bool) -> dict[str, float | int | bool]:
    """Paired, label-preserving targets; positive BCE benefit means masking helped."""
    delta=int(hard_masked_correct)-int(original_correct)
    helped=bool(hard_masked_correct and not original_correct)
    return {"masked_benefit_bce":float(original_bce-hard_masked_bce),"masked_benefit_correctness":delta,"masked_helped":helped,"masked_hurt":delta==-1,"both_correct":bool(hard_masked_correct and original_correct),"both_wrong":bool(not hard_masked_correct and not original_correct)}

def normalize_masked_helped_target(values: pd.Series) -> pd.Series:
    """Return the declared binary masking-benefit target as nullable integers."""
    return normalize_binary_boolean_series(values,name="masked_helped").astype("Int64")

def normalize_binary_boolean_series(values: pd.Series, *, name: str) -> pd.Series:
    """Strictly normalize a 0/1 or Boolean column, preserving missing values."""
    text=values.astype("string").str.strip().str.lower()
    mapped=values.where(~text.isin({"true","false"}),text.map({"true":1,"false":0}))
    numeric=pd.to_numeric(mapped,errors="coerce")
    invalid=values.notna() & (numeric.isna() | ~numeric.isin([0,1]))
    if invalid.any():
        examples=values.loc[invalid].head(3).tolist()
        raise ValueError(f"{name} must contain only Boolean or verified 0/1 values; continuous or invalid values: {examples!r}")
    return numeric.astype("Int64").astype("boolean")

def _criterion4_target(rows: pd.DataFrame) -> pd.Series:
    saved=normalize_masked_helped_target(rows["masked_helped"])
    hard=normalize_binary_boolean_series(rows["hard_masked_correct"],name="hard_masked_correct").astype("Int64")
    original=normalize_binary_boolean_series(rows["original_correct"],name="original_correct").astype("Int64")
    target=pd.Series(pd.NA,index=rows.index,dtype="Int64")
    valid=hard.notna() & original.notna()
    target.loc[valid]=((hard.loc[valid] == 1) & (original.loc[valid] == 0)).astype(int)
    if not saved.dropna().eq(target.loc[saved.notna()]).all(): raise ValueError("masked_helped disagrees with hard_masked_correct and original_correct.")
    return target

def _recalculate_masking_indicators(rows: pd.DataFrame) -> None:
    hard=normalize_binary_boolean_series(rows["hard_masked_correct"],name="hard_masked_correct")
    original=normalize_binary_boolean_series(rows["original_correct"],name="original_correct")
    derived={"masked_helped":hard & ~original,"masked_hurt":~hard & original,"both_correct":hard & original,"both_wrong":~hard & ~original}
    for column,value in derived.items():
        saved=normalize_binary_boolean_series(rows[column],name=column)
        comparable=saved.notna() & value.notna()
        if not saved.loc[comparable].eq(value.loc[comparable]).all(): raise ValueError(f"{column} disagrees with correctness columns.")
        rows[column]=value

def audit(**kwargs: Any) -> dict[str,Any]:
    args=argparse.Namespace(**kwargs); start=datetime.now(timezone.utc)
    out=args.output_directory
    if out.exists() and any(out.iterdir()) and not args.restart: raise FileExistsError("Audit output exists; use --restart to overwrite.")
    if getattr(args,"from_existing_audit_csv",None) is not None:
        frame=_load_existing_audit_csv(args.from_existing_audit_csv)
        print(f"Loaded existing audit CSV with {len(frame)} rows; skipping segmentation and classifier inference.")
        return _finalize_audit(args,out,frame,start,existing_csv=True)
    dev=preflight(args) # no output mutation before this line
    torch.manual_seed(args.seed); np.random.seed(args.seed); device=_device(args.device)
    hs,hc,hm=_checkpoint(args.hard_masked_checkpoint); os,oc,om=_checkpoint(args.original_checkpoint)
    validate_methodological_comparability(hc,oc,supplied_splits_csv=args.splits_csv)
    segmenter=FrozenLungSegmenter(args.segmentation_checkpoint,device); hm.to(device).eval(); om.to(device).eval(); htf,otf=_transform(hc),_transform(oc); threshold=float(hc["mask_threshold"])
    out.mkdir(parents=True,exist_ok=True); cache=MaskCache(args.mask_cache or out/"mask_cache")
    cache.validate_or_initialise_metadata({"schema_version":1,"segmentation_checkpoint_sha256":_sha(args.segmentation_checkpoint),"segmentation_input_size":segmenter.image_size,"mask_threshold":threshold,"postprocessing":"none"})
    rows=[]
    for rec in dev.to_dict("records"):
        rel=Path(str(rec["image_path"]).replace("\\","/")); source=(args.image_root/rel).resolve()
        if not source.is_file(): raise FileNotFoundError(f"Manifest image not found: {source}")
        with Image.open(source) as opened: image=opened.convert("RGB")
        key=cache.key(source,args.segmentation_checkpoint,threshold,segmenter.image_size); probability=cache.get(key)
        if probability is None:
            probability=segmenter.predict_proba(image); cache.set(key,probability,source_path=source)
        q=segmentation_quality_features(probability.numpy(),threshold)
        masked=prepare_classifier_image(image,InputMode.HARD_MASKED,probability_mask=probability,threshold=threshold,output_size=int(hc["classifier_image_size"])).convert("RGB")
        hl,hp=_prediction(hm,masked,htf,device); ol,op=_prediction(om,image,otf,device)
        row={"patient_id":str(rec["patient_id"]),"study_id":str(rec["study_id"]),"image_path":str(rec["image_path"]),"split":rec["split"],"label":int(rec.get("raw_pneumonia_label",rec["pneumonia_label"])),**q,"hard_masked_logit":hl,"hard_masked_probability":hp,"original_logit":ol,"original_probability":op}
        if row["label"] in (0,1):
            _outcomes("hard_masked_",row,row["label"]); _outcomes("original_",row,row["label"]); row["high_confidence_hard_masked_error"]=(not row["hard_masked_correct"] and row["hard_masked_confidence"]>=.8); row.update(masking_benefit_targets(row["original_bce_loss"],row["hard_masked_bce_loss"],row["original_correct"],row["hard_masked_correct"]))
        else: row["supervised_analysis_excluded"]=True
        rows.append(row)
    frame=pd.DataFrame(rows)
    if (frame.split=="test").any(): raise RuntimeError("Safety failure: test rows would enter output.")
    frame.to_csv(out/"segmentation_reliability_audit.csv",index=False)
    return _finalize_audit(args,out,frame,start,existing_csv=False,device=str(device),mask_threshold=threshold)

def _load_existing_audit_csv(path: Path) -> pd.DataFrame:
    if not path.is_file(): raise FileNotFoundError(f"Existing audit CSV does not exist: {path}")
    frame=pd.read_csv(path)
    required={"split","label","hard_masked_correct","original_correct","hard_masked_outcome","high_confidence_hard_masked_error","masked_benefit_bce","masked_benefit_correctness","masked_helped","masked_hurt","both_correct","both_wrong","hard_masked_probability",*QUALITY_FEATURE_COLUMNS}
    missing=required-set(frame)
    if missing: raise ValueError("Existing audit CSV is missing required columns: "+", ".join(sorted(missing)))
    if frame.split.eq("test").any(): raise ValueError("Existing audit CSV contains test rows; finalization refuses test data.")
    if not set(frame.split.dropna()).issubset({"train","validation"}): raise ValueError("Existing audit CSV has unsupported split values.")
    return frame

def _finalize_audit(args: argparse.Namespace, out: Path, frame: pd.DataFrame, start: datetime, *, existing_csv: bool, device: str="not_run", mask_threshold: float | None=None) -> dict[str,Any]:
    out.mkdir(parents=True,exist_ok=True)
    definite=frame.loc[pd.to_numeric(frame.label,errors="coerce").isin([0,1])].copy()
    for column in ("hard_masked_correct","original_correct","high_confidence_hard_masked_error","masked_hurt","both_correct","both_wrong"):
        definite[column]=normalize_binary_boolean_series(definite[column],name=column)
    _recalculate_masking_indicators(definite)
    definite["masked_helped"]=_criterion4_target(definite)
    summaries(out,frame,definite); report=report_and_gate(frame,definite)
    hashes=None if existing_csv else {"segmentation":_sha(args.segmentation_checkpoint),"hard_masked":_sha(args.hard_masked_checkpoint),"original":_sha(args.original_checkpoint)}
    report.update({"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},"start_time":start.isoformat(),"end_time":datetime.now(timezone.utc).isoformat(),"device":device,"python":platform.python_version(),"pytorch":torch.__version__,"git_commit":_git(),"checkpoint_hashes":hashes,"mask_threshold":mask_threshold,"finalized_from_existing_audit_csv":existing_csv,"stochastic_features":"omitted: FrozenLungSegmenter exposes deterministic inference only; no safe TTA/dropout API exists."})
    (out/"audit_report.json").write_text(json.dumps(report,indent=2,default=_json)); (out/"audit_metadata.json").write_text(json.dumps(report,indent=2,default=_json)); plots(out,definite)
    return report

def _git():
    try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    except Exception:return None
def _json(v): return None if isinstance(v,float) and not math.isfinite(v) else str(v)

def _metrics(x: pd.DataFrame) -> dict[str,Any]:
    y=x.label.to_numpy(int); p=x.hard_masked_probability.to_numpy(float); pred=p>=.5; tp=((pred)&(y==1)).sum(); tn=((~pred)&(y==0)).sum(); fp=((pred)&(y==0)).sum(); fn=((~pred)&(y==1)).sum()
    sens=tp/(tp+fn) if tp+fn else np.nan; spec=tn/(tn+fp) if tn+fp else np.nan
    ordered=x.sort_values("hard_masked_probability"); chunks=(ordered.iloc[index] for index in np.array_split(np.arange(len(ordered)),min(10,len(ordered)))) if len(ordered) else (); ece=sum(len(b)/len(y)*abs(b.hard_masked_probability.mean()-b.label.mean()) for b in chunks if len(b)) if len(x) else np.nan
    return {"count":len(x),"auroc":roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,"auroc_reason":"single_class" if len(np.unique(y))<2 else "","pr_auc":average_precision_score(y,p) if len(np.unique(y))==2 else np.nan,"accuracy":(pred==y).mean() if len(y) else np.nan,"balanced_accuracy":np.nanmean([sens,spec]),"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,"sensitivity":sens,"specificity":spec,"ece":ece,"brier_score":np.mean((p-y)**2) if len(y) else np.nan,"negative_log_likelihood":np.mean(-(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1)))) if len(y) else np.nan,"false_negative_rate":fn/(fn+tp) if fn+tp else np.nan,"mean_masked_benefit_bce":x.masked_benefit_bce.mean() if len(x) else np.nan}

def _bh(p: np.ndarray) -> np.ndarray:
    n=len(p); order=np.argsort(p); out=np.empty(n); running=1.
    for rank,idx in reversed(list(enumerate(order,1))): running=min(running,p[idx]*n/rank); out[idx]=running
    return out

def summaries(out: Path, allrows: pd.DataFrame, d: pd.DataFrame) -> None:
    allrows[list(QUALITY_FEATURE_COLUMNS)].agg(["count","mean","std","median","min","max"]).T.reset_index(names="feature").to_csv(out/"segmentation_quality_summary.csv",index=False)
    records=[]
    for grouping,column in [("hard_masked_outcome","hard_masked_outcome"),("hard_masked_correctness","hard_masked_correct"),("high_confidence_error","high_confidence_hard_masked_error")]:
        for group,x in d.groupby(column,dropna=False):
            for f in QUALITY_FEATURE_COLUMNS: records.append({"grouping":grouping,"group":str(group),"feature":f,"count":len(x),"mean":x[f].mean(),"median":x[f].median(),"std":x[f].std()})
    pd.DataFrame(records).to_csv(out/"quality_by_outcome.csv",index=False)
    qrows=[]
    for f in QUALITY_FEATURE_COLUMNS:
        v=d.loc[d.split=="validation"].copy()
        try:v["quartile"]=pd.qcut(v[f],4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
        except ValueError: continue
        for q,x in v.groupby("quartile",observed=True): qrows.append({"feature":f,"quartile":str(q),**_metrics(x)})
    pd.DataFrame(qrows).to_csv(out/"quality_quartile_performance.csv",index=False)
    targets={"hard_masked_bce_loss":"continuous","hard_masked_correct":"binary","hard_masked_confidence":"continuous","false_negative_indicator":"binary","high_confidence_error_indicator":"binary","masked_benefit_bce":"continuous","masked_benefit_correctness":"continuous"}; d=d.assign(false_negative_indicator=d.hard_masked_outcome.eq("FN"),high_confidence_error_indicator=d.high_confidence_hard_masked_error)
    cor=[]
    for f in QUALITY_FEATURE_COLUMNS:
        for target,kind in targets.items():
            x=d[[f,target]].dropna(); n=len(x); r=x[f].corr(x[target],method="spearman") if n>2 and x[f].nunique()>1 and x[target].nunique()>1 else np.nan; z=abs(r)*math.sqrt(max(n-3,0)) if np.isfinite(r) else np.nan; p=math.erfc(z/math.sqrt(2)) if np.isfinite(z) else np.nan
            cor.append({"feature":f,"target":target,"statistic":"point_biserial_equivalent_spearman" if kind=="binary" else "spearman","correlation":r,"p_value":p,"n":n})
    c=pd.DataFrame(cor); valid=c.p_value.notna(); c.loc[valid,"fdr_p_value"]=_bh(c.loc[valid,"p_value"].to_numpy()); c.to_csv(out/"correlation_summary.csv",index=False)
    benefit=[]
    for label,x in [("overall",d),*[("outcome_"+str(k),v) for k,v in d.groupby("hard_masked_outcome")],*[("flag_"+f,v) for f in ("empty_mask","single_side_mask","touches_image_border") for _,v in d.groupby(f)]]: benefit.append({"group":label,"count":len(x),"proportion_helped":x.masked_helped.mean(),"proportion_hurt":x.masked_hurt.mean(),"proportion_same":(x.masked_benefit_correctness==0).mean(),"mean_bce_benefit":x.masked_benefit_bce.mean(),"median_bce_benefit":x.masked_benefit_bce.median()})
    pd.DataFrame(benefit).to_csv(out/"masked_benefit_summary.csv",index=False)

def report_and_gate(frame: pd.DataFrame,d: pd.DataFrame)->dict[str,Any]:
    d=d.copy()
    for column in ("hard_masked_correct","original_correct","high_confidence_hard_masked_error","masked_helped","masked_hurt"):
        d[column]=normalize_binary_boolean_series(d[column],name=column)
    counts=_audit_count_invariants(d)
    v=d.loc[d.split=="validation"].copy(); cor=pd.read_csv(frame.attrs.get("correlation_path","x")) if False else None
    # Criterion 1 is recomputed here to keep the decision gate self-contained.
    associations=[]
    for f in QUALITY_FEATURE_COLUMNS:
        x=v[[f,"masked_benefit_bce"]].dropna(); r=x[f].corr(x.masked_benefit_bce,method="spearman") if len(x)>2 and x[f].nunique()>1 else np.nan; p=math.erfc(abs(r)*math.sqrt(len(x)-3)/math.sqrt(2)) if np.isfinite(r) and len(x)>3 else np.nan; associations.append((f,r,p))
    valid=np.array([a[2] for a in associations],float); adj=_bh(np.where(np.isfinite(valid),valid,1)); strongest=associations[int(np.nanargmax(np.abs([a[1] for a in associations])))] if associations else (None,np.nan,np.nan); criterion1=any(abs(r)>=.1 and q<.05 for (_,r,_),q in zip(associations,adj))
    # Low reliability is operationalised as high entropy (Q4), declared rather than tuned.
    if len(v)>3:
        v["reliability_quartile"]=pd.qcut(v.mask_entropy_mean,4,labels=False,duplicates="drop"); low=v[v.reliability_quartile==v.reliability_quartile.max()]; high=v[v.reliability_quartile==v.reliability_quartile.min()]; lowerr=(~normalize_binary_boolean_series(low.hard_masked_correct,name="hard_masked_correct").dropna()).mean(); higherr=(~normalize_binary_boolean_series(high.hard_masked_correct,name="hard_masked_correct").dropna()).mean(); criterion2=bool(higherr>0 and lowerr/higherr>=1.2); hfn=low.loc[low.hard_masked_outcome.eq("FN")].shape[0]/max((v.hard_masked_outcome=="FN").sum(),1); hen=len(low)/len(v); criterion3=bool(hen>0 and hfn/hen>=1.5)
    else: criterion2=criterion3=False; lowerr=higherr=np.nan
    # Criterion 4: train-only imputation/scaling and an explicitly binary target.
    criterion4,model_info,criterion4_metadata=_criterion4(d)
    fn=[]
    for f in QUALITY_FEATURE_COLUMNS:
        x=v[[f,"hard_masked_outcome"]].dropna(); y=x.hard_masked_outcome.eq("FN"); r=x[f].corr(y,method="spearman") if y.nunique()==2 and x[f].nunique()>1 else np.nan; fn.append((f,r))
    sf=max(fn,key=lambda a:abs(a[1]) if np.isfinite(a[1]) else -1) if fn else (None,np.nan)
    return {"development_images_analyzed":len(frame),"definite_labels":len(d),"train_count":int((frame.split=="train").sum()),"validation_count":int((frame.split=="validation").sum()),"empty_mask_count":int(frame.empty_mask.sum()),"single_side_mask_count":int(frame.single_side_mask.sum()),"border_touch_count":int(frame.touches_image_border.sum()),"hard_masked_error_count":counts["hard_masked_error_count"],"original_image_error_count":counts["original_image_error_count"],"hard_masked_false_negative_count":counts["hard_masked_false_negative_count"],"high_confidence_hard_masked_error_count":counts["high_confidence_hard_masked_error_count"],"fraction_masking_helped":float(d.masked_helped.mean()),"fraction_masking_hurt":float(d.masked_hurt.mean()),"strongest_quality_feature_associated_with_masked_benefit":{"feature":strongest[0],"spearman":strongest[1]},"strongest_feature_associated_with_false_negatives":{"feature":sf[0],"spearman":sf[1]},"quality_quartile_with_highest_false_negative_rate":"high_entropy_Q4",**criterion4_metadata,"go_no_go":{"proceed":bool(criterion1 or criterion2 or criterion3 or criterion4),"criterion_1":criterion1,"criterion_2":criterion2,"criterion_3":criterion3,"criterion_4":criterion4,"logistic_model":model_info}}

def _audit_count_invariants(rows: pd.DataFrame) -> dict[str,int]:
    n=len(rows)
    hard=rows["hard_masked_correct"]; original=rows["original_correct"]
    if hard.isna().any() or original.isna().any(): raise ValueError("Correctness columns contain missing values for definite-label rows.")
    hard_errors=int((~hard.dropna()).sum()); original_errors=int((~original.dropna()).sum())
    outcomes=rows["hard_masked_outcome"].astype("string")
    outcome_counts={name:int(outcomes.eq(name).sum()) for name in ("TP","TN","FP","FN")}
    if outcomes.isna().any() or sum(outcome_counts.values())!=n:
        raise ValueError("Hard-masked outcome invariant failed: TP + TN + FP + FN must equal definite-label rows.")
    if int(hard.sum())+hard_errors!=n or int(original.sum())+original_errors!=n:
        raise ValueError("Correctness invariant failed: correct count + error count must equal definite-label rows.")
    high_confidence=int(rows["high_confidence_hard_masked_error"].dropna().sum())
    values={"hard_masked_error_count":hard_errors,"original_image_error_count":original_errors,"hard_masked_false_negative_count":outcome_counts["FN"],"high_confidence_hard_masked_error_count":high_confidence}
    if any(value<0 or value>n for value in values.values()): raise ValueError("Audit count invariant failed: counts must be between zero and definite-label rows.")
    return values

def _criterion4(rows: pd.DataFrame) -> tuple[bool,dict[str,Any],dict[str,Any]]:
    target=_criterion4_target(rows)
    train=rows.loc[rows.split.eq("train")].copy(); validation=rows.loc[rows.split.eq("validation")].copy()
    y_train=target.loc[train.index].dropna().astype(int); y_validation=target.loc[validation.index].dropna().astype(int)
    metadata={"criterion4_target_definition":"hard_masked_correct & ~original_correct (binary 0/1)","criterion4_training_count":int(len(y_train)),"criterion4_validation_count":int(len(y_validation)),"criterion4_training_positive_count":int(y_train.sum()),"criterion4_validation_positive_count":int(y_validation.sum()),"criterion4_status":"not_evaluable","criterion4_reason":""}
    if not set(y_train.unique()).issubset({0,1}) or not set(y_validation.unique()).issubset({0,1}): raise ValueError("criterion 4 masked_helped target must contain only 0 and 1.")
    if len(y_train)==0: metadata["criterion4_reason"]="no definite training targets"; return False,{"status":"not_evaluable","reason":metadata["criterion4_reason"]},metadata
    if y_train.nunique()!=2: metadata["criterion4_reason"]="training target has a single class"; return False,{"status":"not_evaluable","reason":metadata["criterion4_reason"]},metadata
    if len(y_validation)==0: metadata["criterion4_reason"]="no definite validation targets"; return False,{"status":"not_evaluable","reason":metadata["criterion4_reason"]},metadata
    if y_validation.nunique()!=2: metadata["criterion4_reason"]="validation target has a single class; AUROC is undefined"; return False,{"status":"not_evaluable","reason":metadata["criterion4_reason"]},metadata
    train_features=train.loc[y_train.index,list(QUALITY_FEATURE_COLUMNS)].replace([np.inf,-np.inf],np.nan)
    validation_features=validation.loc[y_validation.index,list(QUALITY_FEATURE_COLUMNS)].replace([np.inf,-np.inf],np.nan)
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("logistic",LogisticRegression(max_iter=1000,random_state=0,C=1.0))])
    model.fit(train_features,y_train)
    score=float(roc_auc_score(y_validation,model.predict_proba(validation_features)[:,1]))
    metadata["criterion4_status"]="evaluated"
    logistic=model.named_steps["logistic"]
    return score>=.60,{"status":"evaluated","validation_auroc":score,"coefficients":dict(zip(QUALITY_FEATURE_COLUMNS,map(float,logistic.coef_[0])))},metadata

def _save(fig: Any, out: Path, name: str) -> None:
    fig.tight_layout(); fig.savefig(out/(name+".png"),dpi=300); fig.savefig(out/(name+".pdf")); plt.close(fig)
def plots(out: Path,d: pd.DataFrame) -> None:
    if d.empty: return
    fig,ax=plt.subplots(figsize=(7,4));
    for g,x in d.groupby("hard_masked_outcome"): ax.hist(x.mask_entropy_mean,alpha=.45,label=str(g))
    ax.set(xlabel="Mask entropy mean",ylabel="Images"); ax.legend(); _save(fig,out,"quality_feature_distributions_by_outcome")
    for feature,name in [("mask_entropy_mean","masked_benefit_vs_mask_entropy"),("foreground_area_ratio","masked_benefit_vs_foreground_area")]:
        fig,ax=plt.subplots(figsize=(6,4)); ax.scatter(d[feature],d.masked_benefit_bce,s=10,alpha=.6); ax.set(xlabel=feature,ylabel="Original BCE âˆ’ masked BCE"); _save(fig,out,name)
    v=d[d.split=="validation"].copy()
    try:v["q"]=pd.qcut(v.mask_entropy_mean,4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
    except ValueError:v["q"]="all"
    for target,name,ylabel in [("hard_masked_correct","error_rate_by_reliability_quartile","Error rate"),("hard_masked_outcome","false_negative_rate_by_reliability_quartile","False-negative rate")]:
        fig,ax=plt.subplots(figsize=(6,4)); values=[]
        for _,x in v.groupby("q",observed=True):
            correct=normalize_binary_boolean_series(x.hard_masked_correct,name="hard_masked_correct")
            values.append((~correct.dropna()).mean() if target=="hard_masked_correct" else x.hard_masked_outcome.eq("FN").mean())
        ax.bar([str(q) for q in v.q.drop_duplicates()],values); ax.set(ylabel=ylabel,xlabel="Entropy quartile (Q4 lowest reliability)"); _save(fig,out,name)
    fig,ax=plt.subplots(figsize=(6,4));
    for q,x in v.groupby("q",observed=True):
        bins=np.linspace(0,1,6); ids=np.digitize(x.hard_masked_probability,bins)-1; ax.plot([x.hard_masked_probability.iloc[ids==i].mean() for i in range(5)],[x.label.iloc[ids==i].mean() for i in range(5)],marker="o",label=str(q))
    ax.plot([0,1],[0,1],"k--"); ax.set(xlabel="Predicted probability",ylabel="Observed frequency"); ax.legend(); _save(fig,out,"calibration_by_reliability_quartile")
    fig,ax=plt.subplots(figsize=(10,8)); corr=d[list(QUALITY_FEATURE_COLUMNS)].corr(method="spearman"); im=ax.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=6); ax.set_yticks(range(len(corr)),corr.index,fontsize=6); fig.colorbar(im,ax=ax); _save(fig,out,"quality_feature_correlation_heatmap")

if __name__ == "__main__": audit(**vars(parse_args()))
