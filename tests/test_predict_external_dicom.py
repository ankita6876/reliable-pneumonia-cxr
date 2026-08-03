import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

import scripts.predict_external as external


def dicom(path: Path, pixels, **fields):
    meta=FileMetaDataset(); meta.TransferSyntaxUID=ExplicitVRLittleEndian; meta.MediaStorageSOPClassUID=generate_uid(); meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(str(path),{},file_meta=meta,preamble=b"\0"*128); ds.Rows,ds.Columns=np.asarray(pixels).shape; ds.SamplesPerPixel=1; ds.PhotometricInterpretation=fields.pop("PhotometricInterpretation","MONOCHROME2"); ds.BitsAllocated=16; ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=0; ds.PixelData=np.asarray(pixels,dtype=np.uint16).tobytes()
    for key,value in fields.items(): setattr(ds,key,value)
    ds.save_as(path); return path


def manifest(tmp_path, paths):
    frame=pd.DataFrame({"patient_id":[f"p{i}" for i in range(len(paths))],"image_path":[str(x) for x in paths],"binary_target":[i%2 for i in range(len(paths))],"has_bounding_box":[False]*len(paths),"bounding_box_count":[0]*len(paths),"split":["external_test"]*len(paths),"dataset":["rsna"]*len(paths)})
    path=tmp_path/"manifest.csv"; frame.to_csv(path,index=False); return path,frame


def test_dicom_monochrome_rescale_window_and_multivalue(tmp_path):
    path=dicom(tmp_path/"x.dcm",[[0,10],[20,30]],RescaleSlope=2,RescaleIntercept=10,WindowCenter=[30,100],WindowWidth=[20,50])
    values=np.asarray(external.load_dicom_as_pil(path).convert("L")); assert values.shape==(2,2) and values[0,0]==0 and values[-1,-1]==255


def test_dicom_monochrome1_inverts_after_normalization(tmp_path):
    regular=np.asarray(external.load_dicom_as_pil(dicom(tmp_path/"m2.dcm",[[0,100],[200,300]])).convert("L"))
    inverted=np.asarray(external.load_dicom_as_pil(dicom(tmp_path/"m1.dcm",[[0,100],[200,300]],PhotometricInterpretation="MONOCHROME1")).convert("L"))
    assert np.array_equal(inverted,255-regular)


def test_dicom_percentile_and_constant_fallback_and_corruption(tmp_path):
    assert np.asarray(external.load_dicom_as_pil(dicom(tmp_path/"constant.dcm",np.ones((2,2))*7)).convert("L")).max()==0
    image=np.asarray(external.load_dicom_as_pil(dicom(tmp_path/"range.dcm",[[0,1],[2,1000]])).convert("L")); assert image.min()==0 and image.max()==255
    bad=tmp_path/"bad.dcm"; bad.write_bytes(b"not a dicom")
    with pytest.raises(ValueError,match="Unable to read DICOM"): external.load_dicom_as_pil(bad)


def test_path_manifest_validation_and_deterministic_stratification(tmp_path):
    files=[dicom(tmp_path/f"{i}.dcm",[[i,i+1],[i+2,i+3]]) for i in range(8)]; path,frame=manifest(tmp_path,files)
    frame.loc[0,"image_path"]=files[0].name; frame.to_csv(path,index=False)
    loaded=external.validate_manifest(path,tmp_path); assert loaded["_source_path"].iloc[0]==files[0] and loaded["_source_path"].iloc[1]==files[1]
    one=external.deterministic_sample(loaded,4,42); two=external.deterministic_sample(loaded,4,42)
    assert one.patient_id.tolist()==two.patient_id.tolist() and len(one)==4 and set(one.binary_target)=={0,1}
    duplicate=loaded.copy(); duplicate.loc[1,"patient_id"]=duplicate.loc[0,"patient_id"]; duplicate.drop(columns="_source_path").to_csv(path,index=False)
    with pytest.raises(ValueError,match="unique"): external.validate_manifest(path,tmp_path)


def checkpoint_config(mode="original"):
    return {"configuration":{"input_mode":mode,"backbone":"densenet121","input_size":8,"preprocessing":"imagenet","mask_threshold":.5,"lung_crop_padding":0},"model_state_dict":{}}


class TinyModel(torch.nn.Module):
    def forward(self,x): return torch.zeros((len(x),1),device=x.device)


def test_mode_requirements_existing_output_and_cpu_smoke(tmp_path,monkeypatch):
    images=[dicom(tmp_path/f"inference{i}.dcm",[[1,2],[3,4]]) for i in range(2)]; manifest_path,_=manifest(tmp_path,images); checkpoint=tmp_path/"model.pt"; checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(external.torch,"load",lambda *args,**kwargs:checkpoint_config())
    monkeypatch.setattr(external,"create_model",lambda *args,**kwargs:TinyModel())
    monkeypatch.setattr(external,"_transforms",lambda *args,**kwargs:(None,lambda image:torch.zeros((3,8,8))))
    output=tmp_path/"predictions.csv"; predictions=external.predict_external(manifest=manifest_path,image_root=tmp_path,checkpoint=checkpoint,output=output,device="cpu",batch_size=2,max_samples=2,seed=42)
    assert list(predictions.columns)==list(external.OUTPUT_COLUMNS) and len(predictions)==2 and not (tmp_path/"pngs").exists()
    metadata=json.loads((tmp_path/"predictions_metadata.json").read_text()); assert {"selected_patient_ids","pydicom_version","row_count"}.issubset(metadata) and metadata["row_count"]==2
    with pytest.raises(FileExistsError,match="overwrite"): external.predict_external(manifest=manifest_path,image_root=tmp_path,checkpoint=checkpoint,output=output)
    monkeypatch.setattr(external.torch,"load",lambda *args,**kwargs:checkpoint_config("hard_masked"))
    with pytest.raises(ValueError,match="segmentation-checkpoint"): external.predict_external(manifest=manifest_path,image_root=tmp_path,checkpoint=checkpoint,output=tmp_path/"masked.csv")


def test_cli_preserves_original_and_adds_controls():
    args=external.parse_args(["--manifest","m.csv","--image-root","images","--checkpoint","m.pt","--output","p.csv","--device","cpu","--max-samples","32","--seed","42"])
    assert args.max_samples==32 and args.seed==42 and args.segmentation_checkpoint is None
