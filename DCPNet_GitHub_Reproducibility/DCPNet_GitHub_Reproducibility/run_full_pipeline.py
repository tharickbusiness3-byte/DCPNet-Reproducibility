from __future__ import annotations
import argparse, yaml, torch
from pathlib import Path
from src.utils import seed_everything, sha256_file
from src.data import load_split_records, build_loaders
from src.train import train_model
from src.model import DCPNet
from src.evaluate import evaluate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root",required=True)
    ap.add_argument("--split_json",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--config",default="configs/dcpnet.yaml")
    ap.add_argument("--resume",default=None)
    args=ap.parse_args()

    cfg=yaml.safe_load(Path(args.config).read_text())
    seed_everything(cfg["seed"])
    split_hash=sha256_file(args.split_json)
    records=load_split_records(args.data_root,args.split_json)
    loaders=build_loaders(records,cfg)

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    model,history=train_model(loaders,cfg,out/"checkpoints",split_hash,args.resume)

    best=out/"checkpoints"/"best.pt"
    ck=torch.load(best,map_location="cuda" if torch.cuda.is_available() else "cpu")
    model=DCPNet(tuple(cfg["model"]["channels"]),
                 cfg["model"]["refinement_fp32"],
                 cfg["model"]["decoder2_fp32"]).to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(ck["model"])

    df=evaluate(model,loaders["test"],cfg,next(model.parameters()).device)
    df.to_csv(out/"locked_test_patient_metrics.csv",index=False)
    print(df[["Dice_WT","Dice_TC","Dice_ET","macro_Dice"]].mean())

if __name__=="__main__":
    main()
