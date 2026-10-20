"""Run a synthetic end-to-end forecast and print holdout scores.

Example:
    python -m stpcuq_gkan --steps 40
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from stpcuq_gkan.conformal import AdaptiveConformalInference
from stpcuq_gkan.data import synthesize_storms
from stpcuq_gkan.metrics import mae, pearson, rmse
from stpcuq_gkan.model import ModelConfig, STPCUQGKAN
from stpcuq_gkan.train import fit


def evaluate(model: STPCUQGKAN, batch) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(batch.solar_wind, batch.nodes, batch.edge_index, batch.edge_attr)
    db_hat = prediction["db"].cpu().numpy()
    db = batch.db.cpu().numpy()
    sme_hat = prediction["sme"].cpu().numpy()
    sme = batch.sme.cpu().numpy()
    risk_hat = prediction["risk_logits"].argmax(dim=-1).cpu().numpy()
    risk = batch.risk.cpu().numpy()
    residual = np.abs(db_hat - db).reshape(-1)
    fallback = float(np.std(db) + 1e-6)
    aci = AdaptiveConformalInference()
    coverage = aci.calibrate(residual, fallback)
    return {
        "db_rmse": rmse(db_hat, db),
        "db_mae": mae(db_hat, db),
        "db_corr": pearson(db_hat, db),
        "sme_corr": pearson(sme_hat, sme),
        "risk_accuracy": float(np.mean(risk_hat == risk)),
        "aci_coverage": coverage["coverage"],
        "aci_half_width": coverage["mean_half_width"],
    }


def mean_baseline_rmse(train_db: np.ndarray, holdout_db: np.ndarray) -> float:
    guess = np.full_like(holdout_db, train_db.mean())
    return rmse(guess, holdout_db)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--windows", type=int, default=160)
    parser.add_argument("--lookback", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train, holdout = synthesize_storms(args.windows, args.lookback, seed=args.seed)
    device = torch.device(args.device)
    train = train.to(device)
    holdout = holdout.to(device)
    model = STPCUQGKAN(ModelConfig(hidden=args.hidden, basis=6)).to(device)

    print("ST-PC-UQ-GKAN synthetic demo")
    print(
        f"stations {len(train.graph.codes)} | lookback {args.lookback} min | "
        f"train windows {train.solar_wind.size(0)} | holdout {holdout.solar_wind.size(0)}"
    )
    print("Targets are normalized synthetic fields, not the AGU storm scores.")
    before = evaluate(model, holdout)
    history = fit(model, train, steps=args.steps, seed=args.seed)
    print(f"train loss {history[0]['total']:.4f} -> {history[-1]['total']:.4f}")
    weights = ", ".join(
        f"{name} {history[-1][f'w_{name}']:.2f}" for name in ("db", "sme", "efield", "risk", "physics")
    )
    print(f"ReLoBRaLo weights: {weights}")

    trained = evaluate(model, train)
    after = evaluate(model, holdout)
    baseline = mean_baseline_rmse(train.db.cpu().numpy(), holdout.db.cpu().numpy())
    print(
        f"train dB RMSE {trained['db_rmse']:.3f} | correlation {trained['db_corr']:.3f}"
    )
    print(f"holdout dB RMSE  model {after['db_rmse']:.3f} | mean baseline {baseline:.3f}")
    print(f"holdout dB MAE {after['db_mae']:.3f} | correlation {after['db_corr']:.3f}")
    print(f"holdout correlation before training {before['db_corr']:.3f}")
    print(f"holdout SME correlation {after['sme_corr']:.3f}")
    print(f"holdout GIC-risk accuracy {after['risk_accuracy']:.1%}")
    print(
        f"ACI coverage {after['aci_coverage']:.1%} "
        f"(target 90%, mean half-width {after['aci_half_width']:.3f})"
    )


if __name__ == "__main__":
    main()
