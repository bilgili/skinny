"""Offline-train RQ and PQ flows on the SAME glass records and export to NFW1.

Isolates the COUPLING (RQ = "my method" vs NIS piecewise-quadratic): same data,
same loss, same hyperparameters, same V1 chart (the renderer's). Conditions on the
scene AABB (from the .nrec header) like the shader; z = V1.direction_to_square(wi)
so train and inference share the chart (the online build_dataset_np used V0 — wrong
for the V1 renderer; we use V1 here).

Run (Vulkan SDK sourced; worktree spline_flow has the nis-pq coupling):
  PYTHONPATH=src SKINNY_SPLINE_FLOW=/Users/ahmetbilgili/projects/spline_flow-nis-baseline \
    ./bin/python3.13 scripts/nis_train_offline.py --nrec .skinny_neural/glass_nodirect.nrec
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
SF = "/Users/ahmetbilgili/projects/spline_flow-nis-baseline"
sys.path.insert(0, SF)

import torch  # noqa: E402
from train import ConditionalSplineFlow2D, CHART, make_cond_encoding  # noqa: E402
from export_weights import export_flow  # noqa: E402
from skinny.sampling.path_records import read_records  # noqa: E402


def lum(c):
    return 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]


def build(nrec, clamp_pct=0.0, chart="V1"):
    recs, bmin, bext = read_records(nrec)
    ext = np.maximum(bext, 1e-6).astype(np.float32)
    p = (recs["pos"].astype(np.float32) - bmin) / ext * 2.0 - 1.0
    cond = np.concatenate([p, recs["normal"].astype(np.float32),
                           recs["wo"].astype(np.float32)], axis=1).astype(np.float32)
    wi = recs["wi_local"].astype(np.float32)
    w = lum(recs["contrib"].astype(np.float32))
    keep = (np.isfinite(w) & (w > 0.0) & np.isfinite(wi).all(axis=1) & (wi[:, 1] > 1e-4))
    cond, wi, w = cond[keep], wi[keep], w[keep]
    if clamp_pct > 0.0:
        # Firefly/contribution clamp: cap the weight at a high percentile so the
        # MLE doesn't over-concentrate on a few caustic spikes (the firefly-chasing
        # that made the unclamped guides barely match bsdf).
        cap = float(np.percentile(w, clamp_pct))
        nclip = int((w > cap).sum())
        w = np.minimum(w, cap)
        print(f"[train] firefly clamp @ p{clamp_pct:g} cap={cap:.4g} clipped {nclip} recs", flush=True)
    w = w / max(float(w.mean()), 1e-12)
    # z via the renderer's chart (default V1; NOT the V0 cylindrical map
    # build_dataset_np uses). Train chart MUST match the render cell's chart.
    z = CHART[chart].direction_to_square(torch.from_numpy(wi), None).numpy()
    z = np.clip(z, 1e-4, 1.0 - 1e-4).astype(np.float32)
    return (torch.from_numpy(cond), torch.from_numpy(z), torch.from_numpy(w.astype(np.float32)))


def train(cond, z, w, coupling, steps, encoding="E0", chart="V1", seed=0):
    torch.manual_seed(seed)
    # Conditioner encoding (AXIS 2, Jacobian-free): E0 raw | E1 gamma-band every
    # path-regime scalar (L_pos) | E3 = E1 + raw tail. make_cond_encoding returns
    # None for E0 (identity). The shader re-encodes with the matching NF_ENCODING.
    encode = make_cond_encoding(encoding, cond_dim=9, regime="path")
    m = ConditionalSplineFlow2D(cond_dim=9, num_layers=6, num_bins=24, hidden=96,
                                coupling=coupling, chart=chart,
                                encode=encode, encoding_name=encoding)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    n = cond.shape[0]; m.train()
    for step in range(steps):
        idx = torch.randint(0, n, (4096,))
        opt.zero_grad(set_to_none=True)
        lq = m.log_pdf_square(z[idx], cond[idx]).squeeze(-1)
        loss = -(w[idx] * lq).sum() / w[idx].sum().clamp_min(1e-12)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  [{coupling}] step {step:5d} NLL={float(loss):+.4f}", flush=True)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrec", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--clamp-pct", type=float, default=0.0,
                    help="firefly clamp percentile (e.g. 99); 0 = off")
    ap.add_argument("--encoding", default="E0", choices=("E0", "E1", "E3"),
                    help="conditioner encoding (renderer-supported subset)")
    ap.add_argument("--chart", default="V1", choices=("V1",),
                    help="hemisphere chart (renderer implements V1 only today)")
    ap.add_argument("--tag", default="glass_off", help="output net name prefix")
    ap.add_argument("--enc-in-name", action="store_true",
                    help="append the encoding to the net name (e.g. _E1_rqs)")
    ap.add_argument("--out-dir", default="/Users/ahmetbilgili/projects/skinny/.skinny_neural")
    args = ap.parse_args()
    cond, z, w = build(args.nrec, args.clamp_pct, args.chart)
    print(f"[train] kept {cond.shape[0]} records from {Path(args.nrec).name}  "
          f"encoding={args.encoding} chart={args.chart}", flush=True)
    out = Path(args.out_dir)
    enc_seg = f"{args.encoding}_" if args.enc_in_name else ""
    for coupling, suff in (("rqs", "rqs"), ("nis-pq", "pq")):
        tag = f"{args.tag}_{enc_seg}{suff}"
        print(f"[train] coupling={coupling}", flush=True)
        m = train(cond, z, w, coupling, args.steps, args.encoding, args.chart)
        path = out / f"{tag}.nfw1"
        export_flow(m, str(path))
        print(f"[train] exported {path}", flush=True)


if __name__ == "__main__":
    main()
