"""Aggregate the 4 held-out-loss eval JSONs into a RICL-vs-plain results table (markdown).

Reads <results_dir>/{eval1,eval2}_{treatment,baseline}.json (from heldout_loss.py) and reports,
per scenario: overall baseline vs treatment dynamics/action loss + delta (treatment - baseline;
NEGATIVE = retrieval helps), per task, and for Eval 2 also per difficulty bucket.
"""
import argparse
import json
import os

# Eval-2 difficulty gradient (the unseen tasks), from the plan.
BUCKETS = {
    "near_duplicate": ["stack_blocks_three", "click_bell"],
    "shared_primitive": ["place_shoe", "place_dual_shoes", "open_microwave",
                         "handover_mic", "stack_bowls_three"],
    "novel": ["turn_switch", "adjust_bottle", "dump_bin_bigbin"],
}


def _load(d, name):
    p = os.path.join(d, f"{name}.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def _row(t, b, key):
    bt, bb = t["per_task"].get(key), b["per_task"].get(key)
    if not bt or not bb:
        return None
    return {
        "task": key,
        "base_dyn": bb["dynamics_loss"], "treat_dyn": bt["dynamics_loss"],
        "d_dyn": bt["dynamics_loss"] - bb["dynamics_loss"],
        "base_act": bb["action_loss"], "treat_act": bt["action_loss"],
        "d_act": bt["action_loss"] - bb["action_loss"],
    }


def _section(lines, title, treat, base, bucketize=False):
    lines.append(f"\n## {title}\n")
    if not treat or not base:
        lines.append("_(missing eval json)_\n")
        return
    o_b, o_t = base["overall"], treat["overall"]
    lines.append(f"**Overall** (n={o_t['n']}): "
                 f"dynamics {o_b['dynamics_loss']:.4f} → {o_t['dynamics_loss']:.4f} "
                 f"(Δ {o_t['dynamics_loss']-o_b['dynamics_loss']:+.4f}); "
                 f"action {o_b['action_loss']:.4f} → {o_t['action_loss']:.4f} "
                 f"(Δ {o_t['action_loss']-o_b['action_loss']:+.4f}).  "
                 f"_(plain → RICL; Δ<0 = retrieval helps)_\n")
    lines.append("\n| task | dyn plain | dyn RICL | Δdyn | act plain | act RICL | Δact |")
    lines.append("|---|---|---|---|---|---|---|")
    tasks = sorted(set(treat["per_task"]) & set(base["per_task"]))
    for k in tasks:
        r = _row(treat, base, k)
        if r:
            lines.append(f"| {r['task']} | {r['base_dyn']:.4f} | {r['treat_dyn']:.4f} | "
                         f"{r['d_dyn']:+.4f} | {r['base_act']:.4f} | {r['treat_act']:.4f} | "
                         f"{r['d_act']:+.4f} |")
    if bucketize:
        lines.append("\n**By difficulty bucket** (mean Δ over tasks; Δ<0 = retrieval helps):\n")
        lines.append("| bucket | tasks | mean Δdyn | mean Δact |")
        lines.append("|---|---|---|---|")
        for bname, bts in BUCKETS.items():
            rows = [_row(treat, base, k) for k in bts]
            rows = [r for r in rows if r]
            if not rows:
                continue
            md = sum(r["d_dyn"] for r in rows) / len(rows)
            ma = sum(r["d_act"] for r in rows) / len(rows)
            lines.append(f"| {bname} | {len(rows)} | {md:+.4f} | {ma:+.4f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir
    e1t, e1b = _load(d, "eval1_treatment"), _load(d, "eval1_baseline")
    e2t, e2b = _load(d, "eval2_treatment"), _load(d, "eval2_baseline")

    lines = ["# DreamZero RoboTwin RICL — held-out loss results",
             "\nTeacher-forced flow-matching loss on held-out frames (lower = better). "
             "Plain = stock DreamZero finetune; RICL = retrieved-frame in-context. "
             "Paired noise seed per frame.\n"]
    _section(lines, "Eval 1 — TRAIN tasks, held-out episodes (in-distribution)", e1t, e1b)
    _section(lines, "Eval 2 — 10 UNSEEN tasks (new-skill test, within-task retrieval)",
             e2t, e2b, bucketize=True)
    md = "\n".join(lines) + "\n"
    out = args.out or os.path.join(d, "RESULTS.md")
    with open(out, "w") as f:
        f.write(md)
    print(md)
    print(f"[aggregate] wrote {out}")


if __name__ == "__main__":
    main()
