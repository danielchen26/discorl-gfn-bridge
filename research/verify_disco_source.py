#!/usr/bin/env python3
"""Re-verify every claim this dossier makes about the DiscoRL source.

The dossier asserts specific things about google-deepmind/disco_rl -- that the
per-trajectory recurrence runs backwards, that all three agent losses are
categorical KLs, that the meta-network is fed consecutive (t, t+1) pairs of its
own y and z predictions, and -- the load-bearing negative claim -- that it is
never fed a backward policy.  Claims about someone else's code rot silently, so
they are encoded here as executable assertions against a pinned commit.

Usage:  python3 research/verify_disco_source.py [--json out.json]
Exit code 0 means every claim in the dossier still holds at COMMIT.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

REPO = "google-deepmind/disco_rl"
COMMIT = "9059a29f7121d60948f25ef165e08e050e9399c8"  # 2025-12-02
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"

FILES = {
    "disco": "disco_rl/update_rules/disco.py",
    "meta_nets": "disco_rl/networks/meta_nets.py",
}


def fetch(path: str) -> list[str]:
    ctx = None
    try:
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    except Exception:  # pragma: no cover
        pass
    with urllib.request.urlopen(f"{RAW}/{path}", context=ctx, timeout=60) as fh:
        return fh.read().decode().splitlines()


def find(lines: list[str], pattern: str) -> list[int]:
    rx = re.compile(pattern)
    return [i + 1 for i, ln in enumerate(lines) if rx.search(ln)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    src = {k: fetch(v) for k, v in FILES.items()}
    checks: list[dict] = []

    def check(cid, claim, ok, where, evidence):
        checks.append(
            {
                "id": cid,
                "claim": claim,
                "ok": bool(ok),
                "file": where[0],
                "lines": where[1],
                "evidence": evidence,
            }
        )

    # ---- S1: the per-trajectory core is unrolled in reverse.
    rev = find(src["meta_nets"], r"reverse=True")
    cmt = find(src["meta_nets"], r"reverse direction for bootstrapping")
    check(
        "S1",
        "The per-trajectory recurrence is unrolled backwards along the trajectory, "
        "and the source comment states the purpose is bootstrapping.",
        len(rev) == 1 and len(cmt) == 1 and 0 < rev[0] - cmt[0] < 12,
        (FILES["meta_nets"], cmt + rev),
        src["meta_nets"][cmt[0] - 1].strip() if cmt else "",
    )

    # ---- S2: all three agent losses are categorical KLs, so y and z are
    #          distribution-valued, not scalar-valued.
    kl = find(src["disco"], r"rlax\.categorical_kl_divergence")
    names = [src["disco"][i - 1].split("=")[0].strip() for i in kl]
    check(
        "S2",
        "pi, y and z are all trained with categorical KL divergence: the "
        "predictions are distributions over bins, not scalars.",
        len(kl) >= 3
        and any("pi_loss" in n for n in names)
        and any("y_loss" in n for n in names)
        and any("z_loss" in n for n in names),
        (FILES["disco"], kl),
        " / ".join(names),
    )

    # ---- S3: the meta-network sees consecutive pairs of its own predictions.
    y_pair = [
        i
        for i in find(src["disco"], r"td_pair")
        if any("agent_out/y" in ln for ln in src["disco"][max(0, i - 4) : i])
    ]
    z_avg = find(src["disco"], r"pi_weighted_avg")
    z_max = find(src["disco"], r"'max_a'")
    check(
        "S3",
        "The meta-network receives (y_t, y_t+1) pairs and both the policy-weighted "
        "average and the max over actions of z at t and t+1 -- the sufficient "
        "statistics of a local balance condition.",
        bool(y_pair) and bool(z_avg) and bool(z_max),
        (FILES["disco"], sorted(y_pair + z_avg + z_max)),
        f"td_pair@y={y_pair} pi_weighted_avg={z_avg} max_a={z_max}",
    )

    # ---- S4: THE NEGATIVE CLAIM. No backward policy anywhere in the inputs.
    #          This is what bounds DiscoRL's hypothesis class away from a
    #          general learned p_B on a multi-parent DAG.
    bwd_terms = r"(p_b|p_B|backward_policy|backward_pol|parent|predecessor)"
    hits = [
        (i, src["disco"][i - 1].strip())
        for i in find(src["disco"], bwd_terms)
        if "should_reset" not in src["disco"][i - 1]
    ]
    check(
        "S4",
        "No backward policy and no parent-set information is ever fed to the "
        "meta-network; the only non-forward signal is the reverse recurrence "
        "over the realised trajectory suffix.",
        len(hits) == 0,
        (FILES["disco"], [i for i, _ in hits]),
        "no matches" if not hits else "; ".join(t for _, t in hits[:4]),
    )

    # ---- S5: two time scales, combined multiplicatively.
    mult = find(src["meta_nets"], r"_multiplicative_interaction\(")
    life = find(src["meta_nets"], r"throughout the agent's lifetime")
    check(
        "S5",
        "Two recurrences exist -- backward within a trajectory and forward across "
        "the agent's lifetime -- and they are combined multiplicatively.",
        len(mult) >= 2 and bool(life),
        (FILES["meta_nets"], sorted(set(mult + life))),
        f"multiplicative_interaction@{mult} lifetime_doc@{life}",
    )

    ok = all(c["ok"] for c in checks)
    width = max(len(c["id"]) for c in checks)
    print(f"{REPO} @ {COMMIT[:12]}\n")
    for c in checks:
        mark = "PASS" if c["ok"] else "FAIL"
        print(f"[{mark}] {c['id']:<{width}}  {c['file']}:{c['lines']}")
        print(f"        {c['claim']}")
        print(f"        evidence: {c['evidence']}\n")
    print("ALL CLAIMS HOLD" if ok else "SOME CLAIMS BROKE -- update the dossier")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"repo": REPO, "commit": COMMIT, "checks": checks, "ok": ok}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
