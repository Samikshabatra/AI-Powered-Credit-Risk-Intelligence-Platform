"""Build the committed `deploy_artifacts/` sample. Run locally, WITH the data.

    python scripts/prepare_streamlit_artifacts.py
    python scripts/prepare_streamlit_artifacts.py --rows 15000 --seed 7

Streamlit Community Cloud clones the repo and runs `streamlit run app/ui.py`.
There is no Kaggle download, no Docker build and no pipeline step on that host,
so whatever the deployed app shows has to already be in git. That cannot be the
real portfolio: 307,511 real applicant records and a 148 MB warehouse do not
belong in a public repository.

So this script builds a small, self-contained, anonymised replica and runs the
**real pipeline** over it - the same `build_database`, `run_training`,
`extract_rules`, `run_full_eda` and fairness `build_report` the full run uses.
Nothing here reimplements a model or hand-writes a metric; the demo is the
platform, executed over less data.

Anonymisation
-------------
On by default. `SK_ID_CURR` is a real Home Credit applicant identifier; every
sampled id is replaced with a synthetic one from a range the source data never
uses, and the same mapping is applied to the bureau and previous-application rows
so the joins still resolve. Nothing else in the table identifies a person - the
Home Credit release carries no names, addresses or dates.

The dataset is a public Kaggle release, so this is a default rather than a
requirement: `--keep-ids` ships the original identifiers, which lets anyone
cross-check a demo applicant against the competition data. What is *not*
optional is the sampling - GitHub rejects files over 100 MB, and the full
warehouse is 148 MB.

What ships, and what does not
-----------------------------
The sampled CSVs this script writes are working files under `deploy_artifacts/raw/`
and are **git-ignored**: the deployed app never reads raw data. What ships is the
built output - model, calibrator, preprocessor, report JSON, figures, the sampled
SQLite warehouse and the scored holdout parquet.

The demo model is trained on ~5% of the data. Its ROC-AUC is genuinely lower than
the shipped model's and the manifest records both, so nobody reads the demo's
numbers as the project's headline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# `src.utils.config` builds its settings singleton at import time, so the path
# overrides have to be in the environment before the first import of anything
# under `src`. Hence the bare Path arithmetic here rather than `deploy_paths()`
# - which is imported straight afterwards and asserted to agree, so the layout
# still has exactly one definition.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEPLOY_ROOT = REPO_ROOT / "deploy_artifacts"
# The sampled CSVs and the joined parquet cache are build inputs, not deliverables.
WORK_RAW = DEPLOY_ROOT / "raw"
WORK_PROCESSED = DEPLOY_ROOT / "_work" / "processed"

os.environ["DATA_DIR"] = str(DEPLOY_ROOT)
os.environ["RAW_DIR"] = str(WORK_RAW)
os.environ["PROCESSED_DIR"] = str(WORK_PROCESSED)
os.environ["MODEL_DIR"] = str(DEPLOY_ROOT / "models")
os.environ["REPORT_DIR"] = str(DEPLOY_ROOT / "reports")
os.environ["SQLITE_PATH"] = str(DEPLOY_ROOT / "credit_risk.db")
os.environ.setdefault("LOG_LEVEL", "INFO")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.config import PROJECT_ROOT, deploy_paths, settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

DEPLOY = deploy_paths()
assert DEPLOY["data_dir"] == DEPLOY_ROOT, "deploy layout moved in config.py"
assert settings.model_dir == DEPLOY["model_dir"], "path overrides did not take"
assert settings.report_dir == DEPLOY["report_dir"], "path overrides did not take"
assert settings.sqlite_path == DEPLOY["sqlite_path"], "path overrides did not take"

logger = get_logger("prepare_streamlit_artifacts")

# Source data, read from the ordinary local layout rather than the overridden one.
SOURCE_RAW = PROJECT_ROOT / "data" / "raw"
SOURCE_REPORTS = PROJECT_ROOT / "reports"

# Real SK_ID_CURR values run 100001-456255. Starting the synthetic range an order
# of magnitude above that makes a demo id unmistakable on sight.
SYNTHETIC_ID_BASE = 9_000_001

# Recorded LLM output the deployed app replays when it has no key or has hit the
# demo budget. Copied, never regenerated - regenerating needs a key.
CACHED_REPORTS = ("nl_sql_eval.json", "token_report.json")


# --------------------------------------------------------------------------- #
# 1. Sample and anonymise
# --------------------------------------------------------------------------- #
def stratified_sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """A `rows`-row sample that keeps the 8.07% default rate intact.

    Stratifying on TARGET is not cosmetic: the base rate sets `scale_pos_weight`,
    the isotonic calibration and the cost-optimal threshold. A sample that drifted
    off it would produce a demo whose decisions differ from the real system's for
    a reason that has nothing to do with the model.
    """
    if rows >= len(frame):
        return frame.copy()
    fraction = rows / len(frame)
    parts = [
        group.sample(n=max(int(round(len(group) * fraction)), 1), random_state=seed)
        for _, group in frame.groupby("TARGET", sort=True)
    ]
    sampled = pd.concat(parts)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_id_map(identifiers: np.ndarray, seed: int) -> dict[int, int]:
    """Real applicant id -> synthetic id, in a shuffled order.

    Shuffled so the new id carries no information about the old one: a sorted
    mapping would preserve the original ordering and, with the public Kaggle file
    in hand, make the join back trivial.
    """
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(identifiers))
    return {
        int(original): SYNTHETIC_ID_BASE + int(position)
        for original, position in zip(identifiers, order)
    }


def write_sampled_application(
    rows: int, seed: int, anonymise: bool = True
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Sample application_train.csv, renumber the ids, write it to the work dir."""
    source = SOURCE_RAW / "application_train.csv"
    if not source.exists():
        raise SystemExit(
            f"{source} not found. This script runs on the machine that has the "
            "Kaggle data; the deployed app never needs it."
        )

    logger.info("Reading %s", source.name)
    full = pd.read_csv(source)
    logger.info("Full application table: %s rows", f"{len(full):,}")

    sampled = stratified_sample(full, rows, seed)
    original_ids = set(sampled["SK_ID_CURR"].astype(int))
    if anonymise:
        id_map = build_id_map(sampled["SK_ID_CURR"].to_numpy(), seed)
        sampled["SK_ID_CURR"] = sampled["SK_ID_CURR"].map(id_map).astype("int64")
        assert not (set(sampled["SK_ID_CURR"]) & original_ids), "real ids survived"
    else:
        # Identity map: the child tables still go through the same filter/remap
        # path, so there is one code path whether or not ids are rewritten.
        id_map = {identifier: identifier for identifier in original_ids}

    WORK_RAW.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(WORK_RAW / "application_train.csv", index=False)
    logger.info("Sampled applications: %s rows, default rate %.4f, ids %s",
                f"{len(sampled):,}", float(sampled["TARGET"].mean()),
                "renumbered" if anonymise else "kept as published")
    return sampled, id_map


def write_sampled_child_table(
    filename: str, id_map: dict[int, int], chunk_rows: int = 500_000
) -> int:
    """Filter bureau.csv / previous_application.csv to the sample, remapping ids.

    Read in chunks: previous_application.csv is 405 MB and this script has to run
    comfortably on the same laptop that trains the model.
    """
    source = SOURCE_RAW / filename
    if not source.exists():
        logger.warning("%s absent - the sample will train without its aggregates",
                       filename)
        return 0

    target = WORK_RAW / filename
    kept = 0
    first = True
    for chunk in pd.read_csv(source, chunksize=chunk_rows):
        chunk = chunk[chunk["SK_ID_CURR"].isin(id_map)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["SK_ID_CURR"] = chunk["SK_ID_CURR"].map(id_map).astype("int64")
        chunk.to_csv(target, index=False, mode="w" if first else "a", header=first)
        first = False
        kept += len(chunk)
    if first:  # nothing matched; leave no half-written file behind
        target.unlink(missing_ok=True)
    logger.info("%s -> %s rows for the sampled applicants", filename, f"{kept:,}")
    return kept


# --------------------------------------------------------------------------- #
# 2. Run the real pipeline over the sample
# --------------------------------------------------------------------------- #
def run_pipeline(seed: int) -> dict:
    """build_database -> train -> rules -> EDA -> fairness, unmodified."""
    from src.data.build_database import build_database, table_row_counts
    from src.data.eda_insights import run_full_eda
    from src.ml.train import run_training
    from src.monitoring.fairness import build_report
    from src.rules.rule_extractor import extract_rules

    settings.ensure_dirs()

    logger.info("[1/5] SQLite warehouse")
    build_database(force=True)

    logger.info("[2/5] Training")
    metrics = run_training(sample=None, seed=seed)

    logger.info("[3/5] Policy rules")
    extract_rules()

    logger.info("[4/5] EDA")
    run_full_eda()

    logger.info("[5/5] Fairness and stability")
    build_report()

    return {"metrics": metrics, "tables": table_row_counts()}


def publish_holdout() -> Path:
    """Move the holdout snapshot out of the work dir into what git tracks."""
    source = WORK_PROCESSED / "holdout.parquet"
    if not source.exists():
        raise SystemExit(f"Training did not write {source}")
    destination = DEPLOY["processed_dir"] / "holdout.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    logger.info("holdout.parquet -> %s", destination.relative_to(PROJECT_ROOT))
    return destination


def copy_cached_llm_reports() -> list[str]:
    """Carry the measured LLM reports over. They cannot be rebuilt without a key."""
    copied = []
    for name in CACHED_REPORTS:
        source = SOURCE_REPORTS / name
        if not source.exists():
            logger.warning("%s absent - the demo assistant will have nothing to "
                           "replay. Run `python -m src.talk_to_data.eval_harness` "
                           "with a key first.", name)
            continue
        shutil.copy2(source, DEPLOY["report_dir"] / name)
        copied.append(name)
    return copied


# --------------------------------------------------------------------------- #
# 3. Record what was built
# --------------------------------------------------------------------------- #
def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def shipped_files() -> list[dict]:
    """Every file that will be committed, with its size, largest first."""
    ignored = {WORK_RAW, DEPLOY_ROOT / "_work"}
    entries = []
    for path in sorted(DEPLOY["data_dir"].rglob("*")):
        if not path.is_file() or any(parent in ignored for parent in path.parents):
            continue
        entries.append({
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
        })
    return sorted(entries, key=lambda e: -e["bytes"])


def write_manifest(rows: int, seed: int, result: dict, elapsed: float,
                   cached: list[str], anonymised: bool = True) -> Path:
    metrics = result["metrics"]
    files = shipped_files()
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_seconds": round(elapsed, 1),
        "sample": {
            "requested_rows": rows,
            "actual_rows": metrics["n_rows"],
            "seed": seed,
            "stratified_on": "TARGET",
            "anonymised": anonymised,
            "anonymisation": (
                "SK_ID_CURR replaced with shuffled synthetic ids from "
                f"{SYNTHETIC_ID_BASE:,}; the same map applied to bureau and "
                "previous_application rows"
            ) if anonymised else "SK_ID_CURR kept as published by Kaggle",
        },
        "model": {
            "holdout_roc_auc": metrics["discrimination"]["holdout"]["roc_auc"],
            "holdout_pr_auc": metrics["discrimination"]["holdout"]["pr_auc"],
            "threshold": metrics["decision"]["threshold"],
            "split_sizes": metrics["split_sizes"],
        },
        "database_rows": result["tables"],
        "cached_llm_reports": cached,
        "shipped_bytes": sum(entry["bytes"] for entry in files),
        "files": files,
    }
    path = DEPLOY["data_dir"] / "build_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


README = """# deploy_artifacts

The Streamlit Community Cloud build. **Generated - do not edit by hand.**

    python scripts/prepare_streamlit_artifacts.py

Streamlit Cloud has no Kaggle data and no Docker build step, so the deployed app
reads everything from here instead of `data/`, `models/` and `reports/`. Set
`STREAMLIT_CLOUD=1` in the app's Secrets and `src/utils/config.py` repoints every
path at this folder; nothing else in the codebase knows about it.

The contents are the output of the real pipeline run over an anonymised
stratified sample of the portfolio. `SK_ID_CURR` values are synthetic, and the
sampled CSVs used to build this are git-ignored - no Home Credit source data is
committed.

**The model here is not the shipped model.** It is trained on the sample, so its
ROC-AUC is lower. `build_manifest.json` records the sample size, the seed and the
demo model's own metrics.
"""


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the committed Streamlit Cloud sample.")
    parser.add_argument("--rows", type=int, default=18_000,
                        help="applications in the sample (default 18000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="defaults to settings.random_seed")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the sampled CSVs and the joined parquet cache")
    parser.add_argument("--keep-ids", action="store_true",
                        help="ship the published SK_ID_CURR values instead of "
                             "synthetic ones")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else settings.random_seed
    started = time.perf_counter()

    logger.info("Building deploy_artifacts/ from a %s-row sample (seed %d)",
                f"{args.rows:,}", seed)
    for directory in (DEPLOY["model_dir"], DEPLOY["report_dir"],
                      DEPLOY["processed_dir"], WORK_RAW, WORK_PROCESSED):
        directory.mkdir(parents=True, exist_ok=True)

    _, id_map = write_sampled_application(args.rows, seed,
                                          anonymise=not args.keep_ids)
    write_sampled_child_table("bureau.csv", id_map)
    write_sampled_child_table("previous_application.csv", id_map)

    result = run_pipeline(seed)
    publish_holdout()
    cached = copy_cached_llm_reports()

    (DEPLOY["data_dir"] / "README.md").write_text(README, encoding="utf-8")
    manifest = write_manifest(args.rows, seed, result,
                              time.perf_counter() - started, cached,
                              anonymised=not args.keep_ids)

    if not args.keep_work:
        shutil.rmtree(WORK_RAW, ignore_errors=True)
        shutil.rmtree(DEPLOY_ROOT / "_work", ignore_errors=True)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    logger.info("=" * 68)
    logger.info("Committed size: %.1f MB across %d files",
                payload["shipped_bytes"] / 1e6, len(payload["files"]))
    for entry in payload["files"][:8]:
        logger.info("  %8.2f MB  %s", entry["bytes"] / 1e6, entry["path"])
    logger.info("Demo model holdout ROC-AUC %.4f on %s rows",
                payload["model"]["holdout_roc_auc"], f"{payload['sample']['actual_rows']:,}")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
