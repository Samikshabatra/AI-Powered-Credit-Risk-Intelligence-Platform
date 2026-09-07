# deploy_artifacts

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
