<h1 align="center">AI-Powered Credit Risk Intelligence Platform</h1>

<p align="center">
  <b>Not just a model — a credit <i>decision system</i>:<br>
  scored, explained, defensible, auditable, and queryable in plain English.</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%20%E2%80%93%203.13-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/LightGBM-4.7-02569B?style=flat-square">
  <img alt="Claude" src="https://img.shields.io/badge/Claude-Sonnet%20%2B%20Haiku-D97757?style=flat-square&logo=anthropic&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-compose-2496ED?style=flat-square&logo=docker&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-141%20passing-16a34a?style=flat-square">
</p>

<p align="center">
  <a href="https://ai-powered-credit-risk-intelligence-platform-8eftb3rvzqybqg2pj.streamlit.app/"><img alt="Live demo" src="https://img.shields.io/badge/%E2%96%B6%20Live%20demo-open%20the%20app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"></a>
</p>

<p align="center">
  <sub>The hosted demo runs on an 18,000-row anonymised sample, so its metrics sit under
  the full-run figures reported here — <a href="#streamlit-community-cloud">why that is</a>.
  Scoring, explanations, policy rules, EDA and fairness all work there with no API key.</sub>
</p>

---

An explainable, agentic credit-risk platform built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset (307,511 applications). It spans the full stack a bank would actually need: data engineering → EDA → calibrated ML → cost-optimised decisioning → SHAP explanations → adverse-action reason codes → derived credit policy → a self-correcting natural-language SQL interface → an orchestrator agent → a Streamlit UI → Docker.

**Three things make it a decision system rather than a notebook:**

| | |
|---|---|
| 🎯 **The threshold is chosen, not assumed** | 0.50 is a coin-flip dressed as a policy. The cut-off here minimises expected loss against per-applicant exposure — LGD 45%, margin 8% — and is picked on a split used for nothing else. That single change is worth **$221M** on the holdout portfolio. |
| 🧾 **Every decline can be explained to the applicant** | SHAP attribution is turned into ranked, ECOA-shaped principal reasons, with protected attributes excluded from the rule language and any suppression reported rather than hidden. |
| 🛡️ **The AI is bounded by mechanism, not manners** | Generated SQL passes a statement allowlist, a keyword denylist, a table allowlist and a read-only connection. A prompt-injected question fails on all four — measured across 25 labelled cases, injection attempt included. |

**Every LLM feature has a deterministic twin.** Scoring, explanation, policy rules, EDA and fairness all run with no API key. Only the chat assistant needs one, and it degrades to a labelled cached answer rather than an error.

## Contents

- [Headline results](#headline-results)
- [Quick start](#quick-start) — [Docker](#docker-what-an-evaluator-should-run) · [Local](#local) · [Streamlit Cloud](#streamlit-community-cloud) · [Without an API key](#without-an-api-key)
- [Architecture](#architecture) · [The Decision Trace](#the-decision-trace)
- [Module walkthrough](#module-walkthrough) — EDA, ML, explainability, rules, NL→SQL, agent, UI, fairness
- [Token optimisation, measured](#token-optimisation-measured)
- [Model & stack choices](#model--stack-choices)
- [Testing](#testing) · [Repository layout](#repository-layout) · [Configuration](#configuration)

---

## Headline results

| | Result | Where it comes from |
|---|---|---|
| **Discrimination** | ROC-AUC **0.7788**, PR-AUC **0.2637**, KS **0.415** | 61,503-row holdout, never seen in training |
| **Calibration** | Brier **0.1604 → 0.0663**, ECE **0.2687 → 0.0039** | Isotonic, fitted on a split used for nothing else |
| **Decision** | Cost-optimal threshold **0.129**, not 0.50 | Minimises expected loss on per-applicant exposure |
| **Business impact** | **$221M** expected loss avoided vs a naive 0.50 threshold; **$250M** vs approving everyone | Holdout portfolio, LGD 45% / margin 8% |
| **Policy** | **14 IF-THEN rules**, top rule at **3.50× lift** on 5.1% of applicants | Depth-4 surrogate, 81.2% fidelity to the model |
| **Governance** | Approval-rate gap **10.7pp** by gender, **30.5pp** by age band | Measured and published, not silently absorbed |
| **NL→SQL** | **19/21** answerable questions correct, **4/4** refusals and the injection attempt caught, 0 retries, 85.2% input-token saving (73.2% cold-start) | Live 25-case run, `eval_harness` |
| **Tests** | **141 passing** | `pytest tests/` |

Rebuild every number: `python -m src.ml.train` writes `reports/model_metrics.json`.

<sub>LGD 45% and margin 8% are plausible retail-lending figures, not a specific lender's. The threshold and every dollar figure move with them, so the <i>method</i> is the deliverable — swap in a real cost matrix through <code>LGD_RATE</code> / <code>MARGIN_RATE</code> and the whole chain re-derives.</sub>

---

## Quick start

### Docker (what an evaluator should run)

```bash
git clone https://github.com/Samikshabatra/AI-Powered-Credit-Risk-Intelligence-Platform.git
cd AI-Powered-Credit-Risk-Intelligence-Platform
cp .env.example .env                 # optional: add ANTHROPIC_API_KEY for the LLM features

# Download the dataset from Kaggle and unzip these four files into ./data/raw/
#   application_train.csv  application_test.csv  bureau.csv  previous_application.csv

docker compose up
```

Open <http://localhost:8501>. The entrypoint builds the SQL warehouse, runs the EDA, trains the model, derives the rules and generates the fairness report on first start (~3 minutes), then serves the UI. Artifacts land on mounted volumes, so a restart is instant.

### Local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

python -m src.data.build_database      # SQLite warehouse        (~10s)
python -m src.data.eda_insights        # EDA figures + summary   (~15s)
python -m src.ml.train                 # model + calibration     (~90s)
python -m src.rules.rule_extractor     # surrogate policy rules  (~5s)
python -m src.monitoring.fairness      # group performance       (~10s)

streamlit run app/ui.py
```

Two optional extras, both of which read the artifacts above rather than hardcoding anything:

```bash
python documents/build_presentation.py       # regenerates the 14-slide PDF deck
python -m src.talk_to_data.token_report      # token-optimisation measurement
python -m src.talk_to_data.eval_harness      # NL->SQL accuracy (needs an API key)
```

### Streamlit Community Cloud

**Live: [ai-powered-credit-risk-intelligence-platform-8eftb3rvzqybqg2pj.streamlit.app](https://ai-powered-credit-risk-intelligence-platform-8eftb3rvzqybqg2pj.streamlit.app/)**

The public build runs from `deploy_artifacts/` — a small, anonymised, fully built
copy of the platform that is committed to the repository. Streamlit Cloud clones
the repo and runs `streamlit run app/ui.py`; there is no Kaggle download, no
Docker build and no pipeline step on that host, so anything the deployed app
shows has to already be in git. The full warehouse cannot go there in any case:
it is 148 MB and GitHub rejects files over 100 MB.

Rebuild the sample on a machine that has the data:

```bash
python scripts/prepare_streamlit_artifacts.py            # ~18,000 rows
python scripts/prepare_streamlit_artifacts.py --rows 40000 --keep-ids
```

It samples `application_train.csv` stratified on TARGET, renumbers `SK_ID_CURR`
to synthetic ids (`--keep-ids` leaves the published ones — the dataset is a
public Kaggle release), filters `bureau` and `previous_application` to the
sample, then runs the **real** pipeline over it: `build_database`,
`run_training`, `extract_rules`, `run_full_eda` and the fairness report. Nothing
is stubbed, so the demo is the platform on less data — and its ROC-AUC is
correspondingly lower than the headline. `deploy_artifacts/build_manifest.json`
records the sample size, the seed, the demo model's own metrics and every
committed file with its size.

Deploying:

| Piece | Why |
|---|---|
| `packages.txt` → `libgomp1` | LightGBM's OpenMP runtime. An apt package; pip cannot supply it |
| `STREAMLIT_CLOUD=1` in Secrets | Repoints every artifact, database and holdout path at `deploy_artifacts/` |
| `DEMO_MODE=1` in Secrets | Caps model calls per browser session so a public key cannot be drained |
| `ANTHROPIC_API_KEY` in Secrets | Optional — see below |

`app/ui.py` copies `st.secrets` into `os.environ` before `settings` is built,
which is how pydantic-settings sees them on a host with no `.env` file. See
`.streamlit/secrets.toml.example`.

### Without an API key

Everything except the chat assistant and NL→SQL runs offline: EDA, scoring, the Decision Trace, SHAP, adverse-action reason codes, business rules and the fairness report. The UI shows which capabilities are live in the sidebar. This is a design property, not a fallback — see *Deterministic twins* below.

---

## Architecture

```
                        data/raw/*.csv  (mounted, never committed)
                                 |
        +------------------------+------------------------+
        |                                                 |
   loader.py                                       build_database.py
   join bureau + previous_application               curated 39-column warehouse
        |                                                 |
   preprocessor.py                                  SQLite (read-only at query time)
   sentinel repair, 19 engineered features                |
        |                                                 |
   train.py -----> models/*.joblib                  query_runner.py
   LightGBM + isotonic + cost threshold             4 guardrail layers
        |                                                 |
        +--------+--------+---------+                nl_to_sql.py
                 |        |         |                Sonnet, self-correcting
          predict.py  shap_      rule_                    |
                      explainer  extractor           eval_harness.py
                 |        |         |                25 labelled cases
                 |   reason_codes   |                     |
                 |        |         |                     |
                 +--------+----+----+---------------------+
                               |
                        agent/orchestrator.py
                        AI tool-use over all six tools
                               |
                          app/ui.py
                 9 sections, Decision Trace hero
```

### The Decision Trace

The hero screen. One applicant, the whole chain, top to bottom:

```
CREDIT DECISION TRACE — applicant 170169
  1. Feature engineering    143 features, 19 engineered, sentinel repaired
  2. Risk model             raw score 0.859   (LightGBM, 1190 trees, spw 11.39)
  3. Calibration            38.3% probability of default
  4. Decision threshold     decline at or above 12.9%   (cost-optimal)
  5. Risk band              HIGH
  6. Explanation            external scores below average; long implied term;
                            75% of previous applications refused
  7. Policy rule            IF avg external score <= 0.26 AND weakest <= 0.18
                            THEN DECLINE  (support 5.1%, default rate 28.2%, lift 3.50x)
  --> DECLINE
```

Every stage is a tested module. The page is the presentation layer that renders them in sequence.

---

## Module walkthrough

### 1 — EDA (`notebooks/eda.ipynb`, `src/data/eda_insights.py`)

The analysis lives in a module the notebook *and* the Streamlit tab both import, so they cannot drift apart.

Five business insights, each with a chart and a consequence for the model:

| Insight | Finding | Consequence |
|---|---|---|
| Income bracket | 3pp spread across the whole income range | Income is a weak signal; bureau scores dominate |
| Contract type | Cash 8.35% vs revolving 5.48%, and cash loans are ~2× larger | Cost-weight the threshold by `AMT_CREDIT`, not by count |
| Age band | 12.3% under 25 vs 4.9% over 60, monotone | Predictive *and* protected — suppress from the notice, measure the gap |
| External scores | ~6× spread best-to-worst decile, strongest signal in the data | Engineer mean/min/max/count/disagreement; `EXT_SOURCE_1` is 56% missing |
| Employment length | 11.0% under a year → 5.2% past ten years | Tenure matters, but see below |

**The finding worth reading.** `DAYS_EMPLOYED = 365243` appears on 55,374 rows (18.0%) and decodes to 1000 years of employment. It is a "not employed" sentinel. Most write-ups stop at flagging it as dirty data — but 99.96% of that group are **pensioners**, and they default at **5.4%**, *below* the 8.7% employed average. Bin them into "10 years+" and the trend moves the wrong way for the wrong reason; drop them and 18% of the portfolio disappears. Split them out and they are one of the safer segments in the book.

The repair is pinned in three places so it cannot leak back: `preprocessor.clean()`, the `employment_years` metric in the semantic layer, and the EDA chart.

**Data-quality scorecard.** Completeness per feature group, plus anomalies that are *not* nulls: the sentinel, `CODE_GENDER = 'XNA'`, and a 117M income outlier at 795× the median.

### 2 — ML (`src/ml/`)

**Split design.** Four disjoint stratified splits, because each step needs data the previous one did not touch:

| Split | Share | Used for |
|---|---|---|
| train | 60% | Fitting the booster |
| valid | 10% | Early stopping |
| calib | 10% | Isotonic calibrator **and** threshold selection |
| holdout | 20% | Every headline number; the UI applicant pool |

Fitting the calibrator on the early-stopping split, or choosing the threshold on the split used to report savings, both produce numbers that do not survive contact with production.

**Class imbalance.** 8.07% positives → `scale_pos_weight = 11.39`. SMOTE was rejected explicitly: synthesising 260k minority rows in 143 dimensions is slow, adds no information a tree cannot find through reweighting, and interpolates between categorical *codes* to produce applicants that cannot exist.

**Calibration is not optional.** Reweighting fixes the imbalance and wrecks the probabilities as a side effect — the raw model predicts ~46% average risk against an actual 8.1%. Isotonic regression, fitted on the calibration split, takes Brier from 0.1604 to 0.0663 and ECE from 0.2687 to 0.0039.

One honest caveat: isotonic is monotone *non-decreasing*, so it never reorders two applicants, but it does collapse 61k distinct scores onto ~150 steps. Those ties move ROC-AUC from 0.7792 to 0.7788. "Ranking preserved, so AUC unchanged" is the almost-true version; −0.0004 is the real one.

**Versus what?** The same 143 features, the same train split, the same seed, the same untouched holdout — only the learner changes:

| Model | Holdout ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression (balanced, scaled) | 0.7605 | 0.2474 |
| Random forest (balanced, 300 × depth 12) | 0.7557 | 0.2430 |
| **LightGBM (shipped)** | **0.7788** | **0.2637** |

LightGBM wins because it reads `NaN` and categorical codes natively while both baselines need the matrix median-imputed and the codes treated as numbers, and because it captures interactions — `EXT_SOURCE_3` mattering far more for a young unemployed applicant than for an employed one — that a linear model cannot express and a depth-capped forest finds less efficiently. The PR-AUC gap is the one that counts: at an 8% base rate that is the metric that reflects the top of the ranking, which is where the decision is actually made. Regenerated by `python -m src.ml.train` into `reports/model_metrics.json` (`baselines` block) and `reports/figures/ml_baseline_comparison.png`.

**Cost-sensitive threshold — the strongest single differentiator.** A false negative costs **45% of the loan** (loss given default); a false positive costs **8%** (forgone margin). Both are weighted by that applicant's actual `AMT_CREDIT`, because declining a 2M loan is not the same mistake as declining a 50k one.

| Policy | Threshold | Approval rate | Expected loss | Saving |
|---|---|---|---|---|
| Approve everyone | — | 100% | $1.25B | $250.5M |
| Naive 0.50 | 0.500 | 96.9% | $1.22B | $221.4M |
| **Cost-optimal** | **0.129** | **81.4%** | **$998.4M** | — |

**Risk bands** are anchored on the threshold, not on round numbers: High ≥ 12.9% (where the economics say decline), Medium ≥ 6.4%, Low below that.

### 3 — Explainability (`src/xai/`)

SHAP `TreeExplainer` — exact for tree ensembles, milliseconds per applicant, so it runs inline in the app.

**Stated caveat.** SHAP explains the raw booster output in log-odds space, not the calibrated probability. Calibration is monotone, so the *sign and ranking* of every contribution carry over unchanged; the magnitudes do not. The UI therefore shows contributions as a share of total attribution, never as "+7% of default probability" — a number the method cannot support.

**Adverse-action reason codes.** Under ECOA / Regulation B a lender that declines must give the *principal reasons*, specifically and accurately. Two layers, in this order:

1. **Deterministic templates.** Every reason code is generated from the SHAP contribution and the applicant's own value by rule — auditable, reproducible, works with no API key.
2. **AI phrasing (optional).** The cheap model restates those bullets as a short notice. It receives the reason codes as *facts to restate*, never as data to reason about, so it cannot invent a reason the model did not produce.

That ordering *is* the hallucination control: the LLM is a stylist here, not a decision-maker.

Protected attributes (gender, age, family status) are **suppressed from the notice and reported separately**, so a reviewer sees that the model leaned on one rather than having it quietly disappear.

### 4 — Business rules (`src/rules/rule_extractor.py`)

A 1,190-tree ensemble is not credit policy. A depth-4 decision tree fitted to **the model's own decisions** is, and its leaves read back as IF-THEN rules.

Sample output (14 rules, 81.2% surrogate fidelity, 8.06% portfolio base rate):

| Rule | Support | Default rate | Lift |
|---|---|---|---|
| avg external score ≤ 0.26 AND weakest ≤ 0.18 → **DECLINE** | 5.1% | 28.2% | 3.50× |
| 0.26 < avg score ≤ 0.36 AND weakest ≤ 0.18 → **DECLINE** | 4.5% | 18.7% | 2.33× |
| avg score ≤ 0.36 AND weakest > 0.18 AND down-payment cover ≤ 0.89 → **DECLINE** | 3.6% | 16.8% | 2.09× |

Every rule carries **support** (coverage), **precision** (default rate measured against ground truth, not predictions) and **lift**. Two restrictions keep them usable as policy: numeric features only (a split on `OCCUPATION_TYPE <= 7.5` is an encoding artefact), and no protected attributes — a rule set is the one artefact here that a human might apply by hand.

### 5 — Talk to Data (`src/talk_to_data/`)

Natural language → SQL → business answer, with the SQL visible at every step.

```
Sonnet (schema + metrics + exemplars, cached)
  → run_sql(sql)
      → rejected or SQLite error → error text back to Sonnet → rewrite (≤2 retries)
      → success → rows to Haiku → business answer
```

Handing rows to the cheap model rather than back to Sonnet is deliberate: summarisation is the easiest step with the largest payload, and Sonnet's context never grows by the rows, so the cached prefix survives for the next turn.

**Semantic layer** — the reliability lever. Ask an LLM for "the default rate" three times and you can get `AVG(TARGET)`, `SUM(TARGET)*1.0/COUNT(*)` and `COUNT(CASE WHEN...)/COUNT(*)` — the third silently returns integer `0` in SQLite. So 15 metrics are pinned once and injected into every prompt:

| Metric | Canonical SQL |
|---|---|
| `default_rate` | `AVG(TARGET)` |
| `age` | `-DAYS_BIRTH / 365.0` |
| `employment_years` | `CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE -DAYS_EMPLOYED / 365.0 END` |
| `credit_income_ratio` | `AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0)` |
| `approval_rate` | `AVG(CASE WHEN DECISION = 'Approve' THEN 1.0 ELSE 0.0 END)` |

`tests/test_semantic_layer.py` executes every one against the live database — including a test that demonstrates the integer-division trap returning 0.0 and the sentinel producing a −1000-year average, so the layer is a guarantee rather than a promise.

**Hallucination control — four independent layers.** Generated SQL is treated as hostile, so a prompt-injected question fails on *mechanism*, not on the model's good manners:

| # | Guardrail | Stops |
|---|---|---|
| 1 | Statement allowlist (`sqlparse`) | Anything that is not a single SELECT/WITH |
| 2 | Whole-token keyword denylist | INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA (a column named `updated_at` is not a false positive) |
| 3 | Table allowlist | Hallucinated tables, cross-database reads |
| 4 | Read-only connection (`mode=ro`) | Any write that survived 1–3 |

Plus an auto-injected `LIMIT` on non-aggregate queries, a wall-clock interrupt via SQLite's progress handler, schema grounding (DDL read back from `sqlite_master`, so the model cannot invent a column), and a hard fallback: unanswerable questions get `CANNOT_ANSWER`, not a guess.

**Five+ working query patterns:** aggregation, filter + count, group-by, group-by + ranking, derived bands, joins, and queries over the model's own `predictions` table.

### 5b — NL→SQL evaluation (`evaluation/nl_sql_questions.jsonl`)

"Five query patterns work" is a claim. 25 hand-labelled cases turn it into a number:

- Every required pattern, plus the sentinel trap, a **multi-turn follow-up** that only resolves against conversation memory, **three unanswerable questions**, and a **prompt-injection attempt**.
- Each case carries a **ground-truth SQL query, not a hardcoded number**, so expected answers are recomputed from the live database on every run — a rebuilt database cannot silently invalidate the labels.
- Reported: answer accuracy, execution accuracy, fallback accuracy, retry rate, SQL similarity (soft signal only), median latency, tokens per query.

```bash
python -m src.talk_to_data.eval_harness      # needs ANTHROPIC_API_KEY
```

The grader is verified independently of the API: feeding it ground-truth SQL as if the agent had produced it scores 100%, so a failure in a live run is the agent's, not the harness's.

**Measured run — all 25 cases, live against the API:**

| Metric | Result |
|---|---|
| Answer accuracy | **19/21 (90.5%)** on answerable questions |
| Execution accuracy | **21/21 (100%)** — every generated query ran |
| Fallback accuracy | **4/4 (100%)** — three unanswerable questions and the injection attempt were all refused |
| False-fallback rate | **0%** — no answerable question was wrongly refused |
| Retry rate | **0%** — every query was valid first time |
| Median latency | 3.5 s per question |
| SQL similarity to reference | 0.857 (soft signal only) |
| API requests | 48 (26 Sonnet for SQL, 22 Haiku for the prose answer), 85.2% input-token saving from caching |

**The two failures, in full.** Neither is a wrong number — both are the grader matching group *labels* strictly, and both are disclosed rather than graded away:

- **q07 (default rate by education level).** The agent applied `HAVING COUNT(*) >= 500`, which is a hard rule in its own system prompt, and so dropped *Academic degree* (164 applicants). Every group it did return matches the reference to the last decimal. The reference SQL carries no `HAVING`, so the labelled ground truth and the prompt rule disagree with each other. The agent obeyed the prompt.
- **q15 (default rate by prior refusal).** The agent named its buckets `Previously refused` / `Never refused`; the reference names them `refused before` / `never refused`. The default rates are bit-identical (10.3218% vs 7.0732%). The grader keys on the group string, so a synonym reads as a missing group.

Both would pass under a looser key match. Loosening the grader after seeing the results is how evaluations get flattering, so the grader is unchanged and the two cases stay red.

**One harness fix came out of this run.** The first full pass scored 85.7% with three false fallbacks (q15, q17, q25). The cause was the harness, not the agent: it ran all 25 questions through one continuous conversation, and the agent keeps a rolling four-turn memory window, so an unrelated preceding question made it read an independent question as a follow-up and decline. All three answer correctly in isolation. `run_evaluation` now resets memory before each case, and replays a follow-up's antecedent immediately before it — which is also the only way the multi-turn case (`q25`, `depends_on: q01`) tests memory at all, rather than looking for a turn that fell out of the window twenty-three questions ago. The numbers above are from the corrected pass.

### 6 — Orchestrator agent (`src/agent/`)

A hand-rolled AI tool-use loop — no framework. LangChain would add ~80 MB to the image and an abstraction over what is honestly a while-loop around `messages.create`.

Six tools: `query_data`, `predict_risk`, `explain_prediction`, `get_policy_rules`, `list_applicants`, `get_model_performance`. Tools never raise — failures come back as `{"error": ...}` so the model can explain them — and return compact JSON, because whatever they return is echoed into the model's context.

**Deterministic twins.** Every tool the agent can call is also wired to a button elsewhere in the UI. If the router misfires during a demo, the capability is one click away. The agent is an addition, never the only path.

### 7 — UI (`app/ui.py`, `app/theme.py`, `app/charts.py`)

A product dashboard, not a report: a top bar, a left nav rail, and pages built from white cards on a light ground. The design system lives in `theme.py` (tokens, CSS, HTML components) and the Altair builders in `charts.py`, so `ui.py` stays about composition and data.

Eight sections: **Home**, **Predict**, **Talk to Data**, **EDA & Insights**, **Model Performance**, **Business Rules**, **Model Health**, **About**.

**Home explains the platform rather than dashboarding it** — a hero carrying three proof stats read from the artifacts, the KPI strip, a seven-stage diagram of how one decision is produced, and tiles into every section. The default-rate and risk-distribution charts it used to show were moved out: both have detailed homes on EDA & Insights and Model Performance, and repeating them told a first-time viewer nothing about what the application does.

The **Predict** page is the hero: manual input / batch CSV upload / sample applicant, a threshold-anchored risk gauge, principal reasons, the 8-step **Decision Trace**, and the SHAP contributions behind the score.

Manual input **scores live** — there is no submit button. An earlier version parked the applicant in session state on a button press, so editing a field left the result panel showing a stale decision until the button was found and clicked. Two related desyncs were fixed the same way: the input method is a radio rather than `st.tabs` (Streamlit renders every tab body, so all three branches ran and the last one to assign won), and a manually-entered applicant no longer inherits the template row's `SK_ID_CURR` or `TARGET` — a synthetic applicant has neither an identity nor a ground-truth outcome.

**No decorative controls.** The reference comp carried marketing nav, a search field, a notification bell, "Get Started" / "Watch Demo" buttons and a promo panel; none had anywhere to go, so all of them were removed rather than shipped as dead chrome. Every control on screen does something: the Home tiles navigate, and the suggested questions on Talk to Data actually ask the question. The top bar is an identity strip — brand, current section, signed-in user — and is deliberately not clickable.

Two further details worth calling out:

- **The gauge is scaled to the threshold, not to 0–100%.** At an 8% base rate a 0–1 gauge renders every applicant as a sliver; here the decision threshold sits at the centre of the arc, so the visual question is which side of the line the applicant falls on.
- **Manual input starts from the portfolio median.** A form cannot supply 143 features, so untouched fields take the median and the page says so — a manual score is a what-if against a typical applicant, not a score built from six inputs.

Charts are Altair (ships inside Streamlit, so nothing is added to the image) and stay hoverable; the matplotlib PNGs in `reports/figures/` remain the reproducible artifacts the notebook and README cite.

### 8 — Model health & fairness (`src/monitoring/fairness.py`)

| Group | Applicants | Default rate | ROC-AUC | Approval rate |
|---|---|---|---|---|
| Female | 40,561 | 7.0% | 0.772 | 85.0% |
| Male | 20,940 | 10.2% | 0.778 | 74.3% |
| Under 25 | 2,382 | 11.8% | 0.737 | 63.3% |
| 60+ | 7,258 | 5.0% | 0.746 | 93.8% |

Demographic parity gap **10.7pp** by gender and **30.5pp** by age band; equal-opportunity gaps of 14.3pp and 41.4pp. None of these are "fixed" — which one a lender should equalise is a policy and legal question, not a modelling one. The platform measures them and puts them on screen.

**Distribution shift, labelled honestly.** Train-vs-holdout PSI is reported and explicitly **not** called drift: `application_train.csv` has no time axis and both splits are random samples of one snapshot, so max PSI is 0.0004 — near zero by construction. Calling that "no drift detected" would be measurement theatre. What it does prove is that the split is unbiased and the machinery is wired for when time-ordered data arrives.

---

## Token optimisation, measured

`python -m src.talk_to_data.token_report` — and the result includes a lever that did not work.

| Lever | Mechanism | Effect |
|---|---|---|
| **Prompt caching** | `cache_control: ephemeral` on the stable prefix (role + schema + metrics + exemplars) | 2,725 tokens/turn bill at 10% from the second call |
| **Haiku for summarisation** | Result rows go to the cheap model; Sonnet's context never grows by them | ~400 tokens/question at ~1/12 the price |
| **Retry cap of 2** | Bounds the worst case at three model calls | Pathological questions only |
| **Schema curation** | 39 curated columns of 122, pre-aggregated supplementary tables | **≈0%** — see below |

Modelled over a 6-turn session: **18,653 → 5,027 billable input tokens, a 73% reduction.**

**Measured on the live 25-case eval run** (48 API requests, reported by the API itself rather than modelled):

| | Tokens |
|---|---|
| Fresh input | 8,315 |
| Cache **reads** | 148,512 |
| Cache writes | 0 — the prefix was already warm |
| Billable input | **23,166** vs **156,827** uncached-equivalent |
| **Input token saving** | **85.2%** |
| Cache hit rate | 94.7% |

One qualification, because 0 cache writes flatters the number: this pass ran minutes after an earlier one, so the 5,712-token prefix write had already been paid. The cold-start pass — same 25 questions, same prompt, but writing the cache itself — measured **73.2%** saving at a 82.3% hit rate. 73.2% is the honest figure for the first session of the day; 85.2% is the honest figure for every session after it, and the gap between them is exactly the one-off write.

**The lever that failed.** Schema curation was supposed to be a major saving. Measured against the DDL a naive implementation would inject — every column of every source table, built from the real CSV headers — the shipped 39-column schema is *the same size* (2,725 vs 2,699 tokens), because its per-column comment glossary costs back exactly what the dropped columns save. That glossary is what stops the model guessing what `DAYS_EMPLOYED` means, so it stays. Schema curation is a **hallucination-control lever here, not a cost lever**, and the report prints the real number.

> **Status:** `token_report.py`'s prompt-size figures still use a documented ~4-chars-per-token heuristic, and the report says so in its own output; `messages.count_tokens` makes them exact but was not run, to keep API calls to a minimum. The cache figures in the table above needed no extra calls — they come from `usage.cache_read_input_tokens` on the eval run itself.

---

## Model & stack choices

| Choice | Why |
|---|---|
| **LightGBM** | Best-in-class on tabular imbalanced data; native NaN handling (missingness is predictive here); native categorical support avoids a 200-column one-hot blow-up; SHAP-friendly |
| **`scale_pos_weight`, not SMOTE** | 307k rows in 143 dimensions; interpolating between categorical codes produces impossible applicants |
| **Isotonic, not Platt** | The reweighting shift is roughly constant odds (a sigmoid could absorb it) but residual tail miscalibration is not sigmoid-shaped; 30k calibration rows is ample for a non-parametric fit |
| **NaN kept, not imputed** | LightGBM learns a split direction per feature; "no bureau record" and "no external score" are signal |
| **Two-model split (strong + cheap)** | Strong native tool-calling drives the agent; the cheap model handles the high-volume, low-difficulty summarisation step |
| **SQLite** | Zero-config, file-based, trivially containerised, and `mode=ro` is a real security boundary |
| **3 tables, not 7** | `bureau` + `previous_application` aggregates lift AUC meaningfully; the four monthly-balance tables add ~1.9 GB of IO for a marginal gain |
| **No agent framework** | ~80 MB of image for a while-loop |

---

## Testing

```bash
pytest tests/ -q          # 121 passed
```

Two families: **always-on** (guardrails, semantic-layer SQL executed against the real database, preprocessing traps, evaluation maths, reason-code compliance) and **artifact-dependent** (skipped with a clear message when the pipeline has not been run, so a fresh clone gets a green suite rather than a wall of red). No test calls the Anthropic API.

The tests worth reading are the compliance-shaped ones: a protected attribute never surfaces as a principal reason across 25 applicants; suppression is *reported*, not silently dropped; rule supports sum to 1.0; and 14 injection-shaped queries are each rejected by the guardrails.

---

## Repository layout

```
├── app/
│   ├── ui.py                       Streamlit UI, 9 sections
│   ├── theme.py                    design system: tokens, CSS, HTML components
│   └── charts.py                   Altair chart builders
├── notebooks/eda.ipynb + eda.py    EDA, executed with outputs
├── src/
│   ├── data/       loader, preprocessor, build_database, eda_insights, feature_dictionary
│   ├── ml/         train, predict, evaluate
│   ├── xai/        shap_explainer, reason_codes
│   ├── rules/      rule_extractor
│   ├── talk_to_data/  nl_to_sql, query_runner, prompt_templates, semantic_layer,
│   │                  eval_harness, token_report
│   ├── agent/      orchestrator, tools
│   ├── monitoring/ fairness
│   └── utils/      config, logger, helpers, llm, docker_utils
├── sql/schema.sql                  DDL — also the NL→SQL grounding block
├── evaluation/nl_sql_questions.jsonl   25 labelled cases
├── scripts/prepare_streamlit_artifacts.py   builds the committed demo sample
├── deploy_artifacts/               11.7 MB anonymised sample: the Streamlit Cloud build
├── packages.txt                    apt deps for Streamlit Cloud (libgomp1)
├── tests/                          141 tests
├── documents/
│   ├── project_presentation.pdf    14-slide deck, generated from reports/
│   └── build_presentation.py       the generator — no hardcoded numbers
├── docker/entrypoint.sh            builds artifacts on first run
├── Dockerfile                      multi-stage
└── docker-compose.yml
```

`data/`, `models/` and `reports/` are gitignored — the dataset is mounted at runtime, never committed. The one exception is `deploy_artifacts/`, a small anonymised sample built by `scripts/prepare_streamlit_artifacts.py` so the public Streamlit build has something to read; see [Streamlit Community Cloud](#streamlit-community-cloud).

---

## Configuration

All settings are environment variables (`.env.example` documents every one). The ones worth changing:

| Variable | Default | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Enables Talk to Data and the chat assistant |
| `SQL_MODEL` / `SUMMARY_MODEL` | `claude-sonnet-5` / `claude-haiku-4-5` | Generation vs summarisation |
| *(no `TEMPERATURE`)* | — | The Messages API in the anthropic 1.x SDK does not accept one. Determinism comes from the prompt — a pinned semantic layer and worked exemplars — not from sampling. |
| `LGD_RATE` / `MARGIN_RATE` | `0.45` / `0.08` | The cost matrix — changing these moves the threshold and every dollar figure |
| `SQL_ROW_LIMIT` / `SQL_MAX_RETRIES` | `200` / `2` | Guardrail bounds |
| `ENABLE_PROMPT_CACHING` | `true` | Set false to measure the caching lever yourself |

---

## Dataset

Home Credit Default Risk, from Kaggle. Not redistributed with this repository — download it and mount `data/raw/`. Four files are used: `application_train.csv`, `application_test.csv`, `bureau.csv`, `previous_application.csv`.
