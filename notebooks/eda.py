#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk - Exploratory Data Analysis
# 
# **AI-Powered Credit Risk Intelligence Platform**
# 
# The question behind this notebook is not "what is in this dataset" but
# **"what would a credit risk officer need to know before trusting a model built on it?"**
# 
# Four things, in order:
# 
# 1. What are we actually predicting, and how rare is it?
# 2. What is the data quality, honestly - including the traps that are not missing values?
# 3. Which factors separate applicants who repay from applicants who default?
# 4. What does all of that imply for how the model must be built?
# 
# Every analysis here is imported from `src/data/eda_insights.py` rather than
# written inline. That module is also what the Streamlit EDA tab calls, so the
# notebook and the deployed app cannot drift apart.

# In[1]:


import sys
from pathlib import Path

# Make the project importable when the notebook runs from notebooks/.
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import Image, display

from src.data.eda_insights import (
    data_quality_scorecard, feature_categorisation, insight_age, insight_contract_type,
    insight_employment, insight_external_scores, insight_income, target_summary,
)
from src.data.loader import dataset_profile, load_dataset
from src.data.preprocessor import missing_report
from src.utils.config import settings

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
FIGURES = settings.figures_dir


# ## 1. Dataset overview
# 
# `application_train.csv` is the core table: one row per loan application, with the
# outcome we are predicting. Two supplementary tables are joined in as
# per-applicant aggregates - `bureau.csv` (credit history reported by external
# bureaus) and `previous_application.csv` (prior applications to this lender).
# 
# The other four Home Credit tables are deliberately out of scope: roughly 1.9 GB
# of monthly balance histories for a marginal gain, which would dominate build
# time without changing the conclusions.

# In[2]:


frame = load_dataset()
profile = dataset_profile(frame)
profile


# In[3]:


frame.head(5).iloc[:, :12]


# ### Feature categorisation
# 
# 122 raw columns plus 33 joined aggregates is too many to reason about one at a
# time. Grouping them by what they *measure* makes the shape of the dataset
# visible - and immediately shows that the largest single block (property building
# statistics) is also the least populated.

# In[4]:


groups = feature_categorisation(frame)
pd.DataFrame([
    {"feature_group": name, "columns": info["count"], "examples": ", ".join(info["examples"][:3])}
    for name, info in groups.items()
])


# ## 2. The target: an 8% event rate
# 
# This single number drives most of the modelling decisions that follow.

# In[5]:


balance, figure = target_summary(frame)
display(balance)
display(Image(filename=str(FIGURES / figure)))


# **Consequence.** At 8.07% positives (1 default per 12.4 applicants):
# 
# - **Accuracy is useless.** A model predicting "everyone repays" scores 91.9%.
# - **ROC-AUC flatters.** PR-AUC is reported alongside it throughout this project.
# - **Imbalance must be handled explicitly** - here with LightGBM
#   `scale_pos_weight = 11.4`, not SMOTE (307k rows in 143 dimensions makes
#   synthetic oversampling slow, and interpolating between categorical codes
#   produces applicants that cannot exist).
# - **Reweighting inflates the scores**, so calibration is not optional if the
#   probability is going to be used for anything but ranking.

# ## 3. Data quality
# 
# ### 3a. Missingness
# 
# 49 of the 122 raw columns are more than 40% empty. Almost all of them belong to
# the property block, and they are missing for a structural reason: the applicant
# does not live in an apartment building the lender has statistics for.

# In[6]:


report = missing_report(frame)
print(f"Columns over 40% missing: {(report['missing_rate'] > 0.4).sum()}")
print(f"Fully populated columns:  {(report['missing_rate'] == 0).sum()}")
report.head(12)[["missing_count", "missing_rate", "dtype"]].round(3)


# In[7]:


scorecard, missing_figure = data_quality_scorecard(frame)
display(scorecard)
display(Image(filename=str(FIGURES / missing_figure)))


# ### 3b. The traps that are not missing values
# 
# A null is easy: every profiler finds it. The expensive errors in this dataset are
# values that are **present, valid-looking and wrong**.

# In[8]:


for anomaly in scorecard.attrs["anomalies"]:
    print("-", anomaly)


# **`DAYS_EMPLOYED = 365243` is the one that matters.** It appears on 55,374 rows
# (18.0%) and decodes to exactly 1000 years of employment. It is a sentinel for
# "no employment record".
# 
# Left untouched it does three separate kinds of damage: the mean employment
# length becomes meaningless, every ratio built on it inherits the corruption, and
# any tree splitting on employment length gets a free, spurious separation.
# 
# Handled here in `CreditRiskPreprocessor.clean()`: the sentinel becomes `NaN` and
# gains an explicit `IS_UNEMPLOYED` flag, so the signal survives without the lie.
# The same repair is pinned into the Talk-to-Data semantic layer, so SQL answers
# cannot reintroduce it.

# In[9]:


sentinel = frame["DAYS_EMPLOYED"] == 365243
print(f"Sentinel rows: {sentinel.sum():,} ({sentinel.mean():.1%})")
print()
print("Who are they? Income type of the sentinel group:")
print(frame.loc[sentinel, "NAME_INCOME_TYPE"].value_counts().head())
print()
print(f"Naive mean 'employment years' if left raw: "
      f"{(-frame['DAYS_EMPLOYED'] / 365).mean():.1f} years")
print(f"Mean after repairing the sentinel:         "
      f"{(-frame.loc[~sentinel, 'DAYS_EMPLOYED'] / 365).mean():.1f} years")


# ## 4. Business insights
# 
# Five insights, each with a chart, each with a consequence for the model or the
# policy. The dashed line on every chart is the 8.07% portfolio average.

# ### Insight 1 - Income predicts default, but weakly
# 
# Higher income does lower default risk, and the gradient is real. But the spread
# across the whole income range is roughly 3 percentage points, against a spread of
# 17 points across external credit-score deciles (Insight 4). Income is a weak
# signal in this portfolio - which is exactly why a bank buys bureau scores.

# In[10]:


table, figure = insight_income(frame)
display(table)
display(Image(filename=str(FIGURES / figure)))


# ### Insight 2 - Cash loans default ~50% more often than revolving credit
# 
# 8.35% against 5.48%. The average cash loan is also nearly double the size
# (628k vs 324k), so the exposure gap is wider than the rate gap alone suggests -
# which is precisely why the decision threshold in this project is cost-weighted
# by loan amount rather than applied uniformly.

# In[11]:


table, figure = insight_contract_type(frame)
display(table)
display(Image(filename=str(FIGURES / figure)))


# ### Insight 3 - Default risk falls monotonically with age
# 
# 12.3% for under-25s against 4.9% for over-60s: a 2.5x spread, and one of the
# cleanest monotone relationships in the dataset.
# 
# It is also the reason the Model Health & Fairness tab exists. Age is a protected
# attribute under ECOA. The model may use it; the adverse-action notice may not
# cite it. `reason_codes.py` suppresses it from the notice and reports the
# suppression rather than hiding it.

# In[12]:


table, figure = insight_age(frame)
display(table)
display(Image(filename=str(FIGURES / figure)))


# ### Insight 4 - External credit scores are the strongest signal in the dataset
# 
# All three `EXT_SOURCE` columns separate the population by roughly **6x** between
# their best and worst deciles. Nothing else in the application form comes close.
# 
# The catch is coverage: `EXT_SOURCE_1` is present for only 44% of applicants. So
# the feature engineering builds `EXT_SOURCE_MEAN` (averaging whichever scores
# exist), `EXT_SOURCE_MIN`, `EXT_SOURCE_MAX` and `EXT_SOURCE_COUNT` - because
# *how many* bureaus will score an applicant is itself informative.

# In[13]:


table, figure = insight_external_scores(frame)
display(table)
display(Image(filename=str(FIGURES / figure)))


# ### Insight 5 - Employment length, and why the sentinel must be split out
# 
# Among employed applicants the trend is clean: 11.0% at under a year, 5.2% past
# ten years.
# 
# The striking part is the sentinel group. Those 55,374 "not employed" applicants
# default at **5.4%** - *below* the employed average of 8.7% - because 99.96% of
# them are pensioners, who have stable income and no employment record.
# 
# Bin them into the "10 years+" bucket and the number moves the wrong way for the
# wrong reason. Drop them and 18% of the portfolio disappears. Split them out and
# they are one of the safer segments in the book. This is the difference between
# a data-quality note and an actual finding.

# In[14]:


table, figure = insight_employment(frame)
display(table)
display(Image(filename=str(FIGURES / figure)))


# ## 5. What the EDA dictates about the model
# 
# | Finding | Consequence |
# |---|---|
# | 8.07% positive rate | `scale_pos_weight = 11.4`; report PR-AUC alongside ROC-AUC; calibrate afterwards |
# | 49 columns >40% missing | Keep NaN and let LightGBM learn a split direction - missingness is itself predictive |
# | `_AVG` / `_MODE` / `_MEDI` triplets | Keep `_AVG`, drop ~30 near-duplicate columns |
# | `DAYS_EMPLOYED = 365243` | Repair to NaN + `IS_UNEMPLOYED`; pin the fix into the SQL semantic layer too |
# | Income outlier at 117M (795x median) | Clip at the 99.95th percentile, learned at fit time and replayed at inference |
# | External scores dominate but are sparse | Engineer mean / min / max / count / disagreement across the three |
# | Age and gender are predictive | Model may use them; the adverse-action notice may not cite them - suppress and report |
# | Cash loans are larger *and* riskier | Cost-weight the decision threshold by `AMT_CREDIT`, not by count |
# 
# Next: `src/ml/train.py` builds the model these findings imply, and
# `src/monitoring/fairness.py` measures the group gaps this section predicts.
