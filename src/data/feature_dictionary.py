"""Human-readable labels and feature groupings.

Two jobs:
  * FEATURE_LABELS  -> plain-English names for adverse-action reason codes and UI.
  * FEATURE_GROUPS  -> the EDA feature categorisation (demographic / financial /
    credit-history / repayment-behaviour / document / external-source / property).
"""

from __future__ import annotations

import re

FEATURE_LABELS: dict[str, str] = {
    # --- identifiers / target ---
    "SK_ID_CURR": "Applicant ID",
    "TARGET": "Defaulted",
    # --- demographic ---
    "CODE_GENDER": "Gender",
    "DAYS_BIRTH": "Age (days before application)",
    "AGE_YEARS": "Age in years",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Family size",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_HOUSING_TYPE": "Housing situation",
    "NAME_INCOME_TYPE": "Income type",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer industry",
    "NAME_TYPE_SUITE": "Accompanied by",
    "REGION_RATING_CLIENT": "Region rating",
    "REGION_RATING_CLIENT_W_CITY": "Region rating (city adjusted)",
    "REGION_POPULATION_RELATIVE": "Region population density",
    # --- financial ---
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Loan amount requested",
    "AMT_ANNUITY": "Loan annuity (yearly payment)",
    "AMT_GOODS_PRICE": "Price of financed goods",
    "NAME_CONTRACT_TYPE": "Contract type",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns property",
    "OWN_CAR_AGE": "Age of car",
    # --- engineered financial ratios ---
    "CREDIT_INCOME_RATIO": "Loan-to-income ratio",
    "ANNUITY_INCOME_RATIO": "Debt-service-to-income ratio",
    "CREDIT_TERM": "Implied repayment term",
    "GOODS_CREDIT_RATIO": "Down-payment coverage",
    "INCOME_PER_PERSON": "Income per household member",
    "CREDIT_PER_PERSON": "Loan amount per household member",
    # --- employment ---
    "DAYS_EMPLOYED": "Employment length (days before application)",
    "EMPLOYMENT_YEARS": "Years in current employment",
    "EMPLOYED_TO_AGE_RATIO": "Share of life spent in current job",
    "IS_UNEMPLOYED": "Not currently employed",
    "DAYS_REGISTRATION": "Days since registration change",
    "DAYS_ID_PUBLISH": "Days since ID document issued",
    "DAYS_LAST_PHONE_CHANGE": "Days since last phone change",
    # --- external scores ---
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_MEAN": "Average external credit score",
    "EXT_SOURCE_MIN": "Weakest external credit score",
    "EXT_SOURCE_MAX": "Strongest external credit score",
    "EXT_SOURCE_STD": "Disagreement between credit bureaus",
    "EXT_SOURCE_COUNT": "Number of external scores available",
    # --- bureau aggregates ---
    "BUREAU_LOAN_COUNT": "Number of past bureau-reported loans",
    "BUREAU_ACTIVE_COUNT": "Currently active external loans",
    "BUREAU_ACTIVE_RATIO": "Share of external loans still active",
    "BUREAU_CLOSED_COUNT": "Closed external loans",
    "BUREAU_DEBT_SUM": "Total outstanding external debt",
    "BUREAU_CREDIT_SUM": "Total external credit granted",
    "BUREAU_DEBT_CREDIT_RATIO": "External credit utilisation",
    "BUREAU_OVERDUE_MAX": "Worst external overdue amount",
    "BUREAU_DAYS_OVERDUE_MAX": "Longest external overdue period",
    "BUREAU_DAYS_CREDIT_MEAN": "Average age of external loans",
    "BUREAU_DAYS_CREDIT_MAX": "Most recent external loan",
    "BUREAU_PROLONG_SUM": "Times an external loan was prolonged",
    "BUREAU_CREDIT_TYPE_NUNIQUE": "Variety of external credit products",
    # --- previous application aggregates ---
    "PREV_APP_COUNT": "Previous applications with Home Credit",
    "PREV_APPROVED_COUNT": "Previously approved applications",
    "PREV_REFUSED_COUNT": "Previously refused applications",
    "PREV_REFUSAL_RATE": "Historic refusal rate",
    "PREV_AMT_CREDIT_MEAN": "Average previous loan size",
    "PREV_AMT_APPLICATION_MEAN": "Average previous amount requested",
    "PREV_CREDIT_TO_APPLICATION": "Previously granted vs requested",
    "PREV_CNT_PAYMENT_MEAN": "Average previous loan term",
    "PREV_DAYS_DECISION_MAX": "Days since most recent decision",
    "PREV_DOWN_PAYMENT_RATE_MEAN": "Average previous down-payment rate",
    # --- documents / contact flags ---
    "DOCUMENT_SUBMITTED_COUNT": "Supporting documents submitted",
    "CONTACT_FLAG_COUNT": "Contact details provided",
    "FLAG_MOBIL": "Mobile phone provided",
    "FLAG_EMP_PHONE": "Employer phone provided",
    "FLAG_EMAIL": "Email provided",
    # --- social circle / enquiries ---
    "OBS_30_CNT_SOCIAL_CIRCLE": "Observations of social circle (30d)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults in social circle (30d)",
    "OBS_60_CNT_SOCIAL_CIRCLE": "Observations of social circle (60d)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Defaults in social circle (60d)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit-bureau enquiries in the last year",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Credit-bureau enquiries in the last quarter",
    "AMT_REQ_CREDIT_BUREAU_MON": "Credit-bureau enquiries in the last month",
}

# Prefix / regex based grouping used by the EDA feature categorisation.
FEATURE_GROUPS: dict[str, list[str]] = {
    "external_source": [r"^EXT_SOURCE"],
    "credit_history_bureau": [r"^BUREAU_", r"^AMT_REQ_CREDIT_BUREAU"],
    "repayment_behaviour": [r"^PREV_", r"_SOCIAL_CIRCLE$"],
    "document": [r"^FLAG_DOCUMENT", r"^DOCUMENT_"],
    "contact": [r"^FLAG_(MOBIL|EMP_PHONE|WORK_PHONE|CONT_MOBILE|PHONE|EMAIL)$", r"^CONTACT_"],
    "property": [
        r"(AVG|MODE|MEDI)$", r"^(APARTMENTS|BASEMENTAREA|YEARS_|COMMONAREA|ELEVATORS|"
        r"ENTRANCES|FLOORS|LAND|LIVING|NONLIVING|TOTALAREA|WALLSMATERIAL|HOUSETYPE|"
        r"EMERGENCYSTATE|FONDKAPREMONT)",
    ],
    "financial": [
        r"^AMT_(INCOME|CREDIT|ANNUITY|GOODS)", r"RATIO$", r"^CREDIT_TERM$",
        r"^INCOME_PER", r"^CREDIT_PER", r"^NAME_CONTRACT_TYPE$", r"^FLAG_OWN",
    ],
    "employment": [r"^DAYS_EMPLOYED", r"^EMPLOYMENT_", r"^EMPLOYED_", r"^IS_UNEMPLOYED$",
                   r"^OCCUPATION_TYPE$", r"^ORGANIZATION_TYPE$"],
    "demographic": [
        r"^CODE_GENDER$", r"^DAYS_BIRTH$", r"^AGE_", r"^CNT_(CHILDREN|FAM_MEMBERS)$",
        r"^NAME_(FAMILY|EDUCATION|HOUSING|INCOME|TYPE_SUITE)", r"^REGION_", r"^LIVE_",
        r"^REG_",
    ],
}

_GROUP_ORDER = [
    "external_source", "credit_history_bureau", "repayment_behaviour", "document",
    "contact", "property", "financial", "employment", "demographic",
]


def feature_group(column: str) -> str:
    """Map a column to exactly one feature group ('other' when nothing matches)."""
    for group in _GROUP_ORDER:
        for pattern in FEATURE_GROUPS[group]:
            if re.search(pattern, column):
                return group
    return "other"


def group_columns(columns: list[str]) -> dict[str, list[str]]:
    """Bucket a column list into feature groups (used by the EDA notebook)."""
    grouped: dict[str, list[str]] = {}
    for column in columns:
        grouped.setdefault(feature_group(column), []).append(column)
    return grouped
