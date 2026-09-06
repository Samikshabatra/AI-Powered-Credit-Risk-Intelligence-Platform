-- =====================================================================
-- Credit Risk Intelligence Platform - analytical SQLite schema
--
-- This file is the single source of truth for the database AND for the
-- Talk-to-Data prompt: the NL->SQL module injects this exact DDL into the
-- model context, so the model can only reference columns that really exist.
-- Keep the comments - they are the column glossary the model reads.
--
-- Scope note: `applications` holds a curated 39-column subset of the 122
-- columns in application_train.csv. The dropped columns are the property
-- building-statistics block (65%+ missing, no business-question value).
-- A compact schema is also the biggest token lever in the NL->SQL prompt.
-- =====================================================================

DROP TABLE IF EXISTS applications;
DROP TABLE IF EXISTS bureau_summary;
DROP TABLE IF EXISTS previous_summary;
DROP TABLE IF EXISTS predictions;

-- ---------------------------------------------------------------------
-- applications: one row per loan applicant (source: application_train.csv)
-- ---------------------------------------------------------------------
CREATE TABLE applications (
    SK_ID_CURR                  INTEGER PRIMARY KEY,  -- applicant id
    TARGET                      INTEGER,  -- 1 = defaulted, 0 = repaid
    NAME_CONTRACT_TYPE          TEXT,     -- 'Cash loans' or 'Revolving loans'
    CODE_GENDER                 TEXT,     -- 'M', 'F' ('XNA' = unknown, 4 rows)
    FLAG_OWN_CAR                TEXT,     -- 'Y' / 'N'
    FLAG_OWN_REALTY             TEXT,     -- 'Y' / 'N'
    CNT_CHILDREN                INTEGER,  -- number of children
    AMT_INCOME_TOTAL            REAL,     -- annual income
    AMT_CREDIT                  REAL,     -- credit amount of the loan
    AMT_ANNUITY                 REAL,     -- loan annuity (yearly payment)
    AMT_GOODS_PRICE             REAL,     -- price of the goods being financed
    NAME_TYPE_SUITE             TEXT,     -- who accompanied the applicant
    NAME_INCOME_TYPE            TEXT,     -- 'Working', 'Pensioner', 'State servant', ...
    NAME_EDUCATION_TYPE         TEXT,     -- highest education level
    NAME_FAMILY_STATUS          TEXT,     -- 'Married', 'Single / not married', ...
    NAME_HOUSING_TYPE           TEXT,     -- 'House / apartment', 'With parents', ...
    REGION_POPULATION_RELATIVE  REAL,     -- normalised population of the region
    DAYS_BIRTH                  INTEGER,  -- NEGATIVE days before application; age = -DAYS_BIRTH/365.0
    DAYS_EMPLOYED               INTEGER,  -- NEGATIVE days before application; 365243 = NOT EMPLOYED sentinel
    DAYS_REGISTRATION           REAL,     -- negative days since registration change
    DAYS_ID_PUBLISH             INTEGER,  -- negative days since identity document issued
    OWN_CAR_AGE                 REAL,     -- age of the applicant's car, in years
    OCCUPATION_TYPE             TEXT,     -- 'Laborers', 'Sales staff', 'Core staff', ... (NULL for ~31%)
    CNT_FAM_MEMBERS             REAL,     -- household size
    REGION_RATING_CLIENT        INTEGER,  -- region rating 1 (best) to 3 (worst)
    WEEKDAY_APPR_PROCESS_START  TEXT,     -- weekday the application started
    HOUR_APPR_PROCESS_START     INTEGER,  -- hour the application started
    ORGANIZATION_TYPE           TEXT,     -- employer industry (58 levels)
    EXT_SOURCE_1                REAL,     -- external credit score 1, 0-1, HIGHER = SAFER (56% NULL)
    EXT_SOURCE_2                REAL,     -- external credit score 2, 0-1, HIGHER = SAFER
    EXT_SOURCE_3                REAL,     -- external credit score 3, 0-1, HIGHER = SAFER (20% NULL)
    OBS_30_CNT_SOCIAL_CIRCLE    REAL,     -- observable people in the social circle
    DEF_30_CNT_SOCIAL_CIRCLE    REAL,     -- people in the social circle who defaulted (30 days past due)
    DAYS_LAST_PHONE_CHANGE      REAL,     -- negative days since the phone number changed
    AMT_REQ_CREDIT_BUREAU_YEAR  REAL,     -- credit-bureau enquiries in the last year
    FLAG_DOCUMENT_3             INTEGER,  -- 1 if document 3 was supplied
    FLAG_EMAIL                  INTEGER,  -- 1 if an email address was supplied
    FLAG_PHONE                  INTEGER,  -- 1 if a home phone was supplied
    REG_CITY_NOT_WORK_CITY      INTEGER   -- 1 if the registered city differs from the work city
);

-- ---------------------------------------------------------------------
-- bureau_summary: external credit-bureau history, aggregated per applicant
-- (source: bureau.csv). Applicants with no bureau record have no row here.
-- ---------------------------------------------------------------------
CREATE TABLE bureau_summary (
    SK_ID_CURR                  INTEGER PRIMARY KEY,
    BUREAU_LOAN_COUNT           INTEGER,  -- number of loans reported by the bureau
    BUREAU_ACTIVE_COUNT         INTEGER,  -- how many are still active
    BUREAU_CLOSED_COUNT         INTEGER,  -- how many are closed
    BUREAU_CREDIT_SUM           REAL,     -- total external credit granted
    BUREAU_DEBT_SUM             REAL,     -- total external debt still outstanding
    BUREAU_OVERDUE_SUM          REAL,     -- total amount currently overdue
    BUREAU_DAYS_OVERDUE_MAX     REAL,     -- longest current overdue period, in days
    BUREAU_DAYS_CREDIT_MEAN     REAL,     -- average age of external loans (negative days)
    BUREAU_DEBT_CREDIT_RATIO    REAL,     -- external credit utilisation (debt / credit)
    BUREAU_CREDIT_TYPE_NUNIQUE  INTEGER   -- variety of external credit products held
);

-- ---------------------------------------------------------------------
-- previous_summary: prior Home Credit applications, aggregated per applicant
-- (source: previous_application.csv).
-- ---------------------------------------------------------------------
CREATE TABLE previous_summary (
    SK_ID_CURR                  INTEGER PRIMARY KEY,
    PREV_APP_COUNT              INTEGER,  -- number of previous applications
    PREV_APPROVED_COUNT         INTEGER,  -- how many were approved
    PREV_REFUSED_COUNT          INTEGER,  -- how many were refused
    PREV_REFUSAL_RATE           REAL,     -- refused / total
    PREV_AMT_CREDIT_MEAN        REAL,     -- average credit granted previously
    PREV_CNT_PAYMENT_MEAN       REAL,     -- average previous loan term, in payments
    PREV_DAYS_DECISION_MAX      REAL      -- negative days since the most recent decision
);

-- ---------------------------------------------------------------------
-- predictions: model output, written by src/ml/train.py after training.
-- Lets the analyst query the model itself, not just the raw data.
-- ---------------------------------------------------------------------
CREATE TABLE predictions (
    SK_ID_CURR                  INTEGER PRIMARY KEY,
    DEFAULT_PROBABILITY         REAL,     -- calibrated probability of default, 0-1
    RISK_BAND                   TEXT,     -- 'Low', 'Medium' or 'High'
    DECISION                    TEXT,     -- 'Approve' or 'Decline' at the cost-optimal threshold
    DATA_SPLIT                  TEXT      -- 'train', 'valid', 'calib' or 'holdout'
);

-- ---------------------------------------------------------------------
-- Indexes for the group-by columns the analyst actually filters on.
-- ---------------------------------------------------------------------
CREATE INDEX idx_app_target        ON applications (TARGET);
CREATE INDEX idx_app_contract      ON applications (NAME_CONTRACT_TYPE);
CREATE INDEX idx_app_gender        ON applications (CODE_GENDER);
CREATE INDEX idx_app_education     ON applications (NAME_EDUCATION_TYPE);
CREATE INDEX idx_app_occupation    ON applications (OCCUPATION_TYPE);
CREATE INDEX idx_app_income        ON applications (AMT_INCOME_TOTAL);
CREATE INDEX idx_pred_band         ON predictions (RISK_BAND);
CREATE INDEX idx_pred_split        ON predictions (DATA_SPLIT);
