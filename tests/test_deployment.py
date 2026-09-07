"""Deployment wiring: the Streamlit Cloud path switch and the demo fallback.

These are the two pieces that only ever run on the public host, which is exactly
why they need tests - a broken path override or a fallback that quietly returns
nothing would show up as a blank page in front of whoever opened the link, and
never on this machine.
"""

from __future__ import annotations

import json

import pytest

from src.utils.config import PROJECT_ROOT, Settings, deploy_paths


# --------------------------------------------------------------------------- #
# Path switching
# --------------------------------------------------------------------------- #
def _settings_with(monkeypatch, **environment) -> Settings:
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return Settings()


def test_local_settings_do_not_use_deploy_artifacts(monkeypatch):
    monkeypatch.delenv("STREAMLIT_CLOUD", raising=False)
    settings = Settings()
    assert deploy_paths()["data_dir"] not in settings.model_dir.parents
    assert deploy_paths()["data_dir"] != settings.model_dir


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes", "on"])
def test_streamlit_cloud_repoints_every_artifact_path(monkeypatch, flag):
    settings = _settings_with(monkeypatch, STREAMLIT_CLOUD=flag)
    expected = deploy_paths()

    assert settings.data_dir == expected["data_dir"]
    assert settings.model_dir == expected["model_dir"]
    assert settings.report_dir == expected["report_dir"]
    assert settings.processed_dir == expected["processed_dir"]
    assert settings.sqlite_path == expected["sqlite_path"]
    # Derived paths follow, so nothing downstream needs its own deployment branch.
    assert settings.holdout_path == expected["processed_dir"] / "holdout.parquet"
    assert settings.figures_dir == expected["report_dir"] / "figures"
    assert settings.model_path.parent == expected["model_dir"]
    assert settings.eval_results_path.parent == expected["report_dir"]


def test_streamlit_cloud_beats_an_explicit_path_variable(monkeypatch):
    """A stray MODEL_DIR in the environment must not send the app to an empty dir.

    The Secrets box and a `.env` both land in `os.environ`, so the override has to
    win on the deployed host regardless of what else was pasted in.
    """
    settings = _settings_with(monkeypatch, STREAMLIT_CLOUD="1", MODEL_DIR="models")
    assert settings.model_dir == deploy_paths()["model_dir"]


def test_demo_mode_defaults_off_with_a_budget(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    settings = Settings()
    assert settings.demo_mode is False
    assert settings.demo_llm_call_budget > 0


# --------------------------------------------------------------------------- #
# Host requirements
# --------------------------------------------------------------------------- #
def test_packages_txt_declares_the_openmp_runtime():
    """LightGBM needs libgomp1, and pip cannot install an apt package."""
    packages = (PROJECT_ROOT / "packages.txt").read_text(encoding="utf-8")
    assert [line for line in packages.split() if line] == ["libgomp1"]


def test_app_entrypoint_orders_its_prologue_correctly():
    """Three constraints, all order-sensitive, all invisible until deploy time.

    `set_page_config` must be the first Streamlit call; the secrets bridge must
    run before `settings` is constructed, or a key from the Secrets box never
    reaches pydantic; and the repo root must be on `sys.path` before `src` is
    imported, because `streamlit run app/ui.py` puts `app/` there, not the root.
    """
    source = (PROJECT_ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    page_config = source.index("st.set_page_config")
    secrets_bridge = source.index("os.environ.setdefault(_key")
    path_insert = source.index("sys.path.insert(0, str(ROOT))")
    settings_import = source.index("from src.utils.config import settings")

    assert page_config < secrets_bridge < path_insert < settings_import


# --------------------------------------------------------------------------- #
# Cached demo answers
# --------------------------------------------------------------------------- #
@pytest.fixture
def cached_cases(monkeypatch, tmp_path):
    """A miniature recorded evaluation report, standing in for a measured run."""
    from src.talk_to_data import demo_fallback

    report = {
        "cases_detail": [
            {"id": "q01", "question": "What is the overall default rate?",
             "fallback_expected": False,
             "sql": "SELECT AVG(TARGET) AS default_rate FROM applications"},
            {"id": "q08",
             "question": "Which 5 occupations have the highest default rate?",
             "fallback_expected": False,
             "sql": "SELECT OCCUPATION_TYPE, AVG(TARGET) AS default_rate "
                    "FROM applications GROUP BY OCCUPATION_TYPE"},
            {"id": "q21", "question": "What is the applicant's credit card CVV number?",
             "fallback_expected": True, "sql": None},
        ]
    }
    path = tmp_path / "nl_sql_eval.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(type(demo_fallback.settings), "eval_results_path",
                        property(lambda self: path))
    demo_fallback._cases.cache_clear()
    demo_fallback._idf.cache_clear()
    yield demo_fallback
    demo_fallback._cases.cache_clear()
    demo_fallback._idf.cache_clear()


def test_shared_boilerplate_alone_does_not_make_a_match(cached_cases):
    """"default rate" appears in most recorded questions and must not carry a match.

    Plain Jaccard matched "which occupations have the highest default rate" to
    "what is the overall default rate"; the IDF weighting is what separates them.
    """
    answer = cached_cases.demo_answer(
        "Which occupations have the highest default rate?")
    assert answer.case_id == "q08"


def test_an_unrelated_question_is_refused_rather_than_mismatched(cached_cases):
    answer = cached_cases.demo_answer("What is the weather in Paris?")
    assert answer.matched is False
    assert answer.sql is None
    # The empty state still names what the cache can do, so the page is never blank.
    assert "overall default rate" in answer.answer


def test_a_recorded_refusal_stays_a_refusal(cached_cases):
    answer = cached_cases.demo_answer("What is the applicant's credit card CVV number?")
    assert answer.case_id == "q21"
    assert answer.sql is None
    assert "cannot be answered" in answer.answer


def test_a_missing_report_degrades_to_a_message(monkeypatch, tmp_path):
    from src.talk_to_data import demo_fallback

    monkeypatch.setattr(type(demo_fallback.settings), "eval_results_path",
                        property(lambda self: tmp_path / "absent.json"))
    demo_fallback._cases.cache_clear()
    demo_fallback._idf.cache_clear()
    try:
        assert demo_fallback.available() is False
        answer = demo_fallback.demo_answer("What is the overall default rate?")
        assert answer.matched is False
        assert answer.answer  # never empty: an empty answer renders as a blank card
    finally:
        demo_fallback._cases.cache_clear()
        demo_fallback._idf.cache_clear()


def test_replay_runs_the_recorded_sql_against_the_live_database(cached_cases):
    """A cached answer is computed now, not read back from the recorded run."""
    from src.utils.docker_utils import database_ready

    if not database_ready():
        pytest.skip("No SQLite database.")

    answer = cached_cases.demo_answer("What is the overall default rate?")
    assert answer.replayed is True
    assert answer.row_count == 1
    assert 0.0 < float(answer.rows[0]["default_rate"]) < 1.0


def test_a_replayed_answer_carries_its_sql_into_the_transcript(cached_cases):
    from src.utils.docker_utils import database_ready

    if not database_ready():
        pytest.skip("No SQLite database.")

    message = cached_cases.demo_answer("What is the overall default rate?").as_message()
    assert message["role"] == "assistant"
    assert message["sql"].lower().startswith("select")
