import json

import run_student


def test_unified_entrypoint_orders_experiments_and_writes_one_summary(monkeypatch, tmp_path):
    calls = []

    def linear():
        calls.append("linear")
        return {"dataset_sha256": "dataset", "split": {"counts": {"train": 2}}}

    def forest():
        calls.append("forest")
        return {"selection_lock": {"selected_model_key": "rf_raw"}}

    def handoff():
        calls.append("handoff")
        return {"selected_model_key": "rf_raw", "validation_rmse": 0.5}

    monkeypatch.setattr(run_student, "run_linear", linear)
    monkeypatch.setattr(run_student, "run_forest", forest)
    monkeypatch.setattr(run_student, "build_handoff", handoff)
    monkeypatch.setattr(run_student, "NOTES_DIR", tmp_path / "notes")
    monkeypatch.setattr(run_student, "RESULTS_DIR", tmp_path / "results")
    run_student.main()
    assert calls == ["linear", "forest", "handoff"]
    summary = json.loads((tmp_path / "notes" / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["overall_selection"]["selected_model_key"] == "rf_raw"
    assert (tmp_path / "results" / "run_metadata.json").exists()
