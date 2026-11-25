import yaml

from aasl.cli import EXIT_WAITING, main
from aasl.io import load_units, write_units_csv
from aasl.simulate import make_corpus, seed_schema


def test_load_units_csv_and_jsonl(tmp_path):
    units, _ = make_corpus(n=50)
    write_units_csv(tmp_path / "u.csv", units, ["firm", "sector"])
    back = load_units(tmp_path / "u.csv", meta_columns=["firm", "sector"])
    assert [u.uid for u in back] == [u.uid for u in units] and back[0].meta["firm"]
    (tmp_path / "u.jsonl").write_text('{"id": 1, "text": "a"}\n{"id": 2, "text": ""}\n')
    assert [u.uid for u in load_units(tmp_path / "u.jsonl")] == ["1"]


def test_demo_command(tmp_path, capsys):
    assert main(["demo", "--run-dir", str(tmp_path / "demo")]) == 0
    out = capsys.readouterr().out
    assert "Standard setting" in out


def test_sample_size_and_codebook(tmp_path, capsys):
    assert main(["sample-size"]) == 0
    seed_schema().save(tmp_path / "s.yaml")
    assert main(["codebook", str(tmp_path / "s.yaml")]) == 0
    assert "RESIDUAL" in capsys.readouterr().out


def test_run_command_pauses_for_humans(tmp_path, capsys, monkeypatch):
    units, _ = make_corpus(n=200)
    write_units_csv(tmp_path / "units.csv", units, ["firm", "sector", "region"])
    seed_schema().save(tmp_path / "seeds.yaml")
    cfg = {"run_dir": "run", "schema": "seeds.yaml",
           "data": {"path": "units.csv", "meta_columns": ["firm", "sector", "region"]},
           "model": {"provider": "openai", "name": "test-model"},
           "settings": {"holdout_size": 20, "calibration_size": 20}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))

    import aasl.config as config_mod
    from aasl.simulate import SimulatedModel
    monkeypatch.setattr(config_mod, "build_model", lambda c, cache_dir=None: SimulatedModel({}))
    code = main(["run", str(tmp_path / "config.yaml")])
    assert code == EXIT_WAITING
    assert "coder_1.csv" in capsys.readouterr().out
    assert (tmp_path / "run" / "human" / "coding" / "round1_calibration" / "coder_2.csv").exists()
    assert main(["status", str(tmp_path / "config.yaml")]) == 0


def test_example_config_builds_and_pauses(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    import aasl.config as config_mod
    from aasl.simulate import SimulatedModel

    src = Path(__file__).resolve().parents[1] / "examples" / "cooperation"
    dst = tmp_path / "examples" / "cooperation"
    shutil.copytree(src, dst)
    monkeypatch.setattr(config_mod, "build_model", lambda c, cache_dir=None: SimulatedModel({}))
    cfg = config_mod.load_config(dst / "config.yaml")
    run = config_mod.build_run(cfg, verbose=False)
    assert len(run.units) == 1500 and run.s.tau == 0.9
    assert main(["run", str(dst / "config.yaml")]) == EXIT_WAITING
    assert (tmp_path / "runs" / "cooperation" / "samples.json").exists()
