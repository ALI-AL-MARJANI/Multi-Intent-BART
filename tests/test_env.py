from __future__ import annotations

from gemis.utils.env import code_sha256, collect_env


def test_code_hash_tracks_sources(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "configs" / "c.yaml").write_text("k: 1\n")
    first = code_sha256(tmp_path)
    assert first == code_sha256(tmp_path) and len(first) == 64
    (tmp_path / "README.md").write_text("ignored")
    assert code_sha256(tmp_path) == first
    (tmp_path / "configs" / "c.yaml").write_text("k: 2\n")
    assert code_sha256(tmp_path) != first


def test_env_has_the_fields_results_rely_on():
    env = collect_env()
    assert {"date_utc", "code_sha256", "git_sha", "python", "platform", "versions"} <= set(env)
    assert env["versions"]["torch"] and env["versions"]["transformers"]
