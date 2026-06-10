import subprocess

import pytest

import ssh_guard.helpers.cache_conda as cache_conda


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)


def test_find_conda_source_returns_none_when_clean_shell_can_activate(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["bash", "-lc"] and "conda activate base" in cmd[2]:
            return _completed(0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cache_conda.subprocess, "run", fake_run)

    assert cache_conda._find_conda_source() is None
    assert len(calls) == 1


def test_find_conda_source_uses_bashrc_when_activate_needs_source(
    monkeypatch, tmp_path
):
    conda_source = tmp_path / "miniforge3" / "etc" / "profile.d" / "conda.sh"
    conda_source.parent.mkdir(parents=True)
    conda_source.write_text("", encoding="utf-8")
    (tmp_path / ".bashrc").write_text(
        "# >>> conda initialize >>>\n"
        f'. "{conda_source.as_posix()}"\n'
        "# <<< conda initialize <<<\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", tmp_path.as_posix())

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["bash", "-lc"] and "conda activate base" in cmd[2]:
            return _completed(0 if "source " in cmd[2] else 1)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cache_conda.subprocess, "run", fake_run)

    assert cache_conda._find_conda_source() == conda_source.as_posix()


def test_find_conda_source_falls_back_to_conda_base(monkeypatch, tmp_path):
    conda_base = tmp_path / "miniforge3"
    conda_source = conda_base / "etc" / "profile.d" / "conda.sh"
    conda_source.parent.mkdir(parents=True)
    conda_source.write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", tmp_path.as_posix())

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["bash", "-lc"] and "conda activate base" in cmd[2]:
            return _completed(0 if "source " in cmd[2] else 1)
        if cmd == ["conda", "info", "--base"]:
            return _completed(0, stdout=f"{conda_base}\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(cache_conda.subprocess, "run", fake_run)

    assert cache_conda._find_conda_source() == conda_source.as_posix()


def test_main_fails_when_conda_activate_cannot_be_initialized(monkeypatch, capsys):
    monkeypatch.setattr(cache_conda, "handle_help", lambda *args, **kwargs: None)
    monkeypatch.setattr(cache_conda, "_find_conda_source", lambda: None)
    monkeypatch.setattr(cache_conda, "_conda_activate_works", lambda source: False)

    with pytest.raises(SystemExit) as exc:
        cache_conda.main()

    assert exc.value.code == 1
    assert "conda activate is not initialized" in capsys.readouterr().err
