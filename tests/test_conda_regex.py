import pathlib
import sys
import types

import pytest

if "yaml" not in sys.modules:  # pragma: no cover - optional dependency shim
    yaml_stub = types.ModuleType("yaml")

    def _safe_load(_stream):
        raise RuntimeError("yaml.safe_load shim should not be called in regex tests")

    yaml_stub.safe_load = _safe_load
    sys.modules["yaml"] = yaml_stub

from remote_http_launcher import CONDA_INIT_BLOCK_RE, CONDA_PATH_RE


def _read_example_bashrc() -> str:
    example_path = pathlib.Path(__file__).with_name("example-bashrc")
    return example_path.read_text(encoding="utf-8")


def test_conda_init_block_regex_finds_block() -> None:
    content = _read_example_bashrc()
    match = CONDA_INIT_BLOCK_RE.search(content)
    assert match is not None, "Conda initialize block should be detected"

    block = match.group(1)
    assert "__conda_setup" in block

    path_match = CONDA_PATH_RE.search(block)
    assert path_match is not None, "conda.sh path should be extracted from block"
    assert path_match.group(1) == "/home/sjoerd/miniconda3/etc/profile.d/conda.sh"


@pytest.mark.parametrize(
    "line,expected",
    [
        (
            '    . "/opt/anaconda3/etc/profile.d/conda.sh"',
            "/opt/anaconda3/etc/profile.d/conda.sh",
        ),
        (
            "    . '/Users/example/miniconda/etc/profile.d/conda.sh'",
            "/Users/example/miniconda/etc/profile.d/conda.sh",
        ),
    ],
)
def test_conda_path_regex_matches_single_and_double_quotes(
    line: str, expected: str
) -> None:
    match = CONDA_PATH_RE.search(line)
    assert match is not None
    assert match.group(1) == expected
