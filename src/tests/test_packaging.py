"""A1：打包卫生。

钉死三件事，防止以后被悄悄改回去：
1. **dist 名不能是 `dgisim`** —— 否则 `pip install dgisim` 会静默装到 PyPI 上归档版的 0.5.0，
   而你以为装的是这个 fork。
2. `pyproject.toml` 的 version 必须与 `src/package/__init__.py` 的 `__version__` 一致。
3. `[project.urls]` 的 source 必须指向本 fork，而不是归档的上游。

import 名仍然是 `dgisim`（由 `[tool.setuptools.package-dir]` 映射），这是有意的：
换 dist 名不影响任何 import 语句。
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
PACKAGE_INIT = REPO / "src" / "package" / "__init__.py"
FORK_URL = "https://github.com/leimuwo/Dottore-Genius-Invokation-TCG-Simulator"


def _pyproject() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def _field(text: str, section: str, key: str) -> str:
    """从 pyproject 里抠出 [section] 下的 key = "value"。"""
    match = re.search(
        rf"^\[{re.escape(section)}\]\s*$.*?^{re.escape(key)}\s*=\s*\"([^\"]+)\"",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{section}.{key} 不存在"
    return match.group(1)


class TestPackaging(unittest.TestCase):
    def test_dist_name_is_not_upstream(self):
        name = _field(_pyproject(), "project", "name")
        self.assertNotEqual(name, "dgisim", "dist 名不能与 PyPI 上的归档版同名（A1）")
        self.assertEqual(name, "dgisim-fork")

    def test_version_is_consistent(self):
        pyproject_version = _field(_pyproject(), "project", "version")
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', PACKAGE_INIT.read_text(encoding="utf-8"), re.M)
        self.assertIsNotNone(match, "src/package/__init__.py 里没有 __version__")
        self.assertEqual(pyproject_version, match.group(1), "两处版本号不一致")

    def test_urls_point_to_fork(self):
        source = _field(_pyproject(), "project.urls", "source")
        self.assertEqual(source, FORK_URL)
