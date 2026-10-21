"""E8: status content has a declared schema, independent of PYTHONHASHSEED."""
from dataclasses import fields
import unittest

from src.dgisim.encoding.encoding_plan import encoding_plan
from src.dgisim.support.support import ChefMaoSupport


class TestStatusFieldOrder(unittest.TestCase):
    def test_usages_first_then_declared_fields(self):
        support = ChefMaoSupport(sid=4, usages=1)
        names = [f.name for f in fields(support) if f.name != 'usages']
        expected = [encoding_plan.encode_item(support), support.usages]
        expected += [int(getattr(support, name)) for name in names]
        expected += [0] * (encoding_plan.STATUS_FIXED_LEN - len(expected))
        self.assertEqual(support.encoding(encoding_plan), expected)
