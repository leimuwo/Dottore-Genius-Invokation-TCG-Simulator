"""Equivalent mapping lookup, without repeated imports and inheritance dispatch."""
from enum import Enum

from src.dgisim.encoding.encoding_plan import EncodingPlan
from src.dgisim.state.enums import Pid


class FastEncodingPlan(EncodingPlan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._flat_codes = {}
        for name in ('mode', 'card', 'enum', 'char', 'effect', 'status', 'summon', 'support'):
            self._flat_codes.update(kwargs[name + '_mapping'])

    def encode_item(self, item):
        if isinstance(item, Pid) and self._perspective is Pid.P2:
            item = item.other
        key = item if isinstance(item, (type, Enum)) else type(item)
        try:
            return self._flat_codes[key]
        except KeyError:
            # Keep the original diagnostic and its type checking for unknown values.
            return super().encode_item(item)
