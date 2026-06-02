"""Direct variant of :class:`legofmt.main.generate.GenerateOut` — uses
:class:`legofmt.main.modules_direct.LEGOLtng` as the flow component."""

from ..main.generate import GenerateOut as _GenerateOut
from ..main.modules_direct import LEGOLtng


class GenerateOut(_GenerateOut):
    flow_cls = LEGOLtng
