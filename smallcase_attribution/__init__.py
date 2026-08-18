"""
smallcase_attribution
=====================
Deterministic pipeline that answers three separate questions about one smallcase:

  A. STRATEGY        - how did the smallcase model portfolio perform?
  B. IMPLEMENTATION  - how did the investor's actual holdings perform?
  C. NET OUTCOME     - what was left after attributable costs and estimated tax?

Design rules
------------
* No stock names, dates, quantities or smallcase identifiers are hardcoded.
* Every number is either derived from the input files or explicitly flagged
  as unavailable. Nothing is forced to reconcile.
* The module performs the calculations; interpretation is left to a later layer.
"""
__version__ = "1.0.0"

from .config import Config
from .pipeline import run

__all__ = ["Config", "run", "__version__"]
