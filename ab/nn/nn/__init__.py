try:
    from .rlfn import Net as Rlfn
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from rlfn import Net as Rlfn
from .rlfn import Net as Rlfn

