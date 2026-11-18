import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_tmp = tempfile.mkdtemp(prefix="drivio-test-")
os.environ.setdefault("DRIVIO_DATA_DIR", _tmp)
os.environ.setdefault("DRIVIO_SIMULATE", "1")
os.environ.setdefault("DRIVIO_AUDIO", "0")
