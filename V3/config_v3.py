"""
V3 config shim.
Adds 00_config/ to sys.path so 'from config_v3 import ...' resolves
against V3/00_config/config.py — the single source of truth.
"""
import sys
from pathlib import Path

_cfg_dir = Path(__file__).parent / "00_config"
if str(_cfg_dir) not in sys.path:
    sys.path.insert(0, str(_cfg_dir))

# Re-export everything from 00_config/config.py
from config import *  # noqa: F401, F403
