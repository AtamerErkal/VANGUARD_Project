"""
VANGUARD AI — Backend launcher
Run this instead of `python -m uvicorn`:

    python run_server.py
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so MODELS_DIR resolves correctly
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import uvicorn
import backend.main as app_module   # triggers model load in THIS process

print(f"[VANGUARD] Model ready: {app_module._model_ready}")
print(f"[VANGUARD] Models dir : {app_module.MODELS_DIR}")

if not app_module._model_ready:
    print("[VANGUARD] ERROR — model failed to load. Run: python retrain.py")
    sys.exit(1)

uvicorn.run(app_module.app, host="127.0.0.1", port=8000, reload=False)
