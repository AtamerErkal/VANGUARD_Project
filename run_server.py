"""
VANGUARD AI — Backend launcher

Local:   python run_server.py
Render:  automatically uses PORT env var and binds to 0.0.0.0
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import uvicorn
import backend.main as app_module

print(f"[VANGUARD] Model ready: {app_module._model_ready}")
print(f"[VANGUARD] Models dir : {app_module.MODELS_DIR}")

if not app_module._model_ready:
    print("[VANGUARD] ERROR — model failed to load. Run: python retrain.py")
    sys.exit(1)

# Render injects PORT; locally defaults to 8000
host = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
port = int(os.environ.get("PORT", 8000))

print(f"[VANGUARD] Starting on {host}:{port}")
uvicorn.run(app_module.app, host=host, port=port, reload=False)
