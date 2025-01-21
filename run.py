import os
import subprocess
from datetime import datetime, timedelta

py_path = "src/distill_train.py"
config_path = "src/configs/distillation.yaml"

subprocess.run([
    "python", py_path,
    "--cfg-path", config_path,
], check=True)