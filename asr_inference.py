import subprocess
import sys


if __name__ == "__main__":
    subprocess.run([f'{sys.executable}', "src/evaluate.py", "--cfg-path", "src/configs/eval_config.yaml", "--mode", "submission_asr", "--skip_scoring"])