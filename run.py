import subprocess
import sys

if __name__ == "__main__":
    #subprocess.run(["torchrun", "--nproc_per_node", "2",  "src/train.py", "--cfg-path", "src/configs/train.yaml"])
    subprocess.run([f'{sys.executable}', "src/train.py", "--cfg-path", "src/configs/train.yaml"])
    #subprocess.run([f'{sys.executable}', "src/evaluate.py", "--cfg-path", "src/configs/eval_config.yaml", "--mode", "submission_asr", "--skip_scoring"])
    #subprocess.run([f'{sys.executable}', "src/evaluate.py", "--cfg-path", "src/configs/eval_config.yaml", "--mode", "submission_aac", "--skip_scoring"])
    #subprocess.run([f'{sys.executable}', "src/evaluate_efficiency_salmonn.py", "--cfg-path", "src/configs/eval_config.yaml"])
