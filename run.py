import subprocess

if __name__ == "__main__":
    subprocess.run(["python", "src/train.py", "--cfg-path", "src/configs/train_stage1.yaml"])
    #subprocess.run(["python", "src/train.py", "--cfg-path", "src/configs/train_stage2.yaml"])
    #subprocess.run(["python", "src/evaluate.py", "--cfg-path", "src/configs/eval_config.yaml", "--skip_scoring"])

