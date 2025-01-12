import subprocess

if __name__ == "__main__":
    subprocess.run(["python", "src/train.py", "--cfg-path", "src/configs/train_stage1.yaml"])
