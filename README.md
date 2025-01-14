# 1. Settings

1. `conda create -n 가상환경이름 python==3.9.17`

2. `git clone https://github.com/boostcampaitech7/level4-nlp-finalproject-hackathon-nlp-07-lv3.git`

3. `git fetch --all`

4. `git switch feature/base_line`
- feature/base_line 으로 브랜치 이동

5. `make setup`
- pre-commit 등 기타 설정 초기화
- requirements.txt 설치
등등

6. src/data 안에 데이터셋 + 파싱해주는 json 전부 넣어두기
- stage(1,2) train.json의 경우 샘플로 넣어둔 것
- 아래 json 전부 있는 것을 가정한 코드

  ![image](https://github.com/user-attachments/assets/b0ff51a2-d00f-4973-be2f-655bdb292cf2)


---
- sample dataset(6G)과 NOTA측에서 제공한 기본 모델은 아래 경로에 올려뒀습니다.
  - https://drive.google.com/drive/u/0/folders/1WppT1b4goghsOI8BXZBCldordnO_M-cd

- json 파일은 아래 데이터셋 참고할 것
  - https://huggingface.co/datasets/lifelongeeek/salmonn_dataset_annotation

---

# 2. train
0. src폴더로 경로를 들어가서 `python3 train.py --cfg-path configs/train.yaml 입력 후 CLI에서 wandb 기록이름 설정
1. train.yaml에서 관련 설정을 체크해줍니다.
2. 본 train 과정은 이전과 다르게 stage-1, stage-2 에 따른 config.yaml을 따로 받지 않고 train.yaml으로 통일하였습니다.
3. 공통으로 사용되는 인자들은 그대로 두었습니다. 다만, stage-1, stage-2에 따라서 optim은 다소 다를 수 있겠다고 판단되어 해당 부분과 output_dir은 남겼습니다.
4. 모델 저장 메트릭도 매 에포크마다 도는 val_data에 대한 성능 측정 결과를 기준으로 best모델과 마지막 epoch에 해당하는 모델만 저장되도록 하였습니다.


# 3. evaluate
- `eval_config.yaml`에서 stage2 마친 가중치 가져와서 경로 설정해주고
- `datasets`에서 `test_ann_path` 에 `/data/test_asr.json` 또는 `/data/test_aac.json` 설정해준 뒤에
- `src` 폴더 경로 들어와서 아래와 같이 실행
`python3 evaluate.py --cfg-path configs/eval_config.yaml --skip_scoring`

- 최종적으로 submission.csv 생성된 것 확인

### evaluate 관련 에러노트
`python3 evaluate.py --cfg-path configs/eval_config.yaml` 실행하면
아래와 같이 에러 발생
```
level4-nlp-finalproject-hackathon-nlp-07-lv3/src/evaluate.py", line 112, in main
ref = samples["text"]
KeyError: 'text'
```

`python3 evaluate.py --cfg-path configs/eval_config.yaml --task asr(또는 aac)` 실행하면
아래와 같이 에러발생
```
level4-nlp-finalproject-hackathon-nlp-07-lv3/src/salmonn_utils.py", line 115, in __getitem__
entity["text"] = ann["text"]
KeyError: 'text'
```

 
