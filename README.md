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

6. src/data 안에 학습 데이터 통째로 넣어두기
- stage(1,2) train.json의 경우 샘플로 넣어둔 것
- 학습 데이터 폴더 넣었다면, 이후에 그에 맞춰 train.json도 변경해야 함

---
sample dataset(6G)과 NOTA측에서 제공한 기본 모델은 아래 경로에 올려뒀습니다.
https://drive.google.com/drive/u/0/folders/1WppT1b4goghsOI8BXZBCldordnO_M-cd


---

# 2. train

## 2.1 stage-1: ASR (기본적인 Text 전사학습)
### 2.1.1 train_stage1.yaml 에서 경로 및 하이퍼 파라미터 설정
  - **model**
    - beats_path의 경우 기본값: NOTA에서 제공한 BEATA
    - ckpt의 경우 stage1에서는 원래대로라면 훈련된 가중치가 없는 것이 맞으나, 훈련 도중 끊긴 체크포인트 혹은 다른 곳에서 얻은 가중치가 있다면 기입 가능
  - **datasets**
    - train/valid/test_ann_path 의 경우에 현재 train.json 밖에 명시적이게 없는데, valid, test.dataset이 없으면 에러가 나서 임시적으로 코드상에서 `train.yaml` `train.py`에서 주석처리, `runner.py`에서 별도 로직 만들어서 train을 valid, test로 쪼개는 방식으로 에러 피해놨음.
    추후에 valid, test 어떤 걸로 할지 정해지면 해당 부분 주석 풀고 로직 수정하면 됨.
      ```
        # train.py 주석부분
        # build datasets
        datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        # "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        # "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    }
      ```
      ```
        # runner.py 로직 부분(train_dataset으로 valid, test 전부 만들기)
        # datasets["train"]는 SALMONNDataset 인스턴스
        train_dataset = datasets["train"]

        # 데이터셋을 train, validation, test로 나누기
        train_dataset, valid_dataset, test_dataset = split_salmonn_dataset(
            train_dataset, val_ratio=0.2, test_ratio=0.5
        )
      ```
  - **run**
    - 현재 기본값은 single GPU에 맞춰져 있음 분산환경에서는 원래의 값으로 변경하면 됨.

### 2.1.2 stage-1 train
`src` 폴더로 경로 들어와서
`python3 train.py --cfg-path configs/train_stage1.yaml` 실행

학습 완료 이후 `outputs_stage1` 폴더 만들어진 것 확인하고 안에 가중치 체크(`.pth`)

## 2.2 stage-2: ACC
- stage1 에서 만들어진 모델 가중치를 `train_stage2.yaml`에서 적절히 경로 설정하여 받아줌,
- 이외에는 `train_stage1.yaml` 설정과 대동소이
- 학습 종료 후 `outputs_stage2` 폴더 만들어진 것 확인하고 안에 가중치 체크(`.pth`)

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

 
