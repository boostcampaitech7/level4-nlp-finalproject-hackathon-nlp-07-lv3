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
    - beats_path의 경우 기본값: NOTA에서 제공한 BEATs
    - ckpt의 경우 stage1에서는 원래대로라면 훈련된 가중치가 없는 것이 맞으나, `훈련 도중 끊긴 체크포인트` 혹은 `다른 곳에서 얻은 가중치`가 있다면 기입 가능
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

## 2.2 stage-2: AAC
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

 
# NOTA 오디오 언어모델의 경량 모델링 레서피 탐구

## 대회 소개
Audio adapter의 결합 및 사전학습을 통해, 언어모델은 음성/음악/환경음 등의 소리를 이해하고 다양한 downstream task를 수행할 수 있게 되었습니다.
</br>VRAM의 크기가 작은 전형적인 디바이스 환경에서는 오디오 언어모델에 대한 경량 모델링이 필수적입니다.

본 프로젝트는 오디오 언어 모델들을 사용하여 ASR, Audiocaps 등의 다양한 오디오 문제를 한번에 해결하는 모델을 제작하는 프로젝트입니다.


## 팀원
<h2 align="center">NLP-7조 NOTY</h3>
<table align="center">
  <tr height="100px">
    <td align="center" width="150px">
      <a href="https://github.com/Uvamba"><img src="https://avatars.githubusercontent.com/u/116945517?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/doraemon500"><img src="https://avatars.githubusercontent.com/u/64678476?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/simigami"><img src="https://avatars.githubusercontent.com/u/46891822?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/DDUKDAE"><img src="https://avatars.githubusercontent.com/u/179460223?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/mrsuit0114"><img src="https://avatars.githubusercontent.com/u/95519378?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/hskhyl"><img src="https://avatars.githubusercontent.com/u/155405525?v=4"/></a>
    </td>
  </tr>
  <tr height="10px">
    <td align="center" width="150px">
      <a href="https://github.com/Uvamba">강신욱</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/doraemon500">박규태</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/simigami">이정민</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/DDUKDAE">장요한</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/mrsuit0114">한동훈</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/hskhyl">홍성균</a>
    </td>
  </tr>
</table>

## 팀원 역할
<div align='center'>

| 팀원  | 역할                                                        |
|-----|-----------------------------------------------------------|
| 홍성균 | Team Leader,           |
| 강신욱 |                                     |
| 박규태 |                    |
| 이정민 | Nvidia Canary, Optimization                               |
| 장요한 |                        |
| 한동훈 |  |


</div>

## 프로젝트 기간
1월 10일 (금) 10:00 ~ 2월 10일 (월) 18:00

## 프로젝트 진행 과정.
<div align='center'>
  
![timeline](./img/timeline.png)

</div>

# 설정 및 사용법

### 1. ``git clone https://github.com/boostcampaitech7/level4-nlp-finalproject-hackathon-nlp-07-lv3.git``
### 2. Move to project directory
### 3. ``pip install -r ./requirements.txt``
### 4. ``python asr_inference.py`` for inference asr tasks
### 4. ``python aac_inference.py`` for inference aac tasks

# config.yaml 사용법


# 주요 기능


# 프로젝트 구조

```plaintext
CSAT-Solver/
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── models/                 # LoRA 학습된 adapter 저장 디렉토리
│
├── output/                 # test.csv로 inference 결과 저장 디렉토리
│
├── src/
│   ├── arguments.py        # 학습에 필요한 여러 인자
│   ├── utils.py            # 시드 고정 및 데이터 셋 chat message 형태로 변환
│   ├── streamlit_app.py    # EDA 시각화 제공 프로그램
│   ├── main.py             # 모델 학습 및 추론
│   ├── ensemble.py         # 추론 결과 앙상블
│   ├── backtranslation_augmentation.py # 역번역 증강
│   └── retrieval_tasks/    # RAG 파이프라인 코드 디렉토리
│       ├── __init__.py
│       ├── LLM_tasks.py          # LLM을 활용한 파이프라인 중의 요약 및 확인 task
│       ├── retrieval.py          # 리트리버 공통 추상 클래스
│       ├── retrieval_semantic.py # Dense 
│       ├── retrieval_syntactic.py # Sparse
│       ├── retrieval_hybrid.py   # 하이브리드 서치 
│       ├── retrieval_rerank.py   # two-stage reranker 
│       └── retrieve_utils.py     # RAG 기반 검색·요약 파이프라인
│
├── requirements.txt
├── README.md
└── run.py                  # 실행 파일
```
---

## Evaluate Environment
```plaintext
CPU : Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
GPU : Tesla V100-SXM2-32GB x 2ea
RAM : 178GB
Nvidia Driver Version: 535.161.08   
CUDA Version: 12.2
```

## Score
```plaintext
ASR Score :  6.99%
AAC Score : 35.93%
Inference Speed : 0.1722(TTFT) + 0.038(TPOT) = 0.2102
VRAM Usage : 3.83 GB
```