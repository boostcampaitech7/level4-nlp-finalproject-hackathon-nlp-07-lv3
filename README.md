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

6. src/data 안에 데이터 파싱해주는 json 넣어두기
- json 파일은 경로 존재하지 않는 부분 제거한 구글 드라이브 참고할 것 (링크된 이슈 참고)
  - https://github.com/boostcampaitech7/level4-nlp-finalproject-hackathon-nlp-07-lv3/issues/14

7. .env 파일을 프로젝트 루트 디렉토리에 생성하고 안에 2가지 키를 적으셔야 합니다.
- HF_KEY='huggingface api key'
- WANDB_KEY='wandb api key'

---

# 2. train
0. src폴더로 경로를 들어가서 `python3 train.py --cfg-path configs/train.yaml` 입력 후 CLI에서 wandb 기록이름 설정
1. train.yaml에서 관련 설정을 체크해줍니다.
2. 본 train 과정은 이전과 다르게 stage-1, stage-2 에 따른 config.yaml을 따로 받지 않고 train.yaml으로 통일하였습니다.
3. 공통으로 사용되는 인자들은 그대로 두었습니다. 다만, stage-1, stage-2에 따라서 optim은 다소 다를 수 있겠다고 판단되어 해당 부분과 output_dir은 남겼습니다.
4. 모델 저장 메트릭도 매 에포크마다 도는 val_data에 대한 성능 측정 결과를 기준으로 best모델과 마지막 epoch에 해당하는 모델만 저장되도록 하였습니다.


# 3. evaluate
- `eval_config.yaml`에서 stage2 마친 가중치 가져와서 경로 설정해주고
- 자신의 모델에 맞게 기타 config 설정 완료해준 뒤에
- `src` 폴더 경로 들어와서 아래와 같이 실행
  - aac 제출용 `python3 evaluate.py --cfg-path configs/eval_config.yaml --mode submission_aac`
  - asr 제출용 `python3 evaluate.py --cfg-path configs/eval_config.yaml --mode submission_asr`

- 최종적으로 submission.csv 생성된 것 확인

# 4. evaluate_efficiency
`python3 evaluate_efficiency_salmonn.py --cfg-path configs/eval_config.yaml`

---

## 분산환경 학습
- train.yaml 파일에 `use_distributed: True` 체크하고
`torchrun --nproc_per_node=2 train.py --cfg-path configs/train.yaml `

## CED 설정
- train.yaml 파일 중 model의 beats_path에 'ced_base' 'ced_small' 'ced_mini' 'ced_tiny' 중 하나를 입력하면 자동으로 zenodo에서 미리 올라간 해당 모델의 가중치를 다운로드하여 로드 합니다.
