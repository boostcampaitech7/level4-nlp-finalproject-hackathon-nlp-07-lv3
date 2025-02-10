# NOTA 오디오 언어모델의 경량 모델링 레서피 탐구

# 1. Introduction
Audio adapter의 결합 및 사전학습을 통해, 언어모델은 음성/음악/환경음 등의 소리를 이해하고 다양한 downstream task를 수행할 수 있게 되었습니다.

VRAM의 크기가 작은 전형적인 디바이스 환경에서는 오디오 언어모델에 대한 경량 모델링이 필수적입니다.

본 프로젝트는 오디오 언어 모델들을 사용하여 ASR, Audiocaps 등의 다양한 오디오 문제를 한번에 해결하는 모델을 제작하는 프로젝트입니다.


# 2. Teams & Schedule
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

## Team member Role
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

## Project Calender
1월 10일 (금) 10:00 ~ 2월 10일 (월) 18:00
<div align='center'>
  
![timeline](./img/timeline.png)

</div>

---
# 3. How to use

1. ``git clone https://github.com/boostcampaitech7/level4-nlp-finalproject-hackathon-nlp-07-lv3.git``
2. Move to project directory
3. ``pip install -r ./requirements.txt``
4. ``python asr_inference.py`` for inference asr tasks
5. ``python aac_inference.py`` for inference aac tasks

# How to use config.yaml
1. wandb : wandb 기록에 대한 설정
2. model : 사용할 모델 및 Q-Former, LoRA 등의 구성
3. datasets : 학습 및 추론에 사용하는 데이터 경로
4. run : 학습 방법(배치 크기, 분산 학습, AMP, Optimizer)

---
# 4. Model Architecture
```plaintext  
1. openai/whisper-large-v3-turbo
2. CED Small 
3. Qwen/Qwen2.5-0.5B-Instruct
4. Window-Level Q-Former
5. LoRA
```
## Key Features
- 본 모델은 멀티모달 처리 능력을 갖추어 음성, 이미지, 텍스트 입력을 통합하여 이해하고 생성할 수 있음.
- 여러 모델을 조합하여 음성 인식 및 언어 이해를 강화하고, LoRA를 활용해 경량화된 학습이 가능함.
- STT, QA, 요약, 번역 등 다양한 멀티모달 NLP 및 음성 관련 작업을 지원함.
- 분산 학습 및 혼합 정밀도(Amp) 지원으로 효율적인 모델 훈련이 가능함.

---
# 5. Project Structure
```plaintext
📦level4-nlp-finalproject-hackathon-nlp-07-lv3
 ┣ 📂src
 ┃ ┣ 📂configs
 ┃ ┃ ┣ 📜eval_config.yaml
 ┃ ┃ ┗ 📜train.yaml
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📂beats
 ┃ ┃ ┃ ┣ 📜backbone.py
 ┃ ┃ ┃ ┣ 📜BEATs.py
 ┃ ┃ ┃ ┣ 📜modules.py
 ┃ ┃ ┃ ┣ 📜quantizer.py
 ┃ ┃ ┃ ┣ 📜Tokenizers.py
 ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📂CED
 ┃ ┃ ┃ ┗ 📂models
 ┃ ┃ ┃ ┃ ┣ 📜audiotransformer.py
 ┃ ┃ ┃ ┃ ┣ 📜checkpoints.py
 ┃ ┃ ┃ ┃ ┣ 📜ensemble.py
 ┃ ┃ ┃ ┃ ┣ 📜layers.py
 ┃ ┃ ┃ ┃ ┗ 📜__init__.py
 ┃ ┃ ┣ 📜modeling_ced.py
 ┃ ┃ ┣ 📜modeling_llama.py
 ┃ ┃ ┣ 📜modeling_whisper.py
 ┃ ┃ ┣ 📜Qformer.py
 ┃ ┃ ┣ 📜salmonn.py
 ┃ ┃ ┣ 📜utils.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂prompts
 ┃ ┃ ┣ 📜test_prompt.json
 ┃ ┃ ┗ 📜train_prompt.json
 ┃ ┣ 📜config.py
 ┃ ┣ 📜dataset.py
 ┃ ┣ 📜dist_utils.py
 ┃ ┣ 📜evaluate.py
 ┃ ┣ 📜evaluate_efficiency_salmonn.py
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜optims.py
 ┃ ┣ 📜runner.py
 ┃ ┣ 📜salmonn_utils.py
 ┃ ┣ 📜submission_validator.py
 ┃ ┣ 📜train.py
 ┃ ┣ 📜utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📜aac_inference.py
 ┣ 📜asr_inference.py
 ┣ 📜Makefile
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┗ 📜run.py
```

## Evaluate Environment
```plaintext
CPU : Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
GPU : Tesla V100-SXM2-32GB x 2ea
RAM : 178GB
Nvidia Driver Version: 535.161.08   
CUDA Version: 12.2
```

## Final Score
```plaintext
ASR Score :  6.99%
AAC Score : 35.93%
Inference Speed : 0.1722(TTFT) + 0.038(TPOT) = 0.2102 second
VRAM Usage : 3.83 GB
```