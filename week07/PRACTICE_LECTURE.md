# 7주차 실습: 문맥 기반 감정분석과 텍스트 임베딩
CNT3054 문화콘텐츠와 자연어처리 · 장수빈

## 1. 분석 질문
같은 리뷰를 사전, TF-IDF, BERT, LLM으로 분석하면 판단이 어떻게 달라지는가? ‘재미있다’와 ‘재미있지 않다’를 구분하는 데 필요한 정보를 확인한다. 모델별 점수뿐 아니라 부정 범위, 반어, 평가 대상이 결과에 미치는 영향을 설명한다.

## 2. 실습 자료
| 파일 | 용도 |
|---|---|
| `data/train_reviews.csv` | TF-IDF 분류기와 Word2Vec에 사용하는 합성 학습 문장 120개 |
| `data/context_reviews.csv` | 문맥 현상을 포함한 합성 진단 문장 16개 |
| `data/student_annotation.csv` | 모델 실행 전에 학생이 직접 판정하는 빈 라벨 파일 |
| `validate_data.py` | CSV 구조와 학습/진단 중복 확인 |
| `run_lab.py` | 터미널 실행 진입점 |
| `lab.py` | 전처리, 예측, 평가, 결과 저장 구현 |

합성 문장은 수업용으로 작성한 예시이며 실제 이용자의 의견이나 대표성 있는 표본이 아니다. 진단 세트를 반복해서 보며 모델을 조정하면 독립적인 최종 평가 자료로 사용할 수 없다.

## 3. VS Code와 기존 uv 환경
기존 수업 폴더를 연다. `.venv`와 `week07` 폴더가 같은 수준인지 확인한다. Python 인터프리터도 기존 `.venv`를 선택한다. 아래 명령은 저장소 최상위 폴더의 통합 터미널에서 실행한다.

```sh
uv --version
uv run --no-sync python -c "import sys; print(sys.executable)"
uv pip install --python .venv -r week07/requirements-core.txt
uv run --no-sync python week07/validate_data.py
```

Python 경로와 검증 완료 메시지를 확인한다. 파일을 더블클릭하거나 VS Code의 Output 창에서 실행하지 않는다. API 키의 숨김 입력에는 Terminal 창이 필요하다.

## 4. 사람의 판정
`student_annotation.csv`의 `human_label`과 `reason`을 채운다. 긍정은 평가 대상에 대한 호감이 우세한 경우, 부정은 불만이 우세한 경우이다. 평가는 없고 사실만 있으면 중립, 긍정과 부정 중 어느 쪽도 우세하지 않으면 혼합으로 적는다. 작품 속 인물의 감정과 작성자의 작품 평가를 구분한다. 먼저 각자 판정한 후 의견이 다른 문장의 기준을 논의한다.

## 5. 사전과 TF-IDF 기준선
```sh
uv run --no-sync python week07/run_lab.py baseline
```

사전 방식은 정해진 긍정·부정 표현을 센다. TF-IDF 방식은 학습 문장의 특징과 라벨 관계를 로지스틱 회귀로 학습한다. 학습 데이터로 벡터화와 분류기를 적합한 뒤 진단 문장을 예측한다. 터미널에 나온 이번 결과 폴더에서 `predictions.csv`와 `metrics.csv`를 연다. 부정어 문장에서 두 방법의 결과를 비교한다.

## 6. NVIDIA zero-shot
[모델 페이지](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b)에서 현재 제공 상태와 계정 조건을 확인하고 API 키를 발급한다. 무료 제공과 사용량은 계정 및 서비스 정책에 따라 달라질 수 있다. 키를 소스나 캡처에 넣지 않는다.

```sh
uv run --no-sync python week07/run_lab.py prompt
uv run --no-sync python week07/run_lab.py smoke
uv run --no-sync python week07/run_lab.py analyze --nvidia
```

`prompt`는 호출 없이 지시문을 보여준다. `smoke`는 한 문장을 실제 전송한다. 숨김 입력에서 키가 보이지 않는 것은 정상이다. `analyze --nvidia`는 진단 문장 16개를 전송한다. 정답 라벨은 전송하지 않는다. JSON의 라벨과 원문 인용을 검사하지만, 형식 통과가 정답을 뜻하지는 않는다. API를 사용할 수 없으면 다음 BERT 경로로 진행한다.

## 7. BERT 추론
```sh
uv pip install --python .venv -r week07/requirements-extra.txt
uv run --no-sync python week07/run_lab.py analyze --bert
```

처음에는 모델 다운로드가 필요하며 CPU 추론에 시간이 걸릴 수 있다. 이미 한국어 영화 리뷰로 미세조정된 모델을 사용하므로 이번 실습에서 BERT를 새로 학습하지 않는다. 긍정·부정만 출력하는 모델이므로 중립과 혼합 문장에서 강제로 이진 판정을 할 수 있다. softmax 점수는 실제 정답 확률로 검증된 신뢰도와 구분한다.

## 8. 단어·문장 임베딩
```sh
uv run --no-sync python week07/run_lab.py embeddings
```

Word2Vec은 수업 문장으로 단어 벡터를 학습한다. 자료가 작아 이웃 단어가 일반적인 의미 관계를 잘 반영하지 못할 수 있다. 문장 임베딩은 사전학습 모델을 사용한다. `sentence_similarity.csv`에서 ‘재미있다’와 ‘재미있지 않다’의 유사도를 확인한다. 주제와 단어가 비슷하면 감정이 반대여도 유사도가 높을 수 있다. 의미 유사도를 감정 일치로 해석하지 않는다.

## 9. 코드의 처리 순서
`load_data`가 자료를 읽고, `run_baselines`가 기준선을 실행한다. 선택한 옵션에 따라 BERT 또는 NVIDIA 결과를 결합한다. `compare_metrics`가 동일한 이진 문장에 대한 지표를 계산하고, `save_results`가 결과와 실행 조건을 저장한다. 예측값을 정답에 맞춰 수동으로 수정하지 않는다.

## 10. 결과 해석
공통 점수는 긍정·부정 12문장에서 계산한다. 중립·혼합 4문장은 별도로 읽는다. 이진 판정률은 유효한 긍정·부정 판정을 낸 비율이다. 정확도는 보류와 실패를 포함한 전체 평가 문장 중 맞힌 비율이다. Macro-F1은 두 클래스의 F1을 평균한다. 높은 점수만 제시하지 말고 어떤 문맥에서 판단이 달라졌는지 확인한다.

`error_review.csv`에서 두 문장을 고르고 원문, 정답, 각 모델의 예측을 나란히 적는다. 부정어가 적용되는 범위나 접속어 뒤의 최종 평가를 근거로 차이를 설명한다. 결과가 같은 문장도 근거가 적절한지 확인한다.

## 11. 보고서
조별 대표 1명이 PDF 한 개를 제출한다. 분량은 표지 없이 2페이지 이내이다. 1쪽에는 데이터, 라벨 기준, 실행한 방법, 판정률·정확도·Macro-F1 비교를 적는다. 2쪽에는 판단이 달랐던 문장 두 개와 문맥 설명, 팀 데이터에 적용할 때의 한계를 적는다. 실행하지 않은 모델의 예시 점수는 결과로 쓰지 않는다. 캡처에는 결과 표와 실행 조건이 보이도록 하고 API 키는 포함하지 않는다. 제출 기한은 수업 공지를 따른다.

## 12. 오류 대처와 참고자료
모듈 오류가 나면 패키지를 설치한 Python과 실행 Python이 같은지 확인한다. 401·403은 키와 권한, 404는 모델 ID, 429는 사용량 제한을 확인한다. 호출을 연속 재실행하기보다 오류를 기록한다. 모델 다운로드가 막히면 기준선 결과까지 저장하고 실행하지 못한 방법을 명시한다.

- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. https://aclanthology.org/N19-1423/
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT. https://aclanthology.org/D19-1410/
- [BERT 체크포인트와 사용법](https://huggingface.co/sangrimlee/bert-base-multilingual-cased-nsmc)
- [NVIDIA 모델 API 공식 문서](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b)
