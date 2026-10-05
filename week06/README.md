# Week 6 · 문화콘텐츠 감정분석 Ⅱ: 머신러닝 기반 분석

**핵심 질문: 감정분석 모델의 성능은 어떻게 평가하고 개선할 수 있는가?**

TF-IDF → 로지스틱 회귀 → 검증 세트 비교 → 오분류 읽기 → 마지막 테스트 평가를 진행합니다.
모든 문장과 라벨은 이 수업을 위해 작성한 **합성 교육용 예제**입니다. 실제 관객 리뷰, NSMC, 크롤링 자료가 아닙니다.
API 키, 유료 서비스, 모델 다운로드 없이 실행합니다. 패키지 설치 때만 인터넷이 필요합니다.

## 1. 파일

- `week06_sentiment_lab.ipynb`: 한 단계씩 실행하는 학생용 노트북
- `train_sentiment.py`: 같은 분석을 한 번에 실행하는 스크립트
- `data/synthetic_reviews.csv`: 120개 합성 리뷰와 교육용 정답 라벨
- `data/README.md`: 라벨 기준, 그룹, 자료의 한계
- `requirements.txt`: 분석 패키지의 검증된 버전
- `requirements-notebook.txt`: 위 패키지 + VS Code 노트북 커널
- `tests/test_sentiment.py`: 누출·중복·지표·출력 테스트
- `INSTRUCTOR_NOTES.md`: 진행안, 예시 결과, 해석 질문의 답안
- `examples/`: 실제 실행해 저장한 합성 자료의 교사용 지표·오분류 예시

기존 저장소의 `requirmentsmac.txt`, `requirmentswin.txt`는 OS 도구 설치용 스크립트입니다.
이 폴더의 `requirements.txt`와는 용도가 다릅니다. 기존 수업 파일은 수정하지 않습니다.

## 2. 설치: 저장소 최상위 폴더에서 실행

저장소: <https://github.com/111usionBin/CNT3054>

이미 받은 저장소라면 해당 폴더를 열고, 처음이면 아래를 실행합니다.

```sh
git clone https://github.com/111usionBin/CNT3054.git
cd CNT3054
```

**Python 3.11 이상**을 사용합니다. Python 3.12.14 / scikit-learn 1.8.0 / pandas 2.2.3 / matplotlib 3.10.8 조합에서 검증했습니다.

### macOS · 터미널

```sh
python3 -m venv week06/.venv
week06/.venv/bin/python -m pip install -r week06/requirements-notebook.txt
week06/.venv/bin/python week06/train_sentiment.py
```

### Windows · PowerShell

```powershell
py -3.11 -m venv week06\.venv
.\week06\.venv\Scripts\python.exe -m pip install -r week06\requirements-notebook.txt
.\week06\.venv\Scripts\python.exe week06\train_sentiment.py
```

Windows에 Python 3.12만 설치되어 있다면 첫 줄을 `py -3.12 -m venv week06\.venv`로 바꿉니다.
가상환경을 직접 활성화하지 않으므로 PowerShell 실행 정책을 변경할 필요가 없습니다.
노트북을 쓰지 않으면 설치 명령의 파일을 `requirements.txt`로 바꿔도 됩니다.

### 기존 수업에서 uv를 설치했다면

```sh
uv venv week06/.venv --python 3.11
```

그다음 macOS는 `uv pip install --python week06/.venv/bin/python -r week06/requirements-notebook.txt`,
Windows는 `uv pip install --python week06\.venv\Scripts\python.exe -r week06\requirements-notebook.txt`를 실행합니다.
분석 실행 명령은 위 OS별 명령과 같습니다.

**VS Code 노트북:** `week06/week06_sentiment_lab.ipynb`를 열고 오른쪽 위 커널에서 `week06/.venv`의 Python을 선택합니다.
커널이 보이지 않으면 “Python Environments”에서 해당 실행 파일을 직접 선택합니다. 위에서 아래로 순서대로 실행합니다.
노트북의 기본 설정은 테스트 평가를 하지 않습니다.

## 3. 실습 순서

1. 데이터가 무엇에 대한 평가인지, `0=부정 / 1=긍정`의 기준을 확인합니다.
2. 관련 예제끼리 같은 그룹으로 묶어 학습 72 / 검증 24 / 테스트 24개로 분리합니다.
3. 학습 자료에서만 TF-IDF 어휘·IDF와 로지스틱 회귀 계수를 학습합니다.
4. 항상 다수 라벨을 내는 기준선과 unigram / unigram+bigram 후보를 검증 세트에서 비교합니다.
5. 검증 macro-F1이 높은 후보를 선택합니다. 동점이면 단순한 unigram을 선택합니다.
6. 검증 오분류를 원문과 함께 읽고 개선 가설을 기록합니다. bigram이 항상 더 좋은 것은 아닙니다.
7. 비교를 끝내고 모델·분리·기준을 고정한 뒤에만 테스트를 한 번 평가합니다.

`--evaluate-test`는 테스트 세트를 여는 **명시적인 학습용 스위치**입니다.
새 실행을 시작해도 같은 자료와 seed로 같은 분리가 재현되며, 테스트가 새로운 자료가 되지는 않습니다.
테스트 결과를 본 뒤 다시 설정을 고르면 그 테스트는 더 이상 독립적인 최종 평가가 아닙니다.
이 예제는 선택된 **학습 세트 전용 모델을 그대로** 테스트합니다. 학습+검증 재학습은 하지 않습니다.

```sh
# 아래 python 자리에 위 OS별 가상환경 실행 파일을 사용합니다.
python week06/train_sentiment.py
python week06/train_sentiment.py --evaluate-test
python week06/train_sentiment.py --no-plots
python week06/train_sentiment.py --output-dir week06/outputs/my_first_run
```

기본 출력은 `week06/outputs/run_날짜시간/`의 새 폴더입니다.
`--output-dir`를 지정하면 비어 있는 폴더만 허용하며 기존 결과는 덮어쓰지 않습니다.

## 4. 예상 결과와 읽는 법

제공된 CSV와 seed=42, 검증된 패키지 버전에서:

| 검증 세트 모델 | Accuracy | 긍정 Precision | 긍정 Recall | 긍정 F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| majority_baseline | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
| unigram | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| unigram_bigram | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |

최종 테스트의 선택된 unigram 모델은 Accuracy 0.8333, 긍정 Precision 0.7857, Recall 0.9167, F1 0.8462, Macro-F1 0.8322입니다.
이 수치는 **합성 자료의 실행 확인값**이며 실제 서비스 성능이나 관객 전체의 반응을 추정한 값이 아닙니다.
작은 검증 세트의 동점/순위는 불안정할 수 있습니다. 높은 점수를 얻는 것보다 평가 절차를 정확히 설명하는 것이 목표입니다.

테스트 혼동행렬은 `[[9, 3], [1, 11]]`입니다. **행=실제 라벨, 열=예측 라벨**, 순서는 `[0,1]`입니다.
따라서 TN=9, FP=3, FN=1, TP=11이고 긍정 정밀도는 `11/(11+3)`, 재현율은 `11/(11+1)`입니다.
`zero_division=0`을 명시하여 긍정 예측이 없는 기준선의 정의 불가능한 정밀도를 0으로 기록합니다.
Macro-F1은 두 라벨의 F1 평균이며 긍정 F1과 다릅니다.
기준선은 후보의 유용성을 점검하는 비교 대상입니다. 실제 자료에서 후보가 기준선보다 못하면 사용을 정당화할 수 없습니다.

## 5. 저장되는 결과

| 파일 | 확인할 내용 |
|---|---|
| `validation_scores.csv` | 기준선과 두 후보의 검증 지표 |
| `validation_predictions.csv` | 선택된 후보의 정답·예측·긍정 확률·정오 여부 |
| `validation_errors.csv` | 검증 오분류, FP/FN 유형 |
| `feature_weights.csv` | 모든 어휘의 계수, 양수는 긍정 방향 |
| `split_assignments.csv` | review_id / group_id별 분리 기록; 라벨은 싣지 않음 |
| `metrics.json` | seed, 자료 해시, 분리 수, 선택 기준, 패키지 버전 |
| `validation_confusion_matrix.png` | 검증 혼동행렬 |
| `feature_weights.png`, `top_features.csv` | 상·하위 계수 8개씩과 그림의 라벨 대응표 |

`--evaluate-test` 사용 시 `test_scores.csv`, `test_predictions.csv`, `test_errors.csv`,
`test_classification_report.txt`, `test_confusion_matrix.png`가 추가됩니다.
`--no-plots`에서는 PNG와 `top_features.csv`를 만들지 않습니다. 모든 계수는 `feature_weights.csv`에 남습니다.
CSV는 Excel에서 한글을 읽기 쉬운 UTF-8 BOM 형식입니다.
한글 폰트가 없으면 계수 그림에 `feature 01` 같은 번호를 쓰고 `top_features.csv`에서 원문을 확인합니다.
혼동행렬 축은 OS에 관계없이 영어와 숫자를 씁니다.

## 6. 직접 모은 CSV로 확장하기

API로 모은 원문만으로 지도학습을 할 수는 없습니다. 독립적인 사람 라벨과 명확한 기준이 필요합니다.
기존 `collect_comments.py`의 `text_raw`를 검토한 뒤 `text`로 정리하고 `label`을 붙입니다.
좋아요 수나 이 모델의 예측을 그대로 “정답”으로 사용하지 않습니다.

필수 열은 `text,label`이고 라벨은 정수 `0` 또는 `1`입니다. 선택 열은 `review_id,group_id,domain,source_type`입니다.
`review_id`가 없으면 자동 생성합니다. `group_id`가 없으면 정규화된 문장이 각 그룹이 됩니다.
같은 작품/작성자/수집 묶음이 양쪽에 섞이면 문제인 연구에서는 적절한 `group_id`를 직접 지정하세요.
중복 문장은 공백·대소문자·구두점 정규화 후 하나만 남깁니다. 같은 문장에 상충 라벨이 있으면 중단합니다.
중립·혼합·무관 문장은 억지로 0/1에 넣지 말고 별도로 검토하세요. 이 모델은 중립 클래스를 학습하지 않습니다.
최소한 각 라벨이 5개 이상의 다른 그룹에 있어야 하며, 모든 분리에 두 라벨이 남아야 합니다.
이 조건은 코드 실행의 최소 조건일 뿐 연구에 충분한 표본 수를 보장하지 않습니다.
불균형한 실제 그룹에서는 60/20/20 비율과 라벨 비율이 근사치입니다. 결과의 실제 분포를 점검하세요.

```sh
python week06/train_sentiment.py --csv week06/private_data/my_labeled_reviews.csv --output-dir week06/outputs/my_validation
```

`private_data/`, `outputs/`, `.venv/`는 이 폴더의 Git 제외 목록에 있습니다.
실제 댓글·이름·계정·민감한 원문은 공개 저장소에 올리지 마세요. 자료의 이용 조건과 재배포 권한을 확인하세요.
연구용 분석이라면 작성 시점, 작품, 작성자, 플랫폼을 고려한 독립 분리와 별도 도메인 평가를 설계해야 합니다.

## 7. 해석과 점검

- TF-IDF는 문맥 이해 모델이 아닙니다. 여기의 토큰화는 정규식 기반이며 한국어 형태소 분석이 아닙니다.
- 한 글자 `안`, `못`도 남기지만 활용형, 띄어쓰기, 반어, 대상 전환을 자동 해결하지 않습니다.
- `p_positive`는 모델의 긍정 클래스 확률 출력입니다. 감정 강도, 정답 확률 보증, 보정된 신뢰도가 아닙니다.
- `known_features=0`이면 학습 어휘와 일치하는 항목이 없어 절편만으로 예측합니다. 중립 판정이 아닙니다.
- 양의 계수는 다른 입력 특징을 고정했을 때 긍정 log-odds를 늘리는 방향입니다. 단어의 보편적 감성이나 인과효과가 아닙니다.
- 감동적인 비극에 대한 긍정 평가와 작품 속 슬픔을 구별하세요.
- 장르·표현이 달라지면 성능이 달라집니다. 이 실습에 BERT·LLM·API 호출은 포함하지 않습니다.

```sh
python -m unittest discover -s week06/tests -v
```

실행 검증: Linux/Python 3.12.14에서 스크립트, 모든 테스트, nbformat 구조 검증과 IPython 내 노트북 전체 코드 셀 실행을 확인했습니다.
이 환경의 로컬 소켓 제한으로 별도 Jupyter 커널 연결을 통한 실행은 검증하지 못했습니다.
Windows/macOS 명령은 경로·셸 문법에 맞춰 제공하지만 해당 OS에서 직접 실행 검증하지는 않았습니다.
오류가 나면 사용한 Python과 선택한 노트북 커널이 동일한 가상환경인지 먼저 확인하세요.

## 공식 참고 문서

- [TF-IDF](https://scikit-learn.org/1.8/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [LogisticRegression](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [StratifiedGroupKFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [데이터 누출 방지](https://scikit-learn.org/1.8/common_pitfalls.html)
- [분류 지표](https://scikit-learn.org/1.8/modules/model_evaluation.html#classification-metrics)
