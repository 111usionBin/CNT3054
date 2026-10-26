# 9주차 실습: LDA로 문화콘텐츠 리뷰의 주제 찾기
CNT3054 문화콘텐츠와 자연어처리 · 장수빈

## 1. 실습의 범위
CSV를 넣으면 한국어 전처리, 정수 빈도 행렬 생성, LDA 추정, 토픽별 단어·문서별 비율 저장을 차례로 수행한다. API 키와 GPU는 필요 없다. 처음 패키지를 설치할 때 인터넷 연결이 필요하며 이후 분석과 HTML 결과 열기는 오프라인으로 가능하다. LDA 추론은 scikit-learn의 batch 변분추론이며 깁스 샘플링이 아니다.

## 2. 파일 받기와 기존 환경
[저장소 main](https://github.com/111usionBin/CNT3054/tree/main)에서 Code > Download ZIP을 선택하고 압축을 푼다. `week09` 폴더만 기존 수업 폴더에 복사한다. 기존 `.venv`를 지우거나 새로 만들지 않는다. Git으로 관리해 온 학생은 변경 파일을 확인한 뒤 `git pull --ff-only`로 업데이트할 수 있다.

VS Code에서 File > Open Folder로 `.venv`와 `week09`를 담은 수업 폴더를 연다. Terminal > New Terminal을 선택한다. 이후 명령은 모두 이 수업 폴더에서 실행한다. Windows와 macOS에서 동일한 명령을 사용한다.

```sh
uv --version
uv run --no-sync python -c "import sys; print(sys.executable)"
uv pip install --python .venv -r week09/requirements.txt
```

Python 경로가 기존 `.venv` 안을 가리키는지 확인한다. VS Code의 Python: Select Interpreter에서도 같은 환경을 선택한다. `--no-sync`는 기존 환경을 자동으로 다시 동기화하지 않는 옵션이다. `pyproject.toml`이 없는 폴더에서는 `--no-sync`가 적용되지 않는다는 경고가 나올 수 있다. 실행 Python 경로가 `.venv`인지 확인한다. 다른 주차에서 `uv sync`를 실행한 뒤 패키지가 사라졌다면 위 설치 명령을 다시 실행한다.

## 3. 예제 데이터 확인
`data/culture_reviews.csv`의 한 행은 리뷰 한 건이다. 학습에 사용하는 열은 `text`뿐이다. 감정 라벨과 정답 토픽은 필요 없다. 데이터는 168개 합성 리뷰이며 실제 이용자 의견이나 모집단을 대표하지 않는다. 자세한 설명은 [데이터 설명](data/README.md)을 읽는다.

## 4. 첫 분석
```sh
uv run --no-sync python week09/run_lda.py
```

기본값은 K=4, alpha=0.1, eta=0.1, seed=42, 30회 반복이다. 기본 Kiwi 전처리에서는 명사·고유명사·외국어 중 두 글자 이상을 사용한다. 기본 예제는 검증 환경에서 168개 유효 문서와 84개 어휘를 얻었다. 패키지 버전이 바뀌면 결과가 달라질 수 있으므로 자신의 출력값을 기록한다.

터미널의 `결과:` 뒤 경로를 확인한다. Explorer에서 `week09/outputs/run_.../index.html`을 찾는다. 파일을 우클릭해 운영체제의 파일 탐색기에 표시한 후 더블클릭하여 브라우저에서 연다. VS Code 편집기에서 HTML 소스가 보이는 것과 브라우저의 결과 화면을 구분한다. 실행마다 새 폴더를 만들므로 이전 결과를 덮어쓰지 않는다.

## 5. 결과 파일
| 위치 | 내용 |
|---|---|
| `index.html` | 실행 요약과 K별 결과 링크 |
| `comparison.csv` | K별 학습 퍼플렉시티와 상위 단어 다양성 |
| `excluded_documents.csv` | 빈 값·중복·남은 어휘 없음으로 제외한 행 |
| `run_metadata.json` | 입력 해시, 버전, 전처리 및 모형 설정 |
| `k4/report.html` | 읽기용 보고서, 네트워크 없이 열기 가능 |
| `k4/topics.csv` | 토픽별 상위 10개 단어와 확률 |
| `k4/topic_word_distribution.csv` | 전체 어휘에 대한 토픽별 단어 확률 |
| `k4/document_topics.csv` | 문서별 모든 토픽 비율, 가장 높은 토픽 |
| `k4/representative_docs.csv` | 각 토픽 비율이 높은 문서 3개 |
| `k4/topic_summary.csv` | 문서 비율의 단순 평균, 최다 토픽 기준 문서 수 |
| `k4/tokens_preview.csv` | 실제 사용한 전처리 토큰 |

문서별 토픽 비율은 각 문서에서 합이 1이다. 전체 단어 분포는 각 토픽에서 합이 1이다. 상위 10개 단어만 표시한 확률의 합은 1보다 작을 수 있다. 문서 평균 비중과 가장 높은 토픽으로 분류한 문서 수는 서로 다른 집계이다. 감정 점수로 해석하지 않는다.

## 6. 토픽 이름 붙이기
상위 단어를 읽고 대표 문서 3개를 확인한다. 단어와 원문이 뒷받침하는 범위로 이름을 정한다. 하나의 토픽에 음악과 영상 표현이 섞일 수 있고, 한 문서가 여러 토픽의 대표 문서로 나타날 수 있다. `dominant_topic`만 보지 말고 전체 비율을 확인한다. 토픽 번호는 고유한 주제 이름이 아니다.

## 7. 토픽 수 비교
```sh
uv run --no-sync python week09/run_lda.py --compare 2 4 6
```

같은 전처리·어휘·seed·alpha·eta 조건에서 K만 바꾼다. 새 결과 폴더의 `index.html`에서 각 K를 열고 토픽이 합쳐지거나 나뉘는 양상을 비교한다. 서로 다른 K의 topic_1은 같은 주제라고 가정하지 않는다.

`training_perplexity`는 학습 문서 자체에서 계산한 참고값이다. 낮은 값만으로 최적 K를 선택하지 않는다. `top_word_diversity`는 상위 단어 중 고유 단어가 차지하는 비율이며, 의미적 일관성 점수가 아니다. 이 실습은 탐색 분석으로, 미관측 문서에 대한 일반화 성능 평가를 포함하지 않는다.

## 8. 조별 데이터 넣기
`data/my_reviews_template.csv`에 `text` 열을 유지하면서 실제 텍스트를 넣고 `my_reviews.csv`로 저장한다. CSV UTF-8로 저장하며 한 행을 한 문서로 만든다. 최소 5개의 유효 문서가 필요하지만 이는 실행 조건일 뿐 충분한 표본 수의 기준이 아니다.

```sh
uv run --no-sync python week09/run_lda.py --input week09/data/my_reviews.csv --topics 4
```

텍스트 열 이름이 `review`라면 다음과 같이 지정한다.

```sh
uv run --no-sync python week09/run_lda.py --input week09/data/my_reviews.csv --text-column review --topics 4
```

개인정보와 공개하면 안 되는 원문은 업로드하지 않는다. 코드는 로컬 분석만 수행하며 외부 API로 텍스트를 전송하지 않는다. 파일명에 공백이 있으면 경로를 큰따옴표로 감싼다.

## 9. 전처리와 설정 변경
`tokens_preview.csv`에서 조사 제거와 명사 추출을 확인한다. `stopwords.txt`에 한 줄씩 단어를 추가할 수 있다. 자주 등장한다는 이유만으로 분석 질문에 중요한 단어를 지우지 않는다.

```sh
uv run --no-sync python week09/run_lda.py --topics 4 --alpha 0.5 --eta 0.1
```

alpha는 문서별 토픽 비율의 사전분포, eta는 토픽별 단어 확률의 사전분포에 영향을 준다. 비교할 때 한 번에 하나의 설정을 바꾸고 `run_metadata.json`을 남긴다.

## 10. 문제 해결
| 증상 | 대처 |
|---|---|
| 파일을 찾을 수 없음 | 터미널 위치와 `week09` 폴더 확인 |
| ModuleNotFoundError | 실행 Python 경로와 requirements 설치 확인 |
| 텍스트 열 없음 | CSV 헤더 확인, `--text-column` 지정 |
| 문자 깨짐 | CSV UTF-8로 다시 저장 |
| 남은 어휘 없음 | 불용어 확인 후 `--min-df 1 --max-df 1.0`으로 진단 |
| 유효 문서 5개 미만 | 빈 값·중복·짧은 문서를 확인하고 자료 보충 |
| Kiwi 설치 실패 | Python 3.10~3.12 환경 여부 확인. 필요한 경우 `--tokenizer regex`로 단순 분리하되 조사 제거가 없다는 한계 기록 |
| HTML 소스만 보임 | 파일을 브라우저로 열기 |

## 11. 분석 기록 구성
분석 기록은 다음 항목으로 정리한다. 제출 형식과 기한은 수업 공지를 따른다. 데이터 출처와 문서 단위, 전처리·K·alpha·eta·seed, 토픽 이름과 대표 단어·문서, K 비교와 선택 이유, 해석의 한계를 포함한다. 합성 예제를 사용했다면 실제 여론 분석으로 표현하지 않는다.

## 12. 검증과 참고
```sh
uv run --no-sync python -m unittest discover -s week09/tests -v
```

- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3, 993–1022. https://www.jmlr.org/papers/v3/blei03a.html
- [scikit-learn LDA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [Kiwi Python API](https://github.com/bab2min/kiwipiepy)
