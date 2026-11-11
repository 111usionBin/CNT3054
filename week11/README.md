# 11주차 BERTopic 실습

조별로 수집한 댓글·대사·제목 CSV를 분석하고 클러스터 원문을 검수합니다.

## 시작

VS Code에서 수업 폴더를 엽니다. 터미널에서 `cd week11`을 실행합니다.
`uv venv .venv --python 3.11`로 환경을 만들고 다음 명령으로 설치합니다.

```
uv pip install -r requirements.txt
uv run --no-project python bertopic_lab.py inspect
uv run --no-project python bertopic_lab.py run
```

명령은 모두 `week11` 폴더에서 실행합니다. `uv run`은 이 폴더의 `.venv`를 사용합니다.

## 입력과 설정

- `data/my_data.csv`: 조별 수집 CSV. 한 행은 댓글 하나, 대사 하나, 제목 하나 등으로 통일합니다.
- `config.json`: `input`, `text_column`을 자신의 파일에 맞춥니다. 나머지는 먼저 기본값을 사용합니다.
- UTF-8 저장이 어렵다면 `encoding`을 `cp949`로 바꿉니다.
- 빈 본문을 제외하고 기본 설정에서는 완전히 같은 본문을 하나만 남깁니다.
- 길이 한도를 넘는 문서가 있으면 실행을 중단합니다. 긴 글은 문장/문단으로 나누고 원문 연결 정보를 보관합니다.
- 재실행 시 `output`을 `outputs_v2` 등 새 폴더로 바꿔 기존 검수를 보존합니다.

## 임베딩 모델

`SentenceTransformer(config['model'])`로 모델 이름을 전달하면 처음에 자동으로 내려받습니다.
Hugging Face 사이트에서 직접 파일을 받을 필요가 없습니다. BERTopic은 설치하는 Python 패키지입니다.
기본 모델은 한국어를 포함한 다국어 문장을 처리하는 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`입니다.
참고 코드의 `dragonkue/BGE-m3-ko`도 `model`에 지정할 수 있으나 더 많은 메모리와 실행 시간이 필요합니다.
모델을 바꾸면 임베딩과 군집을 다시 계산합니다. `.cache`의 모델과 임베딩은 제출하지 않습니다.

## 결과 검수

`outputs/review.md`를 VS Code에서 열고 `Ctrl+Shift+V`(Mac: `Cmd+Shift+V`)로 미리보기를 엽니다.

- `clusters.csv`: 군집별 문서 수·전체 분석 문서 대비 비율·대표 단어.
- `documents.csv`: 분석한 전체 원문과 클러스터 번호, 원본 데이터 행 번호.
- `review_samples.csv`: 군집별 중심 3개·주변 1개·남은 문서 중 무작위 1개. 작은 군집은 가능한 만큼만 표시.
- `cluster_review.csv`: 이름, 공통 특징, 근거 행, 예외, 해석을 기록할 빈 칸.
- `excluded.csv`: 빈 글과 중복으로 제외한 행.
- `run_summary.json`: 전체 군집 수, 미배정 비율, 모델·설정·패키지 버전.

군집 수에서 `-1`은 제외합니다. `-1` 문서는 삭제하지 않고 따로 읽습니다.
중심/주변은 원래 임베딩의 군집 평균과 코사인 유사도로 선정하며, 정답 확률이 아닙니다.
대표 단어만 보고 이름을 확정하지 말고 원문과 예외를 확인합니다.
`min_cluster_size`는 군집의 최소 크기이고 토픽 개수가 아닙니다.

## 제출

보고서 drop-box에 조별 보고서를 제출합니다. 다음 내용을 포함합니다.

- 데이터 출처·수집 범위·분석 단위·원본/분석/제외 건수.
- 모델과 설정, 전체 클러스터 수, 미배정(-1) 건수와 비율.
- 전체 클러스터의 이름·크기·대표 특징을 요약한 표.
- 주요 클러스터의 원문 근거, 공통점과 예외, 해석. 예외가 많거나 이름을 붙이기 어려운 군집도 기록.
- 수집 범위와 모델 결과를 고려한 해석의 한계.

검수 후 ZIP을 만듭니다.

```
uv run --no-project python bertopic_lab.py pack --group 1jo
```

생성된 `week11_1jo_날짜_시각.zip`은 결과물 drop-box에 제출합니다.
ZIP에는 실행 당시 코드·입력 CSV·설정·결과 CSV·검수 내용이 들어갑니다.
보고서는 ZIP과 구분하여 보고서 drop-box에 제출합니다.

ZIP에서 재실행하려면 `results` 폴더에서 `config_used.json`의 input을 `input_snapshot.csv`, output을 `rerun_outputs`로 바꾼 뒤 환경을 준비합니다.
`uv run --no-project python bertopic_lab_used.py run --config config_used.json`으로 실행합니다. 모델 버전과 패키지 정보는 run_summary.json을 참고합니다.

## 참고

- https://maartengr.github.io/BERTopic/
- https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- https://huggingface.co/dragonkue/BGE-m3-ko
- https://www.sbert.net/docs/package_reference/sentence_transformer/model.html

발표 화면의 실행 검증에는 NSMC의 `ratings_test.txt` 처음 800행을 사용했습니다.
출처: https://github.com/e9t/nsmc . 감정 라벨은 군집화에 사용하지 않았습니다.
조별 제출에는 각 조가 수집한 데이터를 사용합니다.
