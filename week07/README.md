# 7주차 문맥 기반 감정분석과 텍스트 임베딩

첫 실습에서 만든 **VS Code + uv 환경**을 사용합니다. 같은 한국어 리뷰를 사전, TF-IDF, NVIDIA zero-shot으로 분류하고 BERT와 텍스트 임베딩을 비교합니다.

**제출물은 실제 결과를 해석한 A4 2쪽 내외 보고서 PDF 1개입니다.**

## 1. 기존 수업 폴더에 week07 추가

1. [실습 브랜치](https://github.com/111usionBin/CNT3054/tree/week07-nvidia-lab)를 열고 브랜치 이름을 확인합니다.
2. **Code → Download ZIP**을 누르고 압축을 풉니다.
3. 압축을 푼 폴더 안의 **week07**을 첫 실습에서 사용한 수업 폴더 안에 복사합니다.
4. VS Code의 **File → Open Folder**에서 기존 수업 폴더를 엽니다.
5. Explorer에서 기존 **.venv**와 **week07**이 같은 수준에 있는지 확인합니다. week07 안에는 `run_lab.py`, `lab.py`, `data`, `requirements-core.txt`가 있어야 합니다.
6. **Terminal → New Terminal**을 누릅니다. 현재 경로는 **.venv와 week07을 담은 수업 폴더**입니다.

이미 최신 week07 파일을 보유하고 있다면 해당 파일을 사용합니다. 아래 명령은 기존 가상환경 이름이 `.venv`인 구성을 기준으로 합니다. 다른 이름을 사용했다면 그 환경의 경로를 지정하고 같은 환경으로 실행하세요.

`Ctrl+Shift+P`(Mac `Cmd+Shift+P`) → **Python: Select Interpreter**에서 기존 `.venv`의 Python을 선택합니다. 목록에 없으면 **Enter interpreter path**에서 기존 실행 파일을 지정합니다.

## 2. uv 환경 확인과 핵심 패키지

Windows와 macOS 모두 수업 폴더에서 한 줄씩 실행합니다.

```sh
uv --version
uv run --no-sync python -c "import sys; print(sys.executable)"
uv pip install --python .venv -r week07/requirements-core.txt
```

출력된 Python 경로가 기존 `.venv` 안의 실행 파일을 가리키는지 확인합니다. `.venv`가 없으면 첫 실습에서 사용한 수업 폴더를 다시 확인하세요. `--no-sync`는 현재 설치한 환경을 그대로 사용하며 이번 패키지는 `uv pip install`로 추가합니다.

이 방식은 기존 `pyproject.toml`과 `uv.lock`을 수정하지 않습니다. 다른 주차 작업에서 `uv sync`를 실행한 뒤 이번 패키지가 없어졌다면 위 설치 명령을 다시 실행합니다.

## 3. 사전과 TF-IDF 기준선

```sh
uv run --no-sync python week07/run_lab.py baseline
```

`학습: 120, 진단: 16`과 공통 12문장의 지표, 결과 폴더 경로를 확인합니다. Explorer에서 **week07 → outputs → 이번 run_... 폴더 → metrics.csv**를 엽니다. 매번 새 결과 폴더에 저장합니다.

## 4. NVIDIA API 키와 연결 점검

1. [모델 페이지](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b)에서 **Free Endpoint** 제공 여부를 확인합니다.
2. 본인 계정으로 로그인하고 입출력 처리 안내와 이용 조건을 읽습니다. 동의하는 경우 화면의 안내에 따라 진행합니다.
3. **Get API Key / Generate API Key**에서 키를 발급받습니다.
4. 키는 명령 실행 중 나오는 숨김 입력에 붙여넣습니다. 코드, 캡처, 보고서에는 넣지 마세요.

2026-10-15에 공식 페이지의 무료 엔드포인트 표시와 한국어 지원을 확인했습니다. 실행 시점의 이용 조건과 한도는 모델 페이지에서 확인합니다. API 사용이 어려우면 BERT로 비교를 수행하고 이유를 보고서에 적습니다.

```sh
uv run --no-sync python week07/run_lab.py prompt
uv run --no-sync python week07/run_lab.py smoke
```

`prompt`는 API 호출 없이 과제 설명을 출력합니다. 모델에는 지시문과 `{target, review}`를 보내며 정답 라벨과 입력-정답 예시는 전송하지 않습니다.

`smoke`는 `이 영화는 재미있지 않다.` 한 문장을 실제 API에 전송합니다. `NVIDIA API key (hidden):`에서 키를 붙여넣고 Enter를 누르세요. **문자나 별표가 보이지 않는 상태가 정상**입니다. VS Code의 **TERMINAL**에서 실행합니다.

응답의 `status`, `llm_label`, `evidence`를 확인합니다. `status: ok`는 출력 형식과 인용 검사를 통과했다는 뜻입니다. 감정 라벨의 적절성은 원문과 함께 검토합니다.

- API 주소: `https://integrate.api.nvidia.com/v1`
- 모델: `nvidia/nemotron-3-ultra-550b-a55b`
- `openai` 패키지는 NVIDIA의 호환 API를 호출하는 클라이언트입니다.
- 설정: `temperature=0`, `max_tokens=256`, thinking 비활성화. 서버 결과는 실행에 따라 달라질 수 있습니다.

## 5. NVIDIA로 16문장 분류

```sh
uv run --no-sync python week07/run_lab.py analyze --nvidia
```

키를 다시 숨김 입력합니다. 기준선과 NVIDIA 결과를 결합하고 CSV와 실행 기록을 결과 폴더에 저장합니다. 연결 점검 1건과 본 실행 16건은 **17개 논리적 요청**이며 재실행과 오류 재시도에 따라 실제 호출 수가 늘어납니다. 중간에 호출이 중단되면 완성된 비교 결과가 저장되지 않으므로 원인을 확인한 뒤 다시 실행하세요.

## 6. BERT와 텍스트 임베딩

기존 수업 폴더에서 확장 패키지를 추가합니다.

```sh
uv pip install --python .venv -r week07/requirements-extra.txt
uv run --no-sync python week07/run_lab.py analyze --bert
uv run --no-sync python week07/run_lab.py embeddings
```

BERT와 임베딩은 CPU로 실행하며 처음에 모델을 다운로드합니다. API와 BERT를 함께 비교할 때는 아래 명령을 사용합니다. NVIDIA API가 새로 호출됩니다.

```sh
uv run --no-sync python week07/run_lab.py analyze --nvidia --bert
```

- **BERT:** `sangrimlee/bert-base-multilingual-cased-nsmc`. 한국어 영화 리뷰로 미세조정된 모델을 추론합니다. 0=부정, 1=긍정이며 최대 256토큰을 사용합니다. `bert_score`는 선택한 라벨의 softmax 점수입니다.
- **Word2Vec:** 수업 문장 120개로 Skip-gram 32차원 벡터를 학습합니다. `영화는`의 벡터 `(32,)`와 이웃 단어를 확인합니다.
- **문장 임베딩:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`가 네 문장을 `(4, 384)` 벡터로 바꾸고 코사인 유사도를 계산합니다.

NVIDIA 미실행 시 `nvidia_executed: false`가 기록되고 LLM 지표 행은 생성되지 않습니다.

## 7. 결과 파일 읽기

터미널의 **이번 실행 경로**를 확인합니다. 분석 결과는 `week07/outputs/run_실행시각/`에 저장합니다.

| 파일 | 보고서에 사용할 정보 |
|---|---|
| `metrics.csv` | 방법별 이진 판정률, 정확도, Macro-F1 |
| `predictions.csv` | 원문, 교육용 정답, 방법별 라벨. NVIDIA 실행 시 실제 응답과 근거 포함 |
| `error_review.csv` | 적어도 한 방법이 틀리거나 보류한 이진 문장 |
| `run_metadata.json` | 실행 방법, 모델 설정, 프롬프트, 버전, 날짜, 데이터 해시 |

임베딩은 별도 `week07/outputs/embeddings_.../`에 이웃 단어, `sentence_similarity.csv`, 문장 목록을 저장합니다.

공통 지표는 긍정과 부정으로 구성된 **합성 진단 문장 12개**에서 계산합니다. 보류와 실패도 분모에 포함합니다. 중립·혼합 4개는 문장 해석에 사용합니다. LLM에는 평가 대상을 별도 필드로 전달하며 모델마다 학습 자료와 규모도 다릅니다.

실행 예시의 정확도는 사전 0.500, TF-IDF 0.833, BERT 1.000입니다. 이 수치는 12개 문장에서의 결과입니다. 보고서에는 본인의 실제 실행값을 기록합니다.

## 8. 제출 보고서

**A4 2쪽 내외 보고서 PDF 1개**를 제출합니다. 첫 쪽 위에 이름과 학번을 적고 별도 표지 없이 본문을 시작합니다.

### 1쪽: 실행 조건과 지표 비교

- 사용한 모델, 실행 날짜, API 실행 여부를 3~4줄로 적습니다.
- 사전과 TF-IDF에 **NVIDIA 또는 BERT**를 더해 이진 판정률, 정확도, Macro-F1을 표로 비교합니다.
- 차이가 큰 지표를 수치로 설명합니다. 판정률이 낮으면 보류나 실패 건수도 확인합니다.
- 공통 평가 대상이 긍정·부정 합성 문장 12개라는 점을 적습니다.

해석 예시: 사전의 이진 판정률 0.667은 12개 중 8개에서 유효한 긍정/부정 결과를 냈다는 뜻입니다. 정확도 0.500은 전체 12개 중 6개를 맞혔다는 뜻입니다. 본인의 지표와 실제 보류·오류 사례를 연결해 설명하세요.

### 2쪽: 문장 사례와 해석

- 방법별 판단이 달랐던 문장 **2개**를 따옴표로 인용하고 각 방법의 라벨을 적습니다.
- 부정 범위, 반어, 평가 대상 중 관련된 문맥을 구체적으로 설명합니다.
- LLM을 실행했다면 `evidence`의 실제 인용 여부와 라벨의 타당성을 검토합니다.
- 끝에 팀 자료의 라벨 기준과 사람 검토 방법을 **3~4문장**으로 제안합니다.

표나 캡처를 넣었다면 본문에서 수치와 의미를 설명합니다. API 미실행 시 이유와 대체 방법을 명시합니다. 실행하지 않은 모델의 예시 점수를 본인 결과로 적지 마세요.

## 9. 문제 해결

| 증상 | 확인할 내용 |
|---|---|
| uv 또는 .venv 확인 실패 | 첫 실습의 수업 폴더와 uv 터미널 확인 |
| run_lab.py 파일 없음 | 수업 폴더 안의 week07과 명령의 경로 확인 |
| ModuleNotFoundError | `uv pip install --python .venv -r ...`로 설치, 실행의 `--no-sync` 확인 |
| 다른 Python 경로 출력 | 기존 수업 폴더와 환경 선택 확인 |
| 키가 보이지 않음 | 정상적인 숨김 입력. TERMINAL에서 붙여넣고 Enter |
| HTTP 401 / 403 | 키, 계정 권한, 모델 이용 동의 확인 |
| HTTP 404 | 모델 ID와 엔드포인트 제공 상태 확인 |
| HTTP 429 / 서버 오류 | 반복 실행 중단, 코드의 최대 2회 재시도 후 상태 확인 |
| invalid_json / invalid_evidence | raw_output과 원문 비교, 실패 상태 기록 |
| 모델 다운로드 지연 | 연결 상태와 저장 공간 확인, 다운로드와 CPU 추론 대기 |

표준 진입점은 `run_lab.py`, 분석 구현은 `lab.py`입니다. 기존 노트북은 참고용입니다. 분석 구현은 Linux CPU에서 검증했고 NVIDIA 인증 실호출 결과는 본인 계정으로 생성합니다.

참고: [uv 환경](https://docs.astral.sh/uv/pip/environments/), [uv 실행과 동기화](https://docs.astral.sh/uv/concepts/projects/sync/), [VS Code Python](https://code.visualstudio.com/docs/python/python-tutorial), [NVIDIA Quickstart](https://docs.api.nvidia.com/nim/docs/api-quickstart), [모델 API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b).
