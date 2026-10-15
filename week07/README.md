# 7주차 · 문맥 기반 감정분석과 텍스트 임베딩

**VS Code 데스크톱 + Python 터미널 실습**입니다. NVIDIA API로 zero-shot 감정분류를 수행하고 같은 리뷰를 사전·TF-IDF·BERT와 비교합니다. Word2Vec와 문장 임베딩도 확인합니다.

## 1. 준비와 파일 받기

- [VS Code](https://code.visualstudio.com/download), Python **3.12**를 설치합니다. Python 3.11도 사용할 수 있습니다. 기존 수업 환경이 있으면 그대로 사용하세요.
- VS Code 왼쪽 **Extensions**(확장) → `Python` 검색 → **Microsoft**가 제공하는 `ms-python.python` 설치. Python 실행기는 확장과 별도로 필요합니다.
- [실습 브랜치](https://github.com/111usionBin/CNT3054/tree/week07-nvidia-lab)에서 **Code → Download ZIP** → 압축 해제합니다. 저장소의 `main`에는 아직 이 자료가 없을 수 있습니다.
- VS Code **File → Open Folder…**(파일 → 폴더 열기)에서 압축 해제한 폴더 **안의 `week07`**을 선택합니다. 왼쪽에 `run_lab.py`, `lab.py`, `requirements-core.txt`, `data`가 보여야 합니다.
- **Terminal → New Terminal**(터미널 → 새 터미널)을 누릅니다. Windows는 터미널 오른쪽 `+` 옆 화살표 → **PowerShell**, macOS는 **zsh**를 사용합니다.

Git을 사용한다면 ZIP 대신 다음처럼 받습니다.

```sh
git clone --branch week07-nvidia-lab --single-branch https://github.com/111usionBin/CNT3054.git
cd CNT3054/week07
code .
```

## 2. 가상환경과 핵심 패키지

터미널의 현재 폴더가 `week07`인지 먼저 확인합니다. 명령은 **한 줄씩** 입력합니다. Python 3.12가 없다는 오류가 나면 설치를 먼저 확인하고 VS Code를 다시 엽니다.

**Windows / PowerShell**

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
```

**macOS / zsh**

```sh
python3.12 --version
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-core.txt
```

Python 3.11을 쓰는 경우 생성 명령의 `3.12`를 `3.11`로 바꿉니다. **Activate.ps1 실행이나 실행 정책 변경은 필요 없습니다.** 아래 모든 명령에서 가상환경의 Python 경로를 직접 사용합니다.

`Ctrl+Shift+P`(Mac `Cmd+Shift+P`) → **Python: Select Interpreter** → 이 폴더의 `.venv` 선택. 표시되지 않으면 **Enter interpreter path… → Find…**에서 Windows `.venv/Scripts/python.exe`, Mac `.venv/bin/python`을 고릅니다.

## 3. 기준선 실행 — API 키 없이 가능

```powershell
.\.venv\Scripts\python.exe run_lab.py baseline
```

Mac은 아래처럼 실행합니다. 이후 명령에서도 앞부분만 `./.venv/bin/python`으로 바꿉니다.

```sh
./.venv/bin/python run_lab.py baseline
```

터미널에 `학습: 120, 진단: 16`, 공통 12문장의 지표, 결과 폴더와 ZIP 경로가 출력됩니다. 왼쪽 Explorer에서 `outputs` 폴더를 펼치고 해당 `run_.../metrics.csv`를 누릅니다. CSV는 텍스트로도 열리며 별도 확장은 필수가 아닙니다.

## 4. NVIDIA 키 발급

1. [NVIDIA Build 모델 페이지](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b)를 열고 **Free Endpoint** 제공 여부를 확인합니다.
2. 계정 로그인, 모델 이용 조건 확인과 동의 등 본인 계정의 안내를 따릅니다.
3. **Get API Key / Generate API Key**에서 키를 발급받습니다. 메뉴 위치나 계정별 인증은 달라질 수 있습니다.
4. 키는 아래 실행 중 나오는 **숨김 입력 프롬프트**에만 붙여넣습니다. `.py`, 명령 문자열, GitHub, 스크린샷, 제출물에 키를 넣지 마세요.

2026-10-15 기준 공식 모델 페이지의 무료 엔드포인트 표시와 API 문서의 한국어 지원을 확인했습니다. 개발·평가용 제공 조건, 사용 한도와 모델은 변경될 수 있으며 무제한 무료를 뜻하지 않습니다. API가 불가능하면 7절의 BERT 대체 실습으로 진행합니다.

## 5. 프롬프트와 한 문장 연결 점검

```powershell
.\.venv\Scripts\python.exe run_lab.py prompt
.\.venv\Scripts\python.exe run_lab.py smoke
```

두 번째 명령의 `NVIDIA API key (hidden):`에서 키를 붙여넣고 Enter. **문자나 별표가 안 보여도 정상**입니다. 숨김 입력을 지원하는 **TERMINAL**에서 실행하세요. Output 패널의 Code Runner 실행은 사용하지 않습니다.

smoke는 `이 영화는 재미있지 않다.` 1문장을 실제 API에 전송합니다. 출력의 `status: ok`와 `llm_label`, 원문 연속 인용 `evidence`를 확인합니다. `ok`는 출력 형식과 인용 검사를 통과했다는 뜻이며 감정 정답을 보장하지 않습니다.

- API 주소: `https://integrate.api.nvidia.com/v1`
- 모델 ID: `nvidia/nemotron-3-ultra-550b-a55b`
- `openai` 패키지는 호환 API 클라이언트입니다. NVIDIA 키와 서버를 사용합니다.
- 입력은 지시문과 `{target, review}`뿐입니다. 정답 라벨과 입력-정답 예시를 보내지 않으므로 **zero-shot**입니다.
- `temperature=0`, `max_tokens=256`, thinking 비활성화. 서버 응답의 완전한 재현을 보장하지는 않습니다.

## 6. NVIDIA 16문장 비교와 결과 저장

```powershell
.\.venv\Scripts\python.exe run_lab.py analyze --nvidia
```

새 프로세스이므로 키를 다시 숨김 입력합니다. 기준선과 NVIDIA 결과를 하나의 표로 결합하고 지표와 ZIP을 저장합니다. 연결 점검 1건 + 본 실행 16건 = **17개 논리적 요청**입니다. 429/서버 오류 재시도 시 호출 수가 늘어납니다. API 호출이 중간에 중단되면 완료된 결과 ZIP이 만들어지지 않으므로 원인을 해결한 후 다시 실행합니다.

BERT까지 같은 실행에서 비교하려면 7절 패키지를 설치한 뒤 **위 명령 대신** `analyze --nvidia --bert`를 실행합니다. 이미 끝난 명령을 반복하면 API도 다시 호출됩니다.

## 7. 확장: BERT와 텍스트 임베딩

BERT·임베딩은 처음에 모델을 다운로드하며 핵심 API 실습보다 디스크와 시간이 더 필요합니다. GPU는 필수가 아니며 코드에서는 CPU를 사용합니다.

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-extra.txt
.\.venv\Scripts\python.exe run_lab.py analyze --bert
.\.venv\Scripts\python.exe run_lab.py embeddings
```

Mac:

```sh
./.venv/bin/python -m pip install -r requirements-extra.txt
./.venv/bin/python run_lab.py analyze --bert
./.venv/bin/python run_lab.py embeddings
```

Linux에서 CPU 전용 PyTorch를 설치하려면 먼저 `./.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu`를 실행한 뒤 확장 패키지를 설치합니다.

- **BERT**: `sangrimlee/bert-base-multilingual-cased-nsmc`, 한국어 영화 리뷰로 미세조정된 모델을 추론합니다. 이번 수업에서 재학습하지 않습니다. 라벨 매핑을 검사하며 0=부정, 1=긍정입니다. 선택 라벨의 softmax 점수는 감정의 세기가 아닙니다. 최대 256토큰을 사용합니다.
- **Word2Vec**: 120개 수업 문장, Skip-gram, 32차원. `영화는`의 벡터 `(32,)`와 이웃 단어를 확인합니다. 작은 합성 말뭉치의 이웃 순위를 한국어 전체 의미로 일반화하지 않습니다.
- **문장 임베딩**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 4문장 → `(4, 384)`. 정규화 벡터의 내적으로 코사인 유사도를 구합니다. 유사도가 감정 일치 확률은 아닙니다.

BERT 대체 실행은 키가 필요 없습니다. NVIDIA를 실행하지 않았다면 `nvidia_executed: false`로 기록되고 LLM 점수 행은 생성하지 않습니다.

## 8. 결과 읽기와 제출

매번 `outputs/run_실행시각/`과 같은 이름의 ZIP을 새로 만듭니다. 이전 결과를 덮어쓰지 않습니다. 터미널에 출력된 **이번 경로**를 확인하세요.

| 파일 | 내용 |
|---|---|
| `predictions.csv` | 원문·교육용 정답·방법별 결과. NVIDIA 실행 시 실제 응답, 근거, 상태 포함 |
| `metrics.csv` | 공통 긍정/부정 12개 지표 |
| `error_review.csv` | 적어도 한 방법이 틀리거나 보류한 이진 문장 |
| `run_metadata.json` | 실행 방법, 모델 설정, 프롬프트, 버전, 날짜, 데이터 해시. 키 제외 |

`embeddings`는 별도 `outputs/embeddings_.../`과 ZIP에 이웃 단어, 유사도 행렬, 문장 목록을 저장합니다.

비교 조건:

- 학습 120개와 진단 16개는 합성 자료이며 동일 원문 중복은 검사합니다.
- 공통 지표는 긍정·부정 12개로 계산합니다. 실패·중립·혼합·보류도 분모에서 빼지 않습니다. 나머지 중립·혼합 4개는 정성 분석용입니다.
- 모든 방법에 같은 리뷰를 주지만 LLM만 `target`을 별도 필드로 받습니다. 사전학습 자료와 규모도 다릅니다. 알고리즘 전체 순위를 매기는 벤치마크가 아닙니다.
- 검증 예시: 사전 정확도 0.500, TF-IDF 0.833, BERT 1.000. 12문장에서의 결과이며 일반 성능을 의미하지 않습니다. NVIDIA 실제 점수는 각자의 키로 실행한 결과로 채웁니다.

제출: 결과 ZIP, 방법별 판단이 다른 문장 2개의 원문과 오류 설명, 팀 프로젝트 적용 계획 3~5문장. API를 미실행했다면 그 사실을 적습니다.

## 9. 문제 해결

| 증상 | 확인 |
|---|---|
| `py` / `python3.12` 없음 | Python 설치·버전 확인 후 VS Code 재시작. 설치된 3.11이면 생성 명령 수정 |
| `requirements-core.txt` / `run_lab.py` 없음 | `week07` 폴더를 열었는지, ZIP 압축을 풀었는지 확인 |
| `ModuleNotFoundError` | 같은 `.venv` Python으로 `-m pip install`을 실행했는지 확인 |
| `.venv` 선택 목록에 없음 | Python: Select Interpreter → Enter interpreter path…로 직접 지정 |
| 키 입력이 안 보임 | 숨김 입력의 정상 동작. 붙여넣고 Enter. Output 아닌 TERMINAL 사용 |
| HTTP 401 / 403 | 키, 계정 권한, 모델 이용 동의 확인 |
| HTTP 404 | 모델 ID와 무료 엔드포인트의 현재 제공 여부 확인 |
| HTTP 429 / 5xx | 반복 실행을 멈추고 잠시 뒤 재시도. 코드는 최대 2회 재시도 |
| `invalid_json` / `invalid_evidence` | `raw_output`과 원문을 비교. 실패를 임의로 중립으로 바꾸지 않음 |
| BERT 다운로드/추론 지연 | 최초 모델 다운로드와 CPU 추론 대기. 저장 공간·연결 상태 확인 |

## 파일과 검증 범위

`run_lab.py`가 표준 수업 진입점, `lab.py`가 분석 구현입니다. `requirements.txt`는 전체 패키지 설치용으로 유지합니다. 이전 `.ipynb`는 참고용이며 이번 강의 흐름에서는 사용하지 않습니다.

기준선·BERT·임베딩은 Linux CPU에서 실행 검증했습니다. Windows/macOS 명령은 공식 Python/VS Code 방식에 맞춰 제공하며 해당 OS에서 직접 실행한 결과로 표시하지 않습니다. NVIDIA 유료/무료 인증 실호출은 제작 환경에서 수행하지 않았고 요청 구성과 파싱은 모의 응답으로 검사했습니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

참고: [VS Code Python](https://code.visualstudio.com/docs/python/python-tutorial), [환경 선택](https://code.visualstudio.com/docs/python/environments), [통합 터미널](https://code.visualstudio.com/docs/terminal/basics), [NVIDIA Quickstart](https://docs.api.nvidia.com/nim/docs/api-quickstart), [모델 API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b).
