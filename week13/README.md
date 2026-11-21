# CNT3054 · 문화콘텐츠와 자연어처리
## 13주차 | 한 공연을 두 언어모델로 읽기

공연 중계의 “제가 객석 마이크를 올릴게요”는 음향 감독의 말일까? 스피커 고장에 대응해 새로운 방법이 등장한 순간일까? 같은 방송을 한국어 RoBERTa와 GPT-OSS로 읽고, 두 답이 갈린 장면을 시각과 함께 찾아본다. Windows PowerShell + VS Code 기준이다.

### 1. 결과부터 열어 보기

1. GitHub Desktop에서 `CNT3054`의 `main`을 선택하고 **Fetch origin → Pull origin**으로 갱신한다. **Open in Visual Studio Code**로 저장소를 연 뒤 **파일 → 폴더 열기**에서 `lab.py`가 있는 `week13`을 선택한다.
2. **터미널 → 새 터미널**을 눌러 아래 명령을 실행한다. 첫 번째 명령은 실습 전용 파이썬을 준비한다. 12주차와 같은 uv를 사용한다.

```powershell
uv venv --python 3.12
.\.venv\Scripts\python.exe lab.py demo
```

3. 왼쪽 파일 목록에서 `outputs/demo/predictions.csv`를 연다. 위쪽에서 **U0006**을 찾는다. `source_speaker`는 12주차 번호이고, `speaker`는 화면의 `model`이 추정한 번호다. 이 발화의 서사 판정은 모의 예제에서 BERT가 ‘승’, GPT가 ‘전’이다.
4. `examples/utterances.json`의 22~28초 발화를 다시 읽는다. 새 행동을 ‘전’으로 볼지, 앞선 요청을 실행하는 ‘승’으로 볼지 팀의 기준으로 골라 본다.

`demo`는 작성된 응답을 재생하므로 API 키와 인터넷이 필요 없다(파이썬 설치 후). `examples/actual_bert/`에는 같은 창작 전사를 **실제 모델 API로 실행한 결과**가 따로 있다. 모의 예제 수치로 성능을 보고하지 않는다.

### 2. 무료로 두 API 준비하기

| API | 학생이 준비할 것 | 데이터와 비용 |
|---|---|---|
| 한국어 BERT 계열 | 한국어 자연어 추론 모델 Huffon/klue-roberta-base-nli를 PC에서 실행 | 최초에 가중치 약 443MB(약 422MiB)와 패키지를 받는다. 이후 같은 PC의 HTTP API를 호출한다. GPU·키·호출 요금이 필요 없다. |
| GPT 계열 | GroqCloud 무료 계정의 API 키와 OpenAI 공개 가중치 모델 openai/gpt-oss-20b | Groq 서버로 대상 발화·앞뒤 문맥·역할 소개·기준표를 보낸다. 공개 또는 실습에 제공할 수 있는 전사를 사용한다. |

Groq Console에 로그인해 **API Keys → Create API Key**를 고른다. 키는 팀 PC의 `week13/.env`에만 적는다. Deepgram 키는 12주차 음성 전사용이다. Free Plan 문서의 기본 조직 한도는 이 모델 기준 분당 30요청·8,000토큰, 하루 1,000요청·200,000토큰이다. 실제 값은 계정 Limits 메뉴가 기준이다. 같은 조직을 여러 조가 쓰면 사용량도 합쳐진다. 수업 기본값은 앞 12발화이고 GPT 요청은 발화당 1건이다. 무료 한도는 바뀔 수 있다.

[Groq API 키](https://console.groq.com/keys) · [무료 한도](https://console.groq.com/docs/rate-limits) · [모델 지원](https://console.groq.com/docs/model/openai/gpt-oss-20b) · [공개 한국어 모델](https://huggingface.co/Huffon/klue-roberta-base-nli)

```powershell
uv pip install --python .venv -r requirements-bert.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

메모장에서 `GROQ_API_KEY=` 뒤에 본인 키를 붙여 넣고 저장한다. 새 파일 이름은 정확히 `.env`다. Windows에서 가상환경을 활성화할 필요 없이 아래처럼 실행할 수 있다. macOS/Linux에서는 `.\.venv\Scripts\python.exe`를 `./.venv/bin/python`으로 바꾸고 `.env.example`을 `.env`로 복사한다.

### 3. 예제 전사를 실제 API로 보내기

**터미널 A**에서 한국어 모델 서버를 켠다. 최초 모델 다운로드가 끝나고 `Application startup complete`가 보이면 이 창을 열어 둔다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn bert_api:app --host 127.0.0.1 --port 8013
```

**터미널 B**를 새로 열어 같은 폴더에서 아래 명령을 실행한다. 첫 시도는 3개만 보내도 된다.

```powershell
.\.venv\Scripts\python.exe lab.py run --input examples/utterances.json --profiles examples/speakers.json --gold examples/gold.csv --limit 3 --out outputs/first3
```

| 화면에 보이는 내용 | 할 일 |
|---|---|
| 3/3 U0003, API 실행 완료 | `outputs/first3`의 CSV를 연다. |
| BERT 연결 실패 | 터미널 A의 서버가 켜졌는지 확인한다. 브라우저에서 http://127.0.0.1:8013/docs 를 열어 같은 PC에서 접속되는지 본다. |
| GPT 401 / 429 | 401이면 Groq에서 받은 키와 .env 위치를 확인한다. 429이면 요청·토큰 한도가 회복된 뒤 같은 명령을 다시 실행한다. 저장 완료된 응답은 재사용한다. |
| 모델 다운로드가 어렵다 | 같은 명령에 `--backend gpt`를 붙이고 새 출력 폴더를 지정한다. 이 경우 `requirements.txt`만 설치해도 된다. |
| GPT 키를 받지 못했다 | `--backend bert --out outputs/bert_only`로 BERT를 실제 호출한다. 두 모델의 비교 과정은 `demo`로 연습할 수 있다. |

### 4. 12주차에 수집한 우리 조 전사로 바꾸기

1. 12주차 `week12/outputs/실행시각/utterances.json`을 이번 폴더로 복사해 `team.json`으로 이름을 바꾼다. 동일한 열을 가진 `utterances.csv`도 지원한다. 영상이나 음성을 다시 올릴 필요가 없다.
2. 후보 화자 목록과 사람 평가표를 만든다.

```powershell
.\.venv\Scripts\python.exe lab.py prepare --input team.json --out team_labels
```

3. `team_labels/speakers.json`을 VS Code로 연다. 원영상의 별도 구간을 보고 번호별 역할을 1~2개의 짧은 문장으로 쓴다. 예를 들어 “악기와 마이크를 담당한다. 음량을 조정한다.”처럼 표현한다. 평가할 발화의 인용이나 ID를 소개에 넣으면 답이 새어 나간다. 영상이 바뀌면 같은 번호여도 인물이 바뀔 수 있다.
4. 모델 결과를 열기 전에 두 명이 앞 12발화를 각각 읽고 불일치를 조정해 `gold.csv`에 **speaker, emotion, stage**를 적는다. 이 실습의 화자 정답은 **텍스트로 귀속할 수 있는 후보**다. 영상에서 신원이 확인돼도 “네.”처럼 텍스트로 구별하기 어려우면 `판단유보`로 기록한다. 소리로 확인한 실제 인물 적중률을 따로 연구할 때는 그 기준으로 별도 표를 만든다. 기준표는 [HARNESS.md](HARNESS.md)에 있다. 미평가 칸은 빈칸으로 둔다.
5. 아래 명령을 실행한다. `--limit`은 시작 시각 기준 앞에서부터 읽을 발화 수다. 전체를 읽으려면 실제 행 수 이상을 지정한다. 다른 클립이나 조건은 다른 `--out`에 저장한다.

```powershell
.\.venv\Scripts\python.exe lab.py run --input team.json --profiles team_labels/speakers.json --gold team_labels/gold.csv --limit 12 --out outputs/team01
```

### 5. 결과를 해석하는 순서

| 파일 | 열어 보고 답할 질문 |
|---|---|
| predictions.csv | 한 행은 한 모델이 한 발화에 붙인 결과다. 같은 ID 두 행의 화자·감정·서사 중 어디가 달라졌나? GPT 인용은 대상 원문의 연속 부분인지 검사했다. |
| speaker_emotions.csv | 예측한 화자마다 감정 건수·분모·비율이 어떻게 달라지나? 유보도 집계한다. 예측 화자가 바뀌면 감정의 귀속도 달라진다. |
| story_segments.csv | 시간순으로 같은 서사 라벨이 이어진 구간의 첫·끝 ID는 무엇인가? 발화 사이에 정적이 있을 수 있다. 영상으로 경계를 다시 확인한다. |
| metrics.csv | `human_accuracy`는 사람이 채운 칸 중 동일한 비율, `model_agreement`는 모델 간 동일한 비율이다. `n`과 유보 건수도 읽는다. 사람과 모델이 모두 유보면 일치다. |
| results.json | 같은 정보를 MCP가 읽는다. 실제 BERT 실행에는 후보별 상대 점수도 있다. |
| checkpoint.json | 요청 직후 저장한 중간 결과다. 같은 입력·기준·설정·출력 경로로 다시 실행하면 완료된 모델 응답을 재사용한다. |

팀 기록 예시: “U0006(22~28초), BERT ‘승’·GPT ‘전’(모의 응답). ‘그 소리를 무대에 보낼게요’를 사건 방향이 바뀌는 행동으로 보고 우리 조는 ‘전’으로 판정했다.” 실제 팀 기록에는 팀 전사와 실제 결과를 넣는다.

### 6. MCP로 결과 탐색하기

아래 명령은 MCP 클라이언트와 서버를 같은 PC에서 실행한다. 설치 전이라면 첫 줄부터 실행한다. 별도의 채팅 서비스나 모델 키가 필요 없다.

```powershell
uv pip install --python .venv -r requirements.txt
.\.venv\Scripts\python.exe mcp_demo.py outputs/demo/results.json --id U0006 --speaker SPEAKER_00
```

`find_disagreements`는 선택한 과제에서 모델 답이 다른 ID를, `lookup_utterance`는 그 ID의 시각·텍스트·라벨을, `speaker_emotions`는 특정 모델 기준의 화자별 감정 건수를 반환한다. 예제 재생 결과에서는 첫 조회가 U0006을 반환한다. 입력 경로를 `outputs/team01/results.json`으로 바꾸면 팀 자료를 조회한다. MCP는 파일 조회 도구를 연결하고 추론 비용은 선택한 모델 제공처가 결정한다. [MCP 구조](https://modelcontextprotocol.io/docs/learn/architecture)

### 다음 주에 가져갈 파일

14주차에 콘텐츠 하이라이트를 만들려면 `results.json`, `story_segments.csv`, 그리고 팀이 근거를 확인한 발화 ID와 시각을 가져간다. 원본의 `utterance_id`는 그대로 유지한다.
