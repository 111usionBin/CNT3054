# 13주차 · 방송에서 누가 어떤 감정으로 말했을까?

CNT3054 문화콘텐츠와 자연어처리 · 장수빈

12주차에는 음성을 듣고 화자가 바뀐 시각을 찾았다. 이번에는 같은 방송의 **전사 텍스트**를 한국어 RoBERTa와 GPT-OSS에 주고, 각 발화의 화자·감정·기승전결을 사람이 붙인 라벨과 비교한다. 아래 명령은 **Windows PowerShell** 기준이다. 이미 전사를 마쳤다면 1단계에서 Groq 키를 확인하고, 2~3단계를 거쳐 **5단계**로 이동한다.

## 1. API 키 발급받기

| 용도 | 발급 방법 | 저장할 파일 |
|---|---|---|
| 영상 음성 전사와 화자 분리 | [Deepgram Console](https://console.deepgram.com/)에 로그인한다. 프로젝트의 **API Keys**에서 키를 만들고 표시된 값을 복사한다. [발급 안내](https://developers.deepgram.com/docs/create-additional-api-keys) | `week12/.env`의 `DEEPGRAM_API_KEY` |
| GPT로 발화 분류 | [Groq Console의 API Keys](https://console.groq.com/keys)에서 **Create API Key**를 누른다. 키 이름을 정하고 생성된 값을 복사한다. | `week13/.env`의 `GROQ_API_KEY` |

기존 키가 해당 파일에 저장되어 있으면 그대로 사용한다. 한국어 BERT 모델은 내 PC에서 실행하므로 키가 필요 없다. Deepgram에는 영상 음성이, Groq에는 분석할 발화와 앞뒤 문맥·역할 소개·평가 기준이 전송된다.

## 2. 사용할 파이썬 선택하기

GitHub Desktop에서 **Fetch origin → Pull origin → Open in Visual Studio Code** 순서로 연다. VS Code에서 **터미널 → 새 터미널**을 선택한다. 아래 두 줄은 현재 위치가 `CNT3054`, `week12`, `week13` 중 어디든 저장소의 `week13`으로 이동한다.

~~~powershell
$repo = git rev-parse --show-toplevel
Set-Location (Join-Path $repo 'week13')
~~~

**A. 13주차용 `.venv`를 새로 만들 학생**은 아래 한 줄을 실행한다. 이미 이 폴더에 환경을 만들었다면 기존 환경을 선택한다.

~~~powershell
if (!(Test-Path .\.venv\Scripts\python.exe)) { uv venv --python 3.11 .venv }; $py = (Resolve-Path .\.venv\Scripts\python.exe).Path
~~~

**B. 지난주 `week12` 환경을 계속 쓸 학생**은 아래 한 줄을 실행한다. 프롬프트가 이미 `(week12)`인 경우에도 같은 명령을 쓸 수 있다.

~~~powershell
$py = (Resolve-Path ..\week12\.venv\Scripts\python.exe).Path
~~~

이제부터 `& $py`는 방금 선택한 파이썬을 실행한다. 아래 첫 줄로 경로를 확인하고, 두 번째 줄로 그 환경에 실습 패키지를 설치한다. 이후 명령은 같은 터미널에서 이어서 실행한다.

~~~powershell
& $py -c "import sys; print(sys.executable)"
uv pip install --python "$py" -r requirements-bert.txt
~~~

## 3. 키 파일과 전사 저장 경로 준비하기

~~~powershell
$w12 = Join-Path $repo 'week12'
$transcripts = Join-Path $w12 'outputs'
New-Item -ItemType Directory -Force -Path $transcripts | Out-Null
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
~~~

메모장에서 `GROQ_API_KEY=` 뒤에 Groq 키를 넣고 저장한다. 나머지 두 줄은 예제 파일에 들어 있는 설정을 사용한다.

~~~dotenv
GROQ_API_KEY=여기에_Groq_키
GPT_MODEL=openai/gpt-oss-20b
BERT_API_URL=http://127.0.0.1:8013/predict
~~~

`$transcripts`는 지난주 전사가 쌓이는 `week12/outputs` 폴더다. 이미 전사한 영상은 **5단계**에서 이 폴더를 읽으면 된다.

## 4. 새 영상을 사용할 때만 전사하기

같은 터미널에서 전사 패키지를 설치하고 Deepgram 키 파일을 연다.

~~~powershell
uv pip install --python "$py" -r ..\week12\requirements.txt
& $py ..\week12\transcribe_youtube.py --setup
notepad ..\week12\.env
~~~

`DEEPGRAM_API_KEY=` 뒤에 Deepgram 키를 넣고, 한국어 방송이면 `DEEPGRAM_MODEL=nova-3`, `DEEPGRAM_LANGUAGE=ko`를 확인한 뒤 저장한다. `--setup`은 키 파일을 준비하고 경로를 출력한다. 이미 키가 들어 있어도 같은 안내가 뜬다.

터미널에서 `ffmpeg -version`, `ffprobe -version`, `deno --version`이 실행되는지 확인한다. 처음 설치하는 PC의 설치 명령은 [12주차 준비 안내](../week12/README.md#준비)에 있다. 설치 후 터미널을 새로 열었다면 2~3단계에서 경로 변수를 다시 설정한다.

~~~powershell
& $py ..\week12\transcribe_youtube.py --output-dir "$transcripts"
~~~

입력창에 영상 링크를 붙여 넣는다. 전사가 끝나면 프로그램이 `outputs` 안에 실행시각 폴더를 만들고 `utterances.json`과 `deepgram_raw.json` 등을 저장한다. **13주차 입력은 발화별 텍스트와 화자 번호가 들어 있는 `utterances.json`이다.**

## 5. 전사 파일을 번호로 선택하기

아래 블록을 한 번에 붙여 넣는다. 최신 전사가 1번이다. Enter를 누르면 1번을 선택하고, 다른 영상을 쓸 때는 목록의 번호만 입력한다.

~~~powershell
$files = @(Get-ChildItem $transcripts -Filter utterances.json -File -Recurse | Sort-Object FullName -Descending)
if ($files.Count -eq 0) { throw "week12/outputs에 utterances.json이 없습니다. 4단계 전사 결과를 확인하세요." }
for ($i = 0; $i -lt $files.Count; $i++) { Write-Host "[$($i+1)] $($files[$i].Directory.Name)" }
$choice = Read-Host '사용할 전사 번호 (Enter = 1)'
if (!$choice) { $choice = '1' }
if ($choice -notmatch '^[1-9][0-9]*$' -or [int]$choice -gt $files.Count) { throw '목록에 있는 번호를 선택하세요.' }
$file = $files[([int]$choice - 1)].FullName
$file
~~~

마지막 줄에 선택한 파일의 전체 경로가 보인다. 이제부터 `$file`이 그 파일을 가리킨다. 아래 명령으로 첫 12개 발화의 ID·화자·원문을 확인한다.

~~~powershell
Get-Content -Raw -Encoding UTF8 $file | ConvertFrom-Json | ForEach-Object { $_ } | Select-Object -First 12 utterance_id,speaker_id,text | Format-Table -Wrap
~~~

## 6. 팀 평가 폴더 만들고 역할과 정답 입력하기

**새로 분석하는 영상**은 아래 두 줄로 영상별 저장 위치를 정한다. `$labels`에는 사람이 작성한 평가를, `$out`에는 모델의 분석 결과를 저장한다.

~~~powershell
$runId = (Get-Item $file).Directory.Name
$labels = "outputs/$runId/labels"; $out = "outputs/$runId/first12"
~~~

**이전에 같은 영상으로 `team_labels`와 `outputs/team01`을 사용한 팀**은 대신 아래 한 줄을 쓴다. 다른 영상이면 새 위치를 사용한다.

~~~powershell
$labels = 'team_labels'; $out = 'outputs/team01'
~~~

아래 블록은 평가 폴더가 처음이면 `prepare`로 두 파일을 만들고, 둘 다 있으면 작성한 파일을 그대로 연다.

~~~powershell
$profiles = Join-Path $labels 'speakers.json'
$gold = Join-Path $labels 'gold.csv'
if (!(Test-Path $profiles) -and !(Test-Path $gold)) {
    & $py lab.py prepare --input "$file" --out "$labels"
}
if (!(Test-Path $profiles) -or !(Test-Path $gold)) { throw '평가 파일 두 개가 모두 필요합니다. 선택한 평가 폴더를 확인하세요.' }
code "$profiles" "$gold"
~~~

- **speakers.json**에서 각 화자 ID 옆에 역할 소개를 1~120자로 적는다. 예를 들어 진행자라면 “상품을 소개하고 출연자에게 사용법을 묻는다.”라고 적는다. 영상의 다른 구간을 참고하고, 평가할 발화의 문장을 그대로 넣지 않는다.
- **gold.csv**에서 첫 12발화를 보고 각 행의 `speaker,emotion,stage`를 채운다. 두 사람이 먼저 독립적으로 판단하고 차이를 조정한다. 텍스트만으로 판단하기 어려워 내린 결정은 `판단유보`, 아직 평가하지 않은 칸은 빈칸으로 둔다. 두 파일 모두 **Ctrl+S**로 저장한다. [라벨별 판정 기준](HARNESS.md)

## 7. 한국어 BERT 서버 켜기

같은 터미널에 아래 한 줄을 입력하면 **방금 선택한 가상환경**으로 서버 창이 따로 열린다. 이미 8013번 포트에서 실습 서버가 실행 중이면 기존 창을 사용한다.

~~~powershell
Start-Process -FilePath $py -WorkingDirectory (Get-Location).Path -ArgumentList '-m uvicorn bert_api:app --host 127.0.0.1 --port 8013'
~~~

첫 실행에서는 한국어 모델을 내려받는다. 새 창에 `Application startup complete`가 보이면 창을 열어 두고 **원래 PowerShell**에서 다음 단계를 진행한다.

## 8. BERT와 GPT로 첫 12발화 분석하기

~~~powershell
& $py lab.py run --input "$file" --profiles "$profiles" --gold "$gold" --limit 12 --out "$out"
~~~

`1/12`부터 발화별 진행 상황이 표시되고, 끝나면 결과 폴더 경로가 뜬다. 같은 입력·역할·모델 설정이면 이미 저장한 응답은 재사용한다. 사람 평가표만 고치면 같은 명령으로 일치율을 다시 계산할 수 있다. 역할 소개나 발화 수를 바꿔 비교할 때는 `$out`에 새 결과 폴더명을 지정한다.

~~~powershell
Invoke-Item (Resolve-Path $out).Path
~~~

## 9. MCP로 결과 조회하기

~~~powershell
$result = Join-Path $out 'results.json'
& $py mcp_demo.py "$result" --id U0006 --speaker SPEAKER_00
~~~

`find_disagreements`는 **기승전결 판정**이 갈린 발화 ID를 보여준다. `lookup_utterance`는 지정한 U0006의 원문·시각과 두 모델의 답을 나란히 보여준다. `speaker_emotions`는 GPT가 SPEAKER_00으로 분류한 발화의 감정 건수를 센다. 다른 발화나 화자를 보려면 마지막 줄의 ID만 바꾼다. 조회가 끝나면 BERT 서버 창에서 **Ctrl+C**로 종료한다.

## 결과 파일 읽기


| 파일 | 열어 보고 답할 질문 |
|---|---|
| predictions.csv | 한 행은 한 모델이 한 발화에 붙인 결과다. 같은 ID 두 행의 화자·감정·서사 중 어디가 달라졌나? GPT 인용은 대상 원문의 연속 부분인지 검사했다. |
| speaker_emotions.csv | 예측한 화자마다 감정 건수·분모·비율이 어떻게 달라지나? 유보도 집계한다. 예측 화자가 바뀌면 감정의 귀속도 달라진다. |
| story_segments.csv | 시간순으로 같은 서사 라벨이 이어진 구간의 첫·끝 ID는 무엇인가? 발화 사이에 정적이 있을 수 있다. 영상으로 경계를 다시 확인한다. |
| metrics.csv | `human_accuracy`는 사람이 채운 칸 중 동일한 비율, `model_agreement`는 모델 간 동일한 비율이다. `n`과 유보 건수도 읽는다. 사람과 모델이 모두 유보면 일치다. |
| results.json | 같은 정보를 MCP가 읽는다. 실제 BERT 실행에는 후보별 상대 점수도 있다. |
| checkpoint.json | 요청 직후 저장한 중간 결과다. 같은 입력·기준·설정·출력 경로로 다시 실행하면 완료된 모델 응답을 재사용한다. |


14주차에는 `results.json`, `story_segments.csv`, 그리고 팀이 영상에서 확인한 발화 ID와 시각을 사용한다.
