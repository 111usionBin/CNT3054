# 12주차 · 유튜브 영상의 음성 전사와 화자 분리

문화콘텐츠와 자연어처리 · CNT3054 · 장수빈

유튜브 링크를 입력하면 음성을 내려받아 Deepgram STT로 전사하고, **누가 언제 무엇을 말했는지** 저장합니다. 자막 다운로드가 아니라 실제 음성 인식입니다. 완료된 영상/방송을 처리하며 실시간 방송 수집은 포함하지 않습니다. 화자 번호는 영상 안에서 구분하는 번호로, 실명을 자동 식별하지 않습니다.

## 준비

GitHub Desktop에서 CNT3054를 열고 Fetch origin → Pull origin으로 갱신한 뒤 Open in Visual Studio Code를 선택합니다. VS Code 터미널에서 다음을 실행합니다.

```bash
cd week12
uv venv --python 3.11
uv pip install -r requirements.txt
```

음성 변환용 FFmpeg와 유튜브 다운로드용 Deno도 설치합니다.

Windows PowerShell:

```powershell
winget install --id Gyan.FFmpeg -e
winget install --id DenoLand.Deno -e
```

macOS:

```bash
brew install ffmpeg deno
```

설치 후 VS Code를 완전히 닫았다가 다시 열고 `ffmpeg -version`, `ffprobe -version`, `deno --version`이 실행되는지 확인합니다.

## API 키 설정

[Deepgram Console](https://console.deepgram.com/)에서 API 키를 생성합니다.

```bash
uv run --active python transcribe_youtube.py --setup
```

위 명령은 `week12/.env`를 생성합니다. 파일을 열어 `DEEPGRAM_API_KEY=` 오른쪽에 본인의 키를 입력하고 저장합니다. `.env.txt`로 저장하지 마세요. 환경변수가 이미 설정되어 있다면 환경변수 값이 우선합니다.

`.env`와 결과 폴더는 Git에서 제외됩니다. GitHub에 올라가는 `.env.example`에는 실제 키가 없습니다. 키를 소스 코드·터미널 명령·보고서에 붙여 넣지 않습니다.

## 실행

```bash
uv run --active python transcribe_youtube.py "https://www.youtube.com/watch?v=영상ID"
```

또는 아래 명령을 실행한 후 나타나는 입력창에 링크 하나를 붙여 넣습니다.

```bash
uv run --active python transcribe_youtube.py
```

`youtube.com/watch`, `youtu.be`, Shorts, 종료된 라이브 링크를 지원합니다. 재생목록이 붙어 있어도 지정된 영상 하나만 처리합니다. 기본 한도는 60분이며 더 긴 영상은 `--max-minutes 120`처럼 지정합니다(최대 240분). 링크의 `t=`는 무시하고 영상 전체를 처리합니다. 분석 권한이 있는 영상으로 실행하세요. 영상에서 추출한 음성은 Deepgram에 전송되며 계정 사용량이 발생합니다.

## 저장 결과

`week12/outputs/실행시각/`에 저장합니다. 재실행할 때 이전 결과를 덮어쓰지 않습니다.

| 파일 | 내용 |
|---|---|
| `utterances.csv` | 엑셀에서 열 수 있는 화자·시간·발화 표(UTF-8 BOM) |
| `utterances.json` | 13~14주차 분석에 사용할 동일 데이터 |
| `transcript.txt` | 화자와 시간이 표시된 전사문 |
| `subtitles.srt` | 화자 표시 자막 |
| `deepgram_raw.json` | 단어별 시간·신뢰도 등을 포함한 원본 API 응답 |
| `metadata.json` | 출처 URL, 영상 ID, 제목, 실행 설정 |
| `speaker_names.json` | 확인한 화자 이름을 입력할 매핑 파일 |

CSV 열: `utterance_id`, `speaker_id`, `speaker_name`, `start_seconds`, `end_seconds`, `start_time`, `end_time`, `text`. 시간은 영상 시작 기준입니다. CSV 수식 실행 방지를 위해 `=`, `+`, `-`, `@`로 시작하는 문자열에는 작은따옴표를 붙입니다. JSON/TXT에는 원문을 유지합니다.

화자 변화·긴 쉼·긴 발화를 기준으로 행을 나누므로 행 하나가 반드시 문법적인 문장 하나는 아닙니다. 화자 미확인은 `UNKNOWN`으로 남깁니다. 겹쳐 말하기·짧은 맞장구·배경음이 있는 구간은 영상을 보며 확인하세요. 서로 다른 영상의 `SPEAKER_00`은 같은 사람을 뜻하지 않습니다.

음성 파일은 성공 후 삭제합니다. 보관하려면 `--keep-audio`를 붙입니다. 실패 시에는 해당 실행 폴더에 중간 파일이 남을 수 있습니다. 영상 파일 자체는 저장하지 않습니다.

## 이름을 확인한 뒤 다시 저장

생성된 `speaker_names.json`의 값을 `SPEAKER_00`에서 `진행자` 또는 확인한 이름으로 변경합니다. 아래 경로에서 `실행시각`을 실제 폴더명으로 바꿉니다.

```bash
uv run --active python transcribe_youtube.py --response outputs/실행시각/deepgram_raw.json --speaker-names outputs/실행시각/speaker_names.json
```

새 폴더에 결과를 만들며 API를 재호출하지 않습니다. 키 없이도 재변환할 수 있습니다.

## 다운로드가 안 될 때

```bash
uv pip install --upgrade "yt-dlp[default]"
```

Deno·FFmpeg 설치를 확인합니다. YouTube의 봇 확인·로그인·지역 제한 때문에 일부 링크는 실패할 수 있습니다. 로그인 쿠키를 자동 추출하거나 제한을 우회하지 않습니다. 확보한 로컬 파일은 다음처럼 입력할 수 있습니다.

```bash
uv run --active python transcribe_youtube.py --file "C:/Users/me/Videos/interview.mp4"
```

401은 키, 402는 크레딧, 403은 권한, 429는 호출 한도를 확인합니다. 응답 시간 초과 시 이미 처리되었을 수 있으므로 계정 사용 내역 확인 후 재실행합니다. 과금 중복을 피하기 위해 전사 요청은 자동 재시도하지 않습니다.

## 검증

```bash
uv run --active python -m unittest discover -s tests -v
```

테스트는 합성 응답과 모의 API를 사용하며 실제 음성을 전송하지 않습니다.

공식 문서: [Deepgram 전사 API](https://developers.deepgram.com/reference/speech-to-text/listen-pre-recorded), [화자 분리](https://developers.deepgram.com/docs/diarization), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [YouTube JS 런타임](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
