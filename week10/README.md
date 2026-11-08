# 10주차 실습: 우리 조 데이터의 토픽 읽기
CNT3054 문화콘텐츠와 자연어처리 · 장수빈

지금까지 조별로 수집한 한국어 댓글·리뷰 CSV를 넣어 토픽모델링을 실행한다. 관심 있는 토픽의 원문을 읽고, 상위 단어에 붙인 이름이 실제 글의 내용과 맞는지 조원끼리 이야기한다. 별도 제출물은 없다.

## 저장소와 실행 환경
GitHub Desktop에서 CNT3054 / main을 선택하고 Fetch origin, 나타나면 Pull origin을 누른다. Open in Visual Studio Code로 연다. 아래 명령은 모두 CNT3054 최상위 폴더의 Terminal > New Terminal에서 실행한다.

```sh
uv pip install --python .venv -r week10/requirements.txt
```
기존 `.venv`를 사용한다. 처음 만드는 경우에만 `uv venv --python 3.12 .venv`를 먼저 실행한다. 이후 명령의 `--no-sync` 경고는 pyproject.toml이 없는 폴더에서 나올 수 있다.

## 1. 조별 CSV 넣기
분석하려는 댓글·리뷰 CSV를 **복사**해 `week10/data/my_comments.csv`로 둔다. 원본은 보관한다. 한 행에는 댓글 또는 리뷰 한 개가 있어야 한다. Excel에서는 CSV UTF-8로 저장한다. 작성자 이름·아이디가 들어 있는 열을 본문 열로 선택하지 않는다.

```sh
uv run --no-sync python week10/team_lab.py inspect
```
출력에 나온 실제 열 이름을 확인한다. 이 명령은 행 수와 열 이름만 출력한다.

## 2. 열 이름 연결하기
`week10/team_config.json`을 연다.
- `text_column`: 실제 본문 열 이름. 예를 들어 CSV에 `comment_text`라는 열이 있다면 `"text_column": "comment_text"`로 수정한다.
- `date_column`: 날짜를 사용하지 않으면 `null`을 유지한다. 수집 시각이 아닌 실제 댓글·리뷰 작성일 열이 있으면 그 열 이름을 문자열로 넣는다. 날짜 표기를 통일한다.
- `topics`: 처음에는 4를 유지한다. 4개가 정답이라는 뜻은 없다.
- `encoding`: UTF-8 CSV는 `utf-8-sig`. CP949 파일은 먼저 UTF-8로 저장하거나 값을 `cp949`로 바꾼다.
- 나머지 경로는 그대로 두고 저장한다.

## 3. 분석 실행
```sh
uv run --no-sync python week10/team_lab.py run
```
한 번 실행하면 토픽별 단어와 원문 파일을 함께 만든다. API 키와 GPU는 필요 없다. 모델은 명사·영문 토큰의 빈도로 학습하며, 한 번만 나온 단어와 거의 모든 문서에 나온 단어를 제외한다. 빈 본문과 전처리 뒤 어휘가 남지 않은 문서는 제외 수를 출력한다. 최소 8개 문서는 코드 실행을 위한 하한이며, 연구에 충분한 표본 크기를 뜻하지 않는다.

## 4. 단어와 원문 함께 읽기
`week10/team_outputs/topics.md`를 열고 Ctrl+Shift+V로 미리보기를 켠다. Mac은 Cmd+Shift+V다.

관심 있는 토픽 하나를 고른다. T1을 골랐다면 같은 폴더의 `T1_read.md`를 연다. 해당 토픽의 비중이 높은 글 최대 5개와 두 토픽이 섞인 후보 최대 2개를 읽는다. 원문 파일의 CSV 행 번호로 원본 위치를 다시 찾을 수 있다.

- 우선 단어만 보고 임시 이름을 말해 본다.
- 원문에서 글쓴이가 실제로 무엇을 이야기하는지 확인한다.
- 이름이 맞지 않으면 조원끼리 더 맞는 이름을 붙여 본다. 찬성과 반대가 같은 토픽에 들어갈 수도 있다.
- 반복 문구만 보이면 `tokens_preview.csv`와 원자료를 살펴본다. 필요하면 `data/stopwords.txt`를 수정하고 다시 실행한다.

이름을 입력하는 별도 양식이나 JSON 기록은 필요 없다. 화면을 함께 보면서 진행한다.

## 5. 작성일이 여러 달에 걸친 조
`date_column`에 작성일 열을 지정하고 run을 실행한 경우에만 이어서 실행한다.
```sh
uv run --no-sync python week10/team_lab.py time
```
`team_outputs/time_comparison.png`를 연다. 각 월의 모든 문서에서 해당 토픽의 비중을 평균한 그래프다. 월 아래에는 분석에 남은 문서 수가 표시된다. `time_summary.csv`에는 분모(`n_documents`), 가장 큰 토픽으로 센 문서 수(`dominant_count`), 평균 비중(`mean_proportion`)이 있다.

날짜가 없거나 한 달에만 모인 자료는 원문을 읽는 단계까지 진행한다. 날짜를 임의로 채우지 않는다. 데이터 수집량과 수집 대상이 달라졌는지도 함께 본다. 증가한 토픽의 글을 `document_topics.csv`에서 월별로 찾아 실제 내용이 어떻게 달라졌는지 읽는다.

## 실행이 막힐 때
- 파일 없음: `week10/data/my_comments.csv` 위치와 확장자 확인.
- 본문 열 없음: inspect 출력과 text_column을 똑같이 맞춤.
- 어휘 부족: 본문 열과 tokens_preview 확인. 너무 적은 자료는 더 모음.
- 날짜 오류: 실제 작성일 형식 확인. 원문 해석부터 할 때는 date_column을 null로 바꾸고 run 재실행.
- 다른 자료로 재실행: 기본 team_outputs는 새 결과로 갱신된다. 기존 결과를 남기려면 config의 out을 `week10/team_outputs/run2`처럼 바꾼다.

## 파일
- `team_lab.py`: 이번 수업 실행 코드
- `team_config.json`: 자기 CSV의 본문·날짜 열 지정
- `team_outputs/`: 실행 뒤 생성되는 결과
- `run_lab.py`, `examples/`, 기존 합성 CSV: 이전 버전의 참고 파일. 이번 실습에서는 사용하지 않는다.

학생 원자료와 생성 결과는 본인 컴퓨터에 둔다. 기본 원자료 경로와 결과 폴더는 Git 추적에서 제외했다.

## 참고문헌
Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3, 993–1022. https://www.jmlr.org/papers/v3/blei03a.html

Chang, J., Gerrish, S., Wang, C., Boyd-Graber, J., & Blei, D. M. (2009). Reading tea leaves: How humans interpret topic models. Advances in Neural Information Processing Systems, 22. https://papers.neurips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html
