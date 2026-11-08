# 10주차 실습: 토픽 해석과 변화 분석
CNT3054 문화콘텐츠와 자연어처리 · 장수빈

공연 예매 댓글 120개에서 토픽을 발견하고, 원문을 읽어 이름을 다듬고, 시기별 변화를 설명한다. 마지막에는 근거 인용 검사 규칙을 실행한다. API 키와 GPU는 필요 없다. 결과는 CSV, Markdown, PNG로 저장하며 VS Code에서 연다.

## 1. GitHub Desktop에서 시작
- 이미 `CNT3054`가 있다면 해당 저장소와 `main` 브랜치를 선택한다. `Fetch origin`을 누르고 `Pull origin`이 나타나면 실행한다. 로컬 수정 사항이 있다면 먼저 보관한다.
- 처음 받는다면 `File > Clone repository > URL`에서 `https://github.com/111usionBin/CNT3054`를 넣고 `Clone`을 누른다.
- `Open in Visual Studio Code`를 누른다. 탐색기에서 `week10`을 확인한다.
- 아래 명령은 모두 `CNT3054` 최상위 폴더의 `Terminal > New Terminal`에서 실행한다. Windows/macOS 공통 명령이다.

## 2. 환경과 데이터 확인
기존 수업 `.venv`를 사용한다. 환경이 없는 경우에만 `uv venv --python 3.12 .venv`를 한 번 실행한다.

```sh
uv pip install --python .venv -r week10/requirements.txt
uv run --no-sync python week10/run_lab.py check
```

`Python:` 경로에 `.venv`가 포함되어야 한다. 예제는 문서 120개, 2026-09 40개, 2026-10 80개다. `--no-sync has no effect` 경고는 `pyproject.toml`이 없는 폴더에서 나올 수 있다. 출력된 실행 경로를 확인한다. 검증 환경의 버전은 `examples/run_metadata.json`에 기록했다. 동일 버전 재현이 필요하면 별도 환경에서 `requirements-reproduce.txt`를 사용한다.

`data/concert_comments.csv`에서 한 행은 문서 한 개다. `document_id`, `date`, `text`를 확인한다. 데이터 생성 방식과 설계는 [data/README.md](data/README.md)에 있다.

## 3. 전체 기간을 한 번에 학습
```sh
uv run --no-sync python week10/run_lab.py fit
```

`outputs/01_topics.md`를 열고 `Ctrl+Shift+V`(Mac: `Cmd+Shift+V`)로 미리보기를 연다. T1~T4의 단어만 보고 임시 이름을 적는다. 기준 실행에서는 T1 무대와 배우, T2 회원 선예매, T3 관람 비용, T4 예매 운영이 두드러진다. 번호 자체는 의미가 없으므로 자신의 결과를 먼저 읽는다.

`outputs/tokens_preview.csv`는 실제 사용한 명사 토큰, `topics.csv`는 상위 단어 확률, `document_topics.csv`는 문서별 토픽 비중이다. 문서별 T1~T4의 합은 약 1이다. 월별로 별도 모델을 만들지 않고 전체 기간에 공통 모델을 학습한다.

## 4. 대표·혼합·무작위 원문 읽기
```sh
uv run --no-sync python week10/run_lab.py evidence
```

`outputs/02_T2_evidence.md`를 미리보기로 연다. 대표 문서 3개, 혼합 후보 최대 2개, 무작위 문서 2개를 읽는다. 혼합 후보는 현재 토픽이 상위 2위 안에 들어가고 1위와 2위의 비중 차이가 작은 문서다. 이는 표본 추출 기준이며 의미상 경계인지는 원문에서 판단한다. 무작위 표본은 토픽이 낮은 문서도 포함한다.

이어 `data/concert_comments.csv`에서 `D023`과 `D025`를 찾아 비교한다. 회비의 보상과 신규 관객의 기회를 어떻게 주장하는지 확인한다. 같은 회원 토픽에 서로 다른 입장이 들어갈 수 있다.

## 5. 해석 기록 작성
`work/interpretations.json`에 각 토픽의 다음 항목을 입력하고 저장한다. `evidence`를 다시 실행해도 이 파일의 기존 내용은 유지한다.

```json
{
  "topic": "T2",
  "label": "회원 선예매의 정당성과 접근 기회",
  "evidence_ids": ["D023", "D029"],
  "counterexample_id": "D025",
  "cultural_question": "기여의 보상과 신규 관객의 기회를 어떻게 조율하는가?",
  "scope_note": "서로 다른 입장을 함께 읽고 자료 범위에서 설명한다."
}
```

T1~T4 네 항목을 채운 뒤 실행한다.
```sh
uv run --no-sync python week10/run_lab.py review
```
`추가 작성`은 빈 항목이나 없는 문서 ID가 있다는 뜻이다. `작성 완료`는 형식 확인 결과다. 근거가 해석을 뒷받침하는지는 조원끼리 원문을 읽어 판단한다. `outputs/03_interpretations.md`에서 정리한 내용을 확인한다. 참고용 완성 예시는 `examples/interpretations_completed.json`에 있다.

## 6. 시기별 양과 비중 비교
```sh
uv run --no-sync python week10/run_lab.py time
```

탐색기에서 `outputs/04_time_comparison.png`를 클릭한다. 왼쪽은 가장 큰 토픽 하나로 문서를 분류한 수, 오른쪽은 문서별 토픽 비중을 평균한 값이다. 각 월의 전체 문서 수가 40개와 80개로 다르다는 점을 먼저 확인한다. `time_summary.csv`와 `04_time_report.md`에서 수치를 읽는다.

기준 실행의 T3는 12개에서 18개로 늘지만 평균 비중은 약 31.8%에서 23.0%로 줄어든다. T2는 약 19.2%에서 38.0%로 커진다. 서로 다른 집계 방식을 구분해 결과 문장을 쓴다. 이어 같은 토픽의 두 시기 원문을 읽어 어떤 주장이 달라졌는지 확인한다. 이 데이터의 월별 구성은 실습용 설계에 따라 달라지며 실제 사건의 효과를 추정하지 않는다.

## 7. 작은 하네스 실행
```sh
uv run --no-sync python week10/run_lab.py harness --topic T2
```

`outputs/05_harness_prompt.md`에서 질문과 제공된 문서 범위를 확인한다. `05_harness_input.json`은 문서 ID, 원문, 출력 형식을 담는다. 이 명령은 외부 LLM을 호출하지 않는다. 수동 작성된 정상·오류 응답을 만들고 같은 검사 규칙으로 검사한다.

`outputs/05_audit.md`에서 결과를 읽는다.
- `valid: HUMAN_REVIEW`: 인용 형식과 원문 일치를 확인했다. 주장과 근거의 관계는 사람이 검토한다.
- `invalid: REJECT`: 원문에 없는 문장을 인용했다. 해당 문서와 인용을 고친 뒤 다시 검사한다.

직접 만든 응답이나 후속 수업의 LLM 응답은 같은 스키마로 저장한 뒤 검사할 수 있다.
```sh
uv run --no-sync python week10/run_lab.py audit --response week10/outputs/response_valid.json
```
`claims` 배열 안에 `statement`, `citations`가 필요하며 각 인용에 `document_id`, `quote`를 넣는다. 인용은 입력 묶음 안에 있는 원문과 정확히 일치해야 한다. `REJECT`는 종료 코드 1, 실행 오류는 2, 검사 완료는 0이다.

## 8. 조별 데이터 적용
`data/my_comments_template.csv`의 열을 유지하고 실제 자료 8개 이상을 `data/my_comments.csv`로 저장한다. 8개는 실행 조건이며 충분한 표본 수의 기준은 아니다. 두 개 이상 월의 날짜와 중복 없는 문서 ID가 필요하다. 날짜·ID가 누락되면 `check`에서 멈춘다. 빈 전처리 결과는 `excluded_documents.csv`에 기록한다. 중복 원문은 자동 삭제하지 않으므로 중복 처리 기준을 정한다.

```sh
uv run --no-sync python week10/run_lab.py check --input week10/data/my_comments.csv
uv run --no-sync python week10/run_lab.py fit --input week10/data/my_comments.csv --out week10/team_outputs
uv run --no-sync python week10/run_lab.py evidence --out week10/team_outputs
uv run --no-sync python week10/run_lab.py time --out week10/team_outputs
```

`fit`은 선택한 출력 폴더의 분석 결과를 다시 쓴다. 다른 자료를 비교할 때는 `--out`을 달리 정한다. `work/interpretations.json`은 공통 작업 파일이므로 새로운 자료를 해석하기 전에 기존 기록을 다른 이름으로 보관하고 내용과 문서 ID를 새 자료에 맞춘다. 원문·개인정보가 담긴 조별 파일은 공개 저장소에 올리지 않는다.

## 9. 실습 정리
토픽 하나를 골라 임시 이름과 수정한 이름, 근거 문서 2개, 다른 해석을 확인한 문서 1개를 기록한다. 시기별 변화는 전체 문서 수와 집계 방식을 함께 적는다. 문화적 의미 한 문장과 그 의미를 더 확인하기 위해 필요한 자료를 정리한다. 제출 형식과 기한은 수업 공지를 따른다.

## 10. 문제 해결
|보이는 증상|확인과 다음 행동|
|---|---|
|week10 파일을 찾지 못함|GitHub Desktop에서 Fetch/Pull 후 VS Code 탐색기 확인|
|ModuleNotFoundError|check의 Python 경로 확인 후 같은 .venv에 requirements 설치|
|터미널 경로가 week10임|`cd ..`로 CNT3054 최상위 폴더로 이동|
|Markdown 소스만 보임|파일을 연 뒤 Ctrl+Shift+V, Mac은 Cmd+Shift+V|
|토픽 번호나 숫자가 다름|run_metadata.json의 패키지와 입력 해시 비교, 자기 결과로 해석|
|JSON 오류|따옴표·쉼표·대괄호 확인, 저장 후 review 재실행|
|한글 그래프가 네모로 표시됨|맑은 고딕 / AppleGothic / Noto Sans CJK KR 중 하나 설치 후 time 재실행|

## 검증
```sh
uv run --no-sync python -m unittest discover -s week10/tests -v
```

## 참고
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. JMLR, 3, 993–1022. https://www.jmlr.org/papers/v3/blei03a.html
- Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C., & Blei, D. M. (2009). Reading tea leaves: How humans interpret topic models. NeurIPS, 22.
- Grimmer, J., & Stewart, B. M. (2013). Text as data. Political Analysis, 21(3), 267–297. https://doi.org/10.1093/pan/mps028
- DiMaggio, P., Nag, M., & Blei, D. (2013). Exploiting affinities between topic modeling and the sociological perspective on culture. Poetics, 41(6), 570–606. https://doi.org/10.1016/j.poetic.2013.08.004
- Pham, C. M., Hoyle, A., Sun, S., Resnik, P., & Iyyer, M. (2024). TopicGPT. NAACL-HLT, 2956–2984. https://aclanthology.org/2024.naacl-long.164/
- scikit-learn LDA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
