# Week 7 · 문맥 기반 감정분석과 텍스트 임베딩

**문맥을 이해하는 감정분석은 기존 방식과 어떻게 다른가?**

[▶ Colab에서 실습 열기](https://colab.research.google.com/github/111usionBin/CNT3054/blob/week07-nvidia-lab/week07/week07_context_sentiment.ipynb)

[실습 노트북](week07_context_sentiment.ipynb) · [NVIDIA 모델 페이지](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b)

## 수업 목표와 진행

1. 사전과 TF-IDF 기준선을 같은 리뷰에 적용합니다.
2. NVIDIA API의 LLM에 **정답 예시 없이 기준만 제시**하여 zero-shot 감정분류를 실행합니다.
3. Word2Vec과 문장 임베딩을 직접 계산하고, 감정이 다르지만 의미가 비슷한 문장을 살펴봅니다.
4. NSMC로 미세조정된 BERT의 추론 결과를 비교하고 오분류 원인을 설명합니다.

핵심 실습 45분, 임베딩·BERT 확장 20분을 예상합니다. 계정 준비와 다운로드 시간은 별도입니다.
모든 제공 문장은 **합성 교육용 자료**이며 실제 리뷰·학생 자료·개인정보가 아닙니다.

## 시작: 클릭 순서

1. 위 **Colab에서 실습 열기**를 클릭합니다.
2. Google 계정으로 로그인한 뒤 **파일 > Drive에 사본 저장**을 선택합니다.
3. 노트북의 각 셀 왼쪽 **▶**를 위에서 아래로 누릅니다. `Shift+Enter`도 가능합니다.
4. NVIDIA 모델 페이지에서 현재 **Free Endpoint: Available**인지 확인합니다.
5. **Review Terms**를 읽고 본인이 동의하면 진행합니다. **Generate API Key** 또는 카탈로그의 **Get API Key**에서 로그인/회원가입과 필요한 인증을 완료하고 키를 복사합니다.
6. 노트북 5번 셀의 `RUN_NVIDIA = False`를 `True`로 바꿉니다. 실행 후 숨김 입력란에 키를 넣고 Enter를 누릅니다.
7. 7번 셀에서 1건을 점검하고, 8번 셀에서 16건을 분류합니다.
8. 9~12번은 임베딩·BERT 확장입니다. 마지막 비교와 저장은 13~14번입니다.

**API 호출에 로컬 NVIDIA 그래픽카드나 유료 Colab이 필요하지 않습니다.** BERT/문장 임베딩은 공개 가중치를 Colab CPU로 내려받아 추론합니다. 최초 다운로드에는 시간이 걸립니다.
키 발급 과정의 계정 인증과 약관 동의는 각 학생이 본인 계정에서 직접 수행합니다. 키를 코드, GitHub, 스크린샷, 제출물에 넣지 마세요. 키를 공유하지 않습니다.

## NVIDIA API와 zero-shot

- 모델: `nvidia/nemotron-3-ultra-550b-a55b`
- 주소: `https://integrate.api.nvidia.com/v1`
- 방식: OpenAI 호환 클라이언트의 `chat.completions.create`
- 입력: 라벨 정의를 담은 system 메시지 + `target`, `review`만 담은 user 메시지
- 출력: `label`, `evidence`를 가진 JSON 객체
- 라벨: 긍정/부정/중립/혼합
- 설정: `temperature=0`, `max_tokens=256`, `enable_thinking=False`, `stream=False`
- 기본 요청 수: 점검 1회 + 16문장 = 17회. 429/서버 오류는 최대 2회 재시도할 수 있습니다.

`openai`는 여기서 통신용 Python 패키지입니다. 요청 대상과 키는 NVIDIA이며 별도 OpenAI API 키를 사용하지 않습니다.
zero-shot은 별도 모델의 이름이 아니라 **현재 과제의 정답 예시를 프롬프트에 주지 않는 적용 방식**입니다. 모델 자체가 사전학습·지시학습을 하지 않았다는 뜻은 아닙니다.
2026-10-15 공식 모델 페이지에서 무료 엔드포인트 제공과 공식 모델 문서의 한국어 지원을 확인했습니다. 무료 개발용 제공은 사용 제한이 있으며 서비스·계정별 조건이 바뀔 수 있습니다. 과거의 1,000/5,000 크레딧을 현재의 고정 지급량으로 가정하지 않습니다.

## 자료와 평가 범위

| 파일 | 의미 |
|---|---|
| `data/train_reviews.csv` | 6주차 합성 리뷰 120개. TF-IDF 학습과 작은 Word2Vec 시연용 |
| `data/context_reviews.csv` | 새 진단 문장 16개. 부정어·대조·반어·평가 대상·중립/혼합 |
| `lab.py` | 분석 함수, API 호출, 출력 검증, 평가, 저장 |
| `week07_context_sentiment.ipynb` | 15단계 학생용 노트북 |
| `examples/` | 실제 로컬 실행한 사전·TF-IDF·BERT 및 임베딩 결과 |
| `tests/test_lab.py` | 라벨 누출 방지, 형식·인용 검증, 평가 분모 점검 |

TF-IDF는 학습용 120개만 fit합니다. 새 진단 문장은 학습에 넣지 않으며 `gold_label`은 API에 보내지 않습니다. 학습 자료는 기존 `week06/data/synthetic_reviews.csv`의 커밋 `f31b5d22bbd3d87a663103ac9abe4d83d79ef955`에서 가져왔습니다.

지표는 긍정 6개·부정 6개의 **공통 12개**로 비교합니다. 중립/혼합 4개는 별도 범위 점검입니다. 사전의 동점(`-1`)은 판정 보류이고 중립의 정답이 아닙니다. LLM의 중립/혼합 응답과 형식 오류는 이진 평가 대상 12개에서 오답으로 계산하며 제거하지 않습니다. `binary_coverage`도 함께 보고합니다. Macro-F1은 부정과 긍정의 F1 평균입니다.

작은 의도적 예제로 방법의 작동 차이를 관찰하는 수업입니다. 실제 플랫폼의 정확도, 모델 전체의 우열, 일반화 성능을 추정하지 않습니다. LLM/BERT/기준선의 학습 규모도 다릅니다. 프롬프트를 수정한 뒤에는 새 독립 평가 자료가 필요합니다.

## 실제 확인한 결과

| 방법 | 공통 n | 이진 판정률 | 전체 정확도 | Macro-F1 |
|---|---:|---:|---:|---:|
| 작은 사전 | 12 | 0.667 | 0.500 | 0.600 |
| TF-IDF + 로지스틱 회귀 | 12 | 1.000 | 0.833 | 0.833 |
| NSMC 미세조정 BERT | 12 | 1.000 | 1.000 | 1.000 |

이 값은 합성 진단 자료의 실행 예시입니다. BERT가 항상 완벽하다는 뜻이 아닙니다. LLM 결과는 학생의 실제 NVIDIA 응답으로 채우며 미리 만든 수치를 제공하지 않습니다.

## 출력 파일과 오류

- `predictions.csv`: 원문, 교육용 정답, 각 방법 예측, 실제 LLM 원문 응답·근거·상태
- `metrics.csv`: 공통 12개 지표
- `error_review.csv`: 하나 이상의 방법이 틀리거나 보류한 이진 대상 문장
- `run_metadata.json`: 모델, 프롬프트, 설정, 버전, 자료 해시. API 키 없음

`outputs/run_.../`마다 새로 저장하며 마지막 셀에서 ZIP을 다운로드합니다.
`invalid_json`, `invalid_label`, `invalid_evidence`는 억지로 정상 결과로 바꾸지 않습니다. 근거가 원문에 있는지 점검해도 분류의 타당성이 자동으로 보장되지는 않습니다.

| 상황 | 조치 |
|---|---|
| 401 | 복사한 키와 유효 상태 확인. 코드에 키를 출력하지 않기 |
| 403 | 계정의 API 권한, 인증, 약관 동의 상태 확인 |
| 404 | 모델의 무료 제공 상태와 모델 ID 확인 |
| 429 | 반복 실행 중지, 잠시 뒤 재시도. 수업 시간에는 조별로 순차 실행 |
| 응답 시간 초과 | 네트워크와 서비스 상태 확인 후 1건 점검부터 실행 |
| API 사용 불가 | 9~12번 BERT·임베딩 실습으로 진행. API 결과는 미실행으로 남김 |
| 다운로드 실패 | 런타임 연결과 인터넷 확인. API 실습은 큰 모델 다운로드와 독립적 |

## 로컬 실행

Python 3.11~3.12 권장. 저장소 최상위에서 가상환경을 만들고 `week07/requirements.txt`를 설치한 뒤 Jupyter 또는 VS Code로 노트북을 엽니다. 노트북 첫 셀은 로컬에서는 현재 작업 폴더 아래로 수업 파일을 내려받습니다.

```sh
python -m venv week07/.venv
# macOS/Linux
week07/.venv/bin/python -m pip install -r week07/requirements.txt ipykernel
# Windows PowerShell
.\week07\.venv\Scripts\python.exe -m pip install -r week07/requirements.txt ipykernel
```

VS Code에서 노트북 오른쪽 위 커널을 해당 가상환경으로 선택합니다. 원문·키·실행 결과는 `.gitignore` 대상 폴더에 보관합니다.

## 검증 범위

Python 3.12에서 사전, TF-IDF, Word2Vec(32차원), 문장 임베딩(384차원), BERT 추론, JSON·원문 인용 검증, 동일 분모 평가를 확인했습니다. BERT revision은 `47fdca16e29c52b9cffc78df8ff7bef7a657c4bb`입니다. API 요청 형식·정답 미전송은 모의 응답으로 검증했습니다. 개인 API 키를 발급하거나 NVIDIA에 실제 인증 요청을 보내지는 않았으므로 실제 API 응답·계정 한도는 7번 셀로 확인해야 합니다.

## 공식 참고자료

- [NVIDIA NIM 무료 개발용 접근 안내](https://developer.nvidia.com/nim)
- [Nemotron Ultra 공식 모델·API 문서](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b)
- [NVIDIA API 시작 안내](https://docs.api.nvidia.com/nim/docs/api-quickstart)
- [NSMC로 미세조정한 BERT 모델 카드](https://huggingface.co/sangrimlee/bert-base-multilingual-cased-nsmc)
- [다국어 문장 임베딩 모델 카드](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [Word2Vec 원논문](https://arxiv.org/abs/1301.3781)
- [BERT 원논문](https://aclanthology.org/N19-1423/)
- [Sentence-BERT 원논문](https://aclanthology.org/D19-1410/)
