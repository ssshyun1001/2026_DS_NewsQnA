<p align="center">
  <img src="web/QuPicT_logo.svg" alt="QuPicT" width="220">
</p>

<h1 align="center">QuPicT 뉴스 Q&A</h1>

<p align="center">
  <b>검색어의 "의미"를 먼저 고르고, 그 의미의 기사만 모아, 기사에 적힌 내용으로만 답하는 뉴스 RAG 서비스</b>
</p>

<p align="center">
  🌐 <a href="https://qupict-920254012527.asia-northeast3.run.app/"><b>웹사이트 바로 가기</b></a>
</p>

---

## 왜 만들었나

"강남"으로 뉴스를 검색하면 **방송인 강남** 기사보다 **지역 강남**(부동산·상권·학원) 기사가 훨씬 많이 나옵니다. "애플"은 기업과 과일이, "당근"은 중고거래 플랫폼과 채소가 섞입니다.
이런 상태로 RAG를 만들면 엉뚱한 기사가 근거로 들어가고, LLM은 그럴듯한 오답을 만듭니다.

QuPicT는 이 문제를 **수집 이전 단계에서** 풉니다.

| 원칙 | 구현 |
| --- | --- |
| **의미를 먼저 고른다** | 검색어 → LLM이 의미 후보(최대 3개) 제시 → 사용자가 선택. 고른 의미의 연관어로 검색하고, 고르지 않은 의미의 연관어는 배제어가 됨 |
| **근거 없는 답은 하지 않는다** | 수집·판정을 통과한 기사의 청크만 근거로 쓰고, 유사도 0.5 미만은 버림. 답변에 `(기사 N)` 출처를 붙이고 실제로 인용한 기사만 보여 줌 |
| **LLM은 꼭 필요한 곳에만** | 중복 제거(임베딩·형태소·글자 n-gram), 1차 필터(규칙 점수), 많이 보도된 기사 요약(문장 추출)은 **LLM 없이** 처리. 같은 입력이면 항상 같은 결과 |
| **한 번 모은 것은 다시 쓴다** | 기사·판정·청크·의미 후보를 Firestore(벡터 검색)에 저장. 같은 검색어면 저장된 기사로 **바로 답하고**, 새 기사만 뒤에서 수집 |
| **감이 아니라 숫자로 정한다** | 정답 라벨 988건, 골든셋 118문항으로 필터링·검색 방식을 비교하고 그 결과로 기본값을 정함 |

---

## 동작 흐름

```text
 검색어 "강남"
   │
   ▼
 [0] 의미 선택      sense.py      LLM 의미 후보 → 사용자 선택 (DB 캐시, 많이 고른 순)
   │                               1. 방송인 강남  2. 지역 강남  …  / "기타" 직접 입력
   ▼
 [1] 수집           collect.py    "검색어 + 연관어"·별칭으로 네이버 뉴스 동시 검색
   │                               → 본문·언론사·이미지 크롤링 (DB에 있는 기사는 재사용)
   ▼
 [2] 중복 제거      filtering.py  제목 임베딩 · 검색어 문단 임베딩 · 고유명사 TF-IDF · 3-gram 자카드
   │                               4개 기준 중 2개 이상 넘으면 같은 사안 → 대표 기사만   (LLM 없음)
   ▼
 [3] 관련성 판정    filtering.py  1차 규칙 점수(연관어 − 배제어)로 분명한 무관만 제거
   │                               → 2차 LLM이 "고른 의미 vs 다른 의미" 대조 → 관련/무관/보류 + 근거
   ▼
 [4] 인덱싱         indexing.py   500자 청크 + 이미지 속 글자(OCR) → 임베딩 → Firestore
   │
   ▼
 [5] 검색·답변      retrieval.py  하이브리드 검색(벡터 + BM25 키워드, RRF) → 청크 → 근거 인용 답변
                    qa.py          최근 대화 2턴을 기억해 "그 사람 직업은?" 같은 후속 질문 처리
```

같은 검색어라도 **고른 의미에 따라 연관어와 배제어가 정확히 뒤바뀝니다.**

| 검색어 | 고른 의미 | 연관어 (찾을 단어) | 배제어 (걸러낼 단어) |
| --- | --- | --- | --- |
| 당근 | 당근마켓 | 중고거래, 동네생활, 당근페이, 직거래 | 농가, 수확, 당근즙, 레시피 |
| 당근 | 채소 당근 | 농가, 수확, 당근즙, 레시피 | 중고거래, 동네생활, 당근페이, 직거래 |
| 애플 | 기업 Apple | 아이폰, 맥북, 팀 쿡, iOS | 과수원, 농가, 사과값, 품종 |
| 애플 | 과일 사과 | 과수원, 농가, 사과값, 품종 | 아이폰, 맥북, 팀 쿡, iOS |

---

## 웹 서비스

**배포 주소: <https://qupict-920254012527.asia-northeast3.run.app/>** (Google Cloud Run, 서울 리전)

로컬에서 실행하려면:

```bash
python web.py           
```

1. 검색어 입력 → 의미 고르기 (후보 옆에 "직접 입력 · N회 선택" 표시)
2. 수집 · 중복 제거 · 판정 · 인덱싱 진행 상황을 로그로 표시 (줄마다 경과 시간, "DB 재사용 N건 + 새로 저장 M건")
3. **많이 보도된 기사 Top 5** — 비슷한 기사가 가장 많이 묶인 사안 5건과 한 줄 요약 (LLM 없이 기사 속 문장 그대로)
4. 질문하기 — 준비가 끝나기 전에도, 저장된 기사가 있으면 바로 답변

DB 재사용으로 체감 속도가 크게 달라졌습니다.

| 상황 | 이전 | 지금 |
| --- | --- | --- |
| 이미 준비한 검색어로 질문 받기 시작 (방송인 강남) | 약 13초 | **1.1초** |
| 같은 검색어 다시 준비 (당근) | 20.1초 | **5.2초** |

Google Cloud Run 배포용 `Dockerfile`이 포함되어 있습니다 (CPU판 torch로 이미지 크기 절감).


---

## 성능 평가

모든 실험 결과는 `evalset/results/`, `experiments/results/`에 CSV·JSON으로 남아 있습니다.

### 실험 A — 관련성 필터링 (`python evaluate.py exp-a`)

9개 topic(검색어 × 의미), **사람이 직접 라벨링한 기사 988건** 기준 (`evalset/labeled_set_*.json`).

| 방법 | Precision | Recall | F1 | 무관 유입률 |
| --- | ---: | ---: | ---: | ---: |
| baseline (API 결과 그대로) | 0.412 | 1.000 | 0.584 | 0.588 |
| 1-a 별칭 포함 여부 | 0.439 | 0.963 | 0.604 | 0.561 |
| 1-b 검증 키워드 (이전 버전 방식)\* | 0.786 | 0.527 | 0.631 | 0.214 |
| 1-c 규칙 점수만 | 0.482 | 0.774 | 0.594 | 0.518 |
| 2-b LLM 판정만 | 0.578 | 0.953 | **0.720** | 0.422 |
| **제안 (규칙 → LLM)** | **0.585** | 0.929 | 0.718 | **0.415** |

\* 1-b는 검증 키워드를 정할 수 있는 topic(349건)만 평가. Precision은 높지만 관련 기사의 절반 가까이를 놓칩니다.

- 제안 방식은 Recall 0.93을 유지하면서 무관 기사 유입을 **58.8% → 41.5%** 로 줄였습니다.
- 동음이의어가 뚜렷한 **방송인 강남**은 baseline Precision 0.105 → 제안 방식 **1.000** (F1 0.19 → 1.00).
- 반대로 지역 강남·과일 사과처럼 다른 의미와 경계가 넓은 topic은 개선 폭이 작아, 남은 과제로 두었습니다.

### 실험 B — 검색 방식 (`python evaluate.py exp-b`)

팀 골든셋 118문항 중 수집된 87문항, 그중 **근거 기사가 검색 대상에 들어 있는 57문항** 기준.

| 방식 | Recall@5 | MRR | 질문당 시간 |
| --- | ---: | ---: | ---: |
| base (질문 원문) | 0.526 | 0.447 | 2.4초 |
| **HyDE** | 0.684 | 0.582 | 3.6초 |
| Multi-Query | 0.614 | 0.518 | 7.3초 |
| HyDE + Multi-Query | **0.702** | **0.589** | 8.8초 |

- **HyDE 효과가 가장 큽니다**: +1.2초로 Recall@5 0.53 → 0.68.
- Multi-Query를 더하면 0.68 → 0.70이지만 시간이 2배 이상 → 시간 대비 효율은 HyDE 단독이 가장 좋습니다.
- **최종 서비스 검색 = Hybrid** (`experiments/exp_b_hyde_hybrid.py`, 골든셋 87문항): 벡터 검색 + BM25 키워드 검색을 RRF로 합치는 방식이 Recall@5 0.609 / MRR 0.508 / 0.93초로 가장 정확하고 빠릅니다 (같은 측정에서 HyDE 0.402, HyDE + Hybrid 0.598 · 2.39초). 그래서 웹·`run.py`는 HyDE · Multi-Query 없이 Hybrid만 씁니다 (`qupict/bm25.py`).
- 87문항 전체(수집 + 검색 종합) 기준 Recall@5는 base 0.345 → both 0.460.

---

## 시작하기

**요구 사항**: Python 3.11 / 3.12 (3.13은 `numpy==1.26.4` 설치 실패), Google Cloud Firestore (gcloud CLI로 로그인)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # API 키, FIRESTORE_PROJECT 입력
gcloud auth application-default login               # Firestore 로그인 (키 파일 없이 본인 Google 계정)
gcloud auth application-default set-quota-project qupict
python admin.py init-db                             # 검색 인덱스 만들기 (처음 한 번, 몇 분 걸림)
python tests/check_db.py                            # DB 점검 (LLM·API 호출 없음)
```

| 환경변수 | 설명 |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI 호환 API (비우면 OpenAI 공식, 학교 팩트챗 Gateway 지원) |
| `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL` | 채팅 모델 / `text-embedding-3-small` (1536차원) |
| `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | 네이버 클라우드 플랫폼 **API HUB** 뉴스 검색 키 (개발자센터 키는 401) |
| `FIRESTORE_PROJECT` | Firestore가 있는 GCP 프로젝트 id (`qupict`). 비우면 DB 없이 메모리 버전 |
| `RELEVANCE_SIMILARITY_THRESHOLD` | 답변 근거 최소 유사도 (기본 0.5) |
| `IMAGE_TEXT` | 이미지 처리: `ocr`(기본, EasyOCR·비용 0) / `vlm`(LLM 설명) / `off` |
| `DEDUP_BY`, `DEDUP_RULE`, `DEDUP_*_THRESHOLD` | 중복 제거 기준·규칙·임계값 |

### 실행 파일 — 용도마다 하나

| 파일 | 대상 | 주요 명령 |
| --- | --- | --- |
| `web.py` | 서비스 (브라우저) | `python web.py [--host 0.0.0.0] [--port 8080]` |
| `run.py` | 서비스 (터미널) | `python run.py 강남 30` · `--memory`(DB 없이 FAISS) · `--timing`(단계별 시간표) |
| `admin.py` | 서비스 DB 관리 | `collect 검색어` · `collect-all` · `filter-all [--redo]` · `index-all` · `status` · `init-db` |
| `evaluate.py` | 평가 | `exp-a` · `exp-b` · `sim` · `make-qa` (명령 없이 실행하면 번호 메뉴) |
| `evalset/build_labeled_set.py` | 정답 라벨링 | 검색어마다 의미 선택 → 대표 기사 150건에 관련/무관 라벨 → JSON 누적 |

모든 일괄 명령은 **아직 처리하지 않은 것만** 처리합니다. 중간에 끊겨도 다시 실행하면 이어서 하고, LLM 비용이 두 번 들지 않습니다.

### 테스트

```bash
python tests/run_offline.py   # 단계별 오프라인 테스트 18묶음 — API 크레딧 0, DB 없이도 실행 (DB는 가짜 Firestore)
```

---

## 프로젝트 구조

```text
2026_DS_NewsQnA/
├── web.py · run.py · admin.py · evaluate.py   실행 파일 (서비스 / DB 관리 / 평가)
├── Dockerfile                                  Cloud Run 배포
├── tools/                                      일회성 도구 (Supabase → Firestore 데이터 이전)
├── qupict/                 파이프라인: 단계 하나 = 파일 하나
│   ├── sense.py            [0] 검색어 → 의미 후보 → 선택
│   ├── collect.py          [1] 의미 기반 검색 + 본문·언론사·이미지 수집
│   ├── filtering.py        [2·3] 중복 제거 → 규칙 점수 → LLM 판정
│   ├── indexing.py         [4] 청크 · 이미지 글자(OCR) · 임베딩
│   ├── retrieval.py        [5] 하이브리드 검색 (Firestore 벡터 + BM25, RRF)
│   ├── bm25.py             [5] 하이브리드의 키워드(BM25) 검색 · RRF 합치기
│   ├── qa.py               [5] 근거 인용 답변, 대화 기억
│   ├── hot.py              많이 보도된 기사 Top 5 + 한 줄 요약 (LLM 없음)
│   ├── pipeline.py         단계 묶음, 할 일 없는 단계 건너뛰기
│   ├── db.py · schemas.py · llm.py · config.py
│   └── ocr.py · concurrency.py · timing.py
├── web/                    웹 화면 (index.html · app.js · style.css)
├── evalset/                정답 JSON 평가셋 + 실험 A
├── experiments/            실험 B, 골든셋(qa_set.csv), 중복 제거 확인 도구
├── tests/                  오프라인 테스트, DB 점검
├── old/                    이전 버전 평가 스크립트 (현재 구조에 맞게 수정)
└── docs/
    ├── PROGRESS.md         진행 상황 · 설계 결정 · 실험 상세 (팀 공유용)
    └── STRUCTURE.md        폴더 설계, DB 저장 형태
```

### DB 설계 요점

DB는 Google Cloud Firestore (서울 `asia-northeast3`, `(default)` DB)입니다. 컬렉션 구조는 `qupict/db.py` 맨 위 설명에 있습니다.

| 컬렉션 | 내용 |
| --- | --- |
| `query_senses` | 검색어별 의미 후보 캐시 (LLM 후보 + 직접 입력, 선택 횟수) |
| `topics` · `topic_keys` | 검색어 + 고른 의미, 배제어, 대조할 다른 의미 (번호는 `counters/topics`) |
| `articles` | 기사 본문·언론사·발행일·임베딩 + 이미지(주소·캡션·OCR/설명, 파일은 저장 안 함). 문서 id = link 해시 — 여러 topic이 공유 |
| `article_topics` | topic별 판정(관련/무관/보류 + 근거), 중복 표시, 정답 라벨 |
| `chunks` | 청크 + 임베딩. `search_keys`(topic · 판정)와 발행일로 거른 뒤 벡터 검색(`find_nearest`, 코사인) |

- 기사는 한 번만 저장하고, **기간은 발행일 기준**으로 조회할 때 정합니다 → 같은 topic을 기간만 바꿔 재사용.
- 시스템 판정(`status`)과 사람의 정답(`label`)을 별도 필드로 두어 실험에서 직접 비교합니다.

---

## 알려진 한계와 다음 단계

- LLM이 지어낸 별칭(예: 방송인 강남 → 엉뚱한 본명)이 그대로 쓰일 수 있음
- 유사도 기준 0.5 근처(0.46~0.54)의 질문은 실행마다 근거를 찾기도, 못 찾기도 함 → `evaluate.py sim`으로 기준값 비교 예정
- 중복 제거 `noun` 기준: 제목 고유명사가 검색어뿐이면 무관한 기사끼리 묶이는 문제 (검색어 제외 계산 검토 중)
- 의미별로 따로 수집하다 보니 다른 의미 topic에서 '무관'으로 걸러진 기사가 정작 그 의미 topic에는 없는 경우가 있음


자세한 설계 결정과 실험 기록은 [docs/PROGRESS.md](docs/PROGRESS.md)에 있습니다.
