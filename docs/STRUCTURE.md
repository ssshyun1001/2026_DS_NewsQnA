# QuPicT 파일 구성 (실험 전까지)

범위: 0단계 의미 선택 → 1단계 수집 → 2·3단계 필터링 → 4단계 DB·임베딩 → 5단계 답변.
웹사이트와 실험은 나중에 추가합니다.

## 구성

```text
2026_DS_NewsQnA/
├── README.md
├── requirements.txt
├── .env.example          API 키, DB(Firestore) 프로젝트
├── .gitignore
├── run.py                실행: python run.py <단계>
│
├── qupict/               파이프라인 코드: 단계 하나 = 파일 하나
│   ├── config.py         설정값
│   ├── schemas.py        기사·의미 등 데이터 형태
│   ├── llm.py            OpenAI 채팅·임베딩 연결
│   ├── db.py             DB 연결, 저장·조회
│   ├── sense.py          [0단계] 검색어 → 의미 후보 TOP3 + 연관어
│   ├── collect.py        [1단계] 네이버 API 수집 + 본문·이미지 크롤링
│   ├── filtering.py      [2·3단계] 중복 제거 → 규칙 필터 → LLM 검증(관련/무관/보류)
│   ├── indexing.py       [4단계] 이미지 설명 → 청크 → 임베딩
│   ├── retrieval.py      [5단계] SQL 조건 + 벡터 검색, HyDE, Multi-Query
│   └── qa.py             [5단계] 근거 기반 답변 + 출처
│
├── old/                  예전 평가·라벨링 스크립트 (지금 구조에 맞게 고침, 아래 표)
│
├── evalset/              정답 JSON 평가셋 (DB 안 씀)
│   ├── build_labeled_set.py  라벨링 → labeled_set_N.json (검색어 단위로 나눔)
│   ├── exp_a_filtering.py    실험 A: 필터링 방법 비교 (labeled_set_*.json 기준, 결과 evalset/results/)
│
├── experiments/          [7단계] 실험 (python evaluate.py exp-b / make-qa / sim)
│   ├── exp_b_retrieval.py  실험 B: 검색 방식 비교
│   ├── qa_set.py / .yaml   실험 B 질문 세트
│   ├── similarity.py       유사도 확인 (기준값 없이 청크 유사도 보기)
│   └── common.py           지표 계산, 결과 저장
│
└── tests/                오프라인 테스트 (python tests/run_offline.py, 크레딧 0)
```

`old/`는 예전 평가·라벨링 스크립트를 **지금 구조(`qupict/`)에 맞게 고친 것**입니다. DB 없이도 돌아가고(파일·메모리), 프로젝트 맨 위 폴더에서 실행합니다.

| 파일 | 하는 일 | 실행 | 같은 일을 하는 run.py 명령 |
| --- | --- | --- | --- |
| `evalset/build_labeled_set.py` (옮김) | 검색어 반복 입력(q=종료) → LLM 의미 고르기 → 수집 → 중복 제거 → 최신 대표 기사 150건에 1 관련 / 2 무관 라벨 → `evalset/labeled_set_N.json`에 검색어 단위로 나눠 누적 | `python evalset/build_labeled_set.py` | `run.py label` (DB에 저장) |
| `eval_filter.py` | `labeled_set_*.json`으로 V0 필터 없음 / V1 규칙 필터 / V2 규칙 + LLM 검증 비교 (Precision·Recall·F1·Accuracy) | `python old/eval_filter.py` | `run.py exp-a` (방법 6가지) |
| `eval_retrieval.py` | 골든셋(제논·당근·서울여대·강남 39문항)으로 Base / HyDE / Multi-Query / Both 검색 성공률 (키워드 포함 여부) | `python old/eval_retrieval.py 강남` | `run.py exp-b` (Recall@5·MRR) |
| `debug_similarity.py` | 기준값 없이 질문과의 유사도 출력 (HyDE 없음 / HyDE) | `python old/debug_similarity.py 강남 "방송인 강남" "질문"` | `run.py sim` (방식 4가지, 질문 세트 전체) |
| `_common.py` | 위 스크립트 공용: 의미 정하기 (LLM 의미 후보에서 이름으로 찾기 → 없으면 후보에서 고르기) | — | — |

- 예전 "검증 키워드" 자리는 **고른 의미 이름**(예: `방송인 강남`)이 대신합니다.
- `eval_retrieval.py`·`debug_similarity.py`는 DB에 topic이 있으면 실제 서비스 검색(DB 벡터 검색)을 쓰고, 없거나 `--memory`를 주면 예전처럼 그 자리에서 수집합니다.
- `eval_retrieval.py`의 골든 질문은 2026년 7월 기사로 만든 것이라, 최근 30일 기사만 있는 DB에서는 성공률이 낮게 나옵니다 (`[DEBUG]` 줄로 확인).

DB는 팀 공용 Firestore(GCP `qupict`) 하나를 씁니다. 연결 방법은 `docs/PROGRESS.md` 3-4 (로컬 PostgreSQL · `docker-compose.yml`은 2026-11-16에 없앰).

## 실행

```bash
python admin.py init-db   # 검색 인덱스 만들기 (Firestore)
python run.py senses      # 0단계
python run.py collect     # 1단계
python run.py filter      # 2·3단계
python run.py index       # 4단계
python run.py ask 서울여대  # 5단계 (질문하기)
```

각 단계는 DB에서 앞 단계 결과를 읽고 결과를 DB에 저장합니다. 수집은 한 번만 하고, 규칙을 고치면 해당 단계만 다시 돌립니다.

## 기존 파일 → 새 파일

| 새 파일 | 기존 파일 |
| --- | --- |
| `run.py` | `main.py` |
| `qupict/config.py` | `config.py` |
| `qupict/schemas.py` | `app/schemas.py` |
| `qupict/llm.py` | `app/embeddings.py` |
| `qupict/sense.py` | `app/query_understanding.py` |
| `qupict/collect.py` | `app/news_search.py` + `app/crawler.py` |
| `qupict/filtering.py` | `app/deduplication.py` + `app/relevance_filter.py` |
| `qupict/retrieval.py` | `app/vector_store.py` (FAISS는 실험 B 비교용으로 유지, 서비스는 DB 벡터 검색) |
| `qupict/qa.py` | `app/rag_chain.py` |
| `qupict/db.py`, `qupict/indexing.py` | 새로 작성 |

## DB 컬렉션 (Google Cloud Firestore)

| 컬렉션 | 내용 |
| --- | --- |
| `query_senses` | 검색어별 의미 후보 (같은 검색어면 재사용) |
| `topics` | 수집 대상 + 선택한 의미 |
| `articles` | 제목, 링크, 게시일, 언론사, 본문, 수집 일시, 임베딩 + 이미지(주소, 캡션, LLM 설명 · OCR 글자) |
| `article_topics` | 기사별 판정(관련/무관/보류), 근거 한 줄, 정답 라벨 |
| `chunks` | 청크 텍스트 + 임베딩 벡터 + 검색 대상 topic · 판정(`search_keys`) |

정해 둔 규칙 (자세한 필드와 저장·조회 함수는 `qupict/db.py`)

- **기사는 link 기준 한 번만 저장**: 여러 topic에 걸린 기사는 `article_topics`로 연결합니다. 본문·청크·임베딩은 한 번만 만듭니다.
- **기간은 topic에 저장하지 않음**: 질문한 시각 기준 `[지금 - period_days, 지금]` 안에 **발행된**(`articles.pub_date`) 기사만 씁니다.
- **같은 topic 재사용**: `(검색어, 의미 이름)`이 같으면 같은 topic입니다. 다시 요청하면 검색은 다시 하지만, DB에 있는 기사는 크롤링하지 않고 새 기사만 가져와 저장합니다 (`collect.collect_to_db`).
- **중복은 `duplicate_of` 하나로 표시**: 값이 있으면 중복이고, 판정은 대표 기사의 판정을 복사해 받습니다 (`db.copy_duplicate_judgments`).
- **판정과 정답 라벨은 따로**: `status`(시스템 판정, `stage` = rule/llm)와 `label`(사람이 붙인 정답)은 같은 값(관련/무관/보류)을 쓰는 서로 다른 필드입니다.
- **임베딩은 1536차원** (`text-embedding-3-small`). 모델을 바꾸면 `db.EMBEDDING_DIM`과 벡터 인덱스(`python admin.py init-db`)도 바꿔야 합니다.
- **DB 위치**: 팀 공용 DB는 Firestore(GCP `qupict`, 서울 `asia-northeast3`) 하나를 같이 씁니다 (2026-11-16 Supabase에서 옮김). 코드는 `.env`의 `FIRESTORE_PROJECT`를 봅니다. 연결 방법은 `docs/PROGRESS.md` 3-4, 9-3.

## DB 담당에게: 0단계에서 저장할 값

`qupict/sense.py`가 돌려주는 값은 `qupict/schemas.py`의 아래 형태입니다. `.model_dump()`로 dict가 됩니다.

| 형태 | 필드 | 저장 위치 (제안) |
| --- | --- | --- |
| `SenseCandidates` | `query`, `senses`(최대 3개, 순서 = 자주 쓰이는 순), `model` | `query_senses`: 검색어 + 순번 + 아래 `Sense` 필드 + 생성 모델·일시 |
| `Sense` | `name`, `definition`, `aliases[]`, `related_terms[]`, `is_custom` | 배열은 `TEXT[]` 또는 `JSONB` |
| `SelectedSense` | `query`, `sense`, `excluded_terms[]`, `other_senses[]` | `topics`: 선택한 sense 참조 + `excluded_terms` + 기간 |

- 같은 검색어는 `query_senses`에 이미 있으면 LLM을 다시 부르지 않고 그 후보를 씁니다 (캐싱 = 후보 일관성).
- 직접 입력한 의미(`is_custom = true`)는 LLM 후보가 아니므로 `topics`에 sense 내용을 직접 저장하거나, `query_senses`에 별도 행으로 넣고 표시해 주세요.
- `excluded_terms`는 2단계 규칙 필터의 배제어, `other_senses`는 3단계 LLM 검증의 대조 기준으로 씁니다.
