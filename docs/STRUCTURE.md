# QuPicT 파일 구성 (실험 전까지)

범위: 0단계 의미 선택 → 1단계 수집 → 2·3단계 필터링 → 4단계 DB·임베딩 → 5단계 답변.
웹사이트와 실험은 나중에 추가합니다.

## 구성

```text
2026_DS_NewsQnA/
├── README.md
├── requirements.txt
├── .env.example          API 키, DB 주소
├── .gitignore
├── schema.sql            DB 테이블 6개
├── targets.yaml          수집 대상 9개 (검색어, 고를 의미, 기간)
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
└── old/                  실험 때 다시 쓸 기존 파일 (지금은 안 건드림)
    ├── eval_filter.py
    ├── eval_retrieval.py
    ├── build_labeled_set.py
    └── debug_similarity.py
```

로컬 개발용 PostgreSQL은 `docker compose up -d`로 띄웁니다 (`docker-compose.yml`, `.env`에 `POSTGRES_PASSWORD` 필요).

## 실행

```bash
python run.py init-db     # 테이블 만들기
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
| `qupict/retrieval.py` | `app/vector_store.py` (FAISS는 pgvector로 바꿀 때까지 유지) |
| `qupict/qa.py` | `app/rag_chain.py` |
| `qupict/db.py`, `qupict/indexing.py` | 새로 작성 |

## DB 테이블

| 테이블 | 내용 |
| --- | --- |
| `query_senses` | 검색어별 의미 후보 (같은 검색어면 재사용) |
| `topics` | 수집 대상 + 선택한 의미 + 기간 |
| `articles` | 제목, 링크, 게시일, 언론사, 본문, 수집 일시 |
| `article_images` | 이미지 원본, 캡션, LLM 설명 |
| `article_topics` | 기사별 판정(관련/무관/보류), 근거 한 줄, 정답 라벨 |
| `article_chunks` | 청크 텍스트 + 임베딩 벡터 |

정해 둔 규칙 (자세한 컬럼은 `schema.sql`, 저장·조회 함수는 `qupict/db.py`)

- **기사는 link 기준 한 번만 저장**: 여러 topic에 걸린 기사는 `article_topics`로 연결합니다. 본문·청크·임베딩은 한 번만 만듭니다.
- **기간은 topic에 저장하지 않음**: 질문한 시각 기준 `[지금 - period_days, 지금]` 안에 **발행된**(`articles.pub_date`) 기사만 씁니다.
- **같은 topic 재사용**: `(검색어, 의미 이름)`이 같으면 같은 topic입니다. 다시 요청하면 검색은 다시 하지만, DB에 있는 기사는 크롤링하지 않고 새 기사만 가져와 저장합니다 (`collect.collect_to_db`).
- **중복은 `duplicate_of` 하나로 표시**: 값이 있으면 중복이고, 그 행의 `status`는 비워 둡니다 (DB 제약으로 강제).
- **판정과 정답 라벨은 따로**: `status`(시스템 판정, `stage` = rule/llm)와 `label`(사람이 붙인 정답)은 같은 값(관련/무관/보류)을 쓰는 서로 다른 컬럼입니다.
- **임베딩은 1536차원** (`text-embedding-3-small`). 모델을 바꾸면 `schema.sql`의 `vector(1536)`도 바꿔야 합니다.
- **DB 위치**: 코드는 `.env`의 `DATABASE_URL`만 봅니다. 개발은 로컬 Docker로 하고, 팀 공유·서비스는 pgvector를 지원하는 관리형 PostgreSQL(예: Supabase, 서울 리전)로 옮길 때 주소만 바꾸면 됩니다.

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
