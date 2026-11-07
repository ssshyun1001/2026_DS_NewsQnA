# QuPicT 진행 상황 정리 (2026-11-07)

팀원 공유용 문서입니다. 지금까지 한 일, 내 컴퓨터에서 실행하는 방법, 정해야 할 것을 정리했습니다.
관련 PR: [#1 폴더 구조 정리 + 0단계 검색어 의미 선택](https://github.com/ssshyun1001/2026_DS_NewsQnA/pull/1)

---

## 1. 한눈에 보기

| 단계 | 파일 | 상태 |
| --- | --- | --- |
| 폴더 구조 정리 | 전체 | ✅ 완료 (기존 코드 내용은 그대로, 위치와 import만 변경) |
| LLM 연결 | `qupict/llm.py` | ✅ 학교 팩트챗 Gateway로 연결 완료 |
| 네이버 뉴스 API | `qupict/collect.py` | ✅ 연결 확인 (NCP API HUB) |
| 0단계 의미 선택 | `qupict/sense.py` | ✅ 구현 + 실제 9개 검색어로 확인 |
| DB | `schema.sql`, `qupict/db.py` | ✅ 테이블 6개 + 단계별 저장·조회 함수 + 점검 스크립트 (아래 7) |
| 1단계 수집 개편 | `qupict/collect.py` | ✅ 의미 기반 검색 + 본문·언론사·이미지 수집 |
| 2·3단계 필터링 개편 | `qupict/filtering.py` | ⏳ |
| 4·5단계 인덱싱·검색·답변 | `indexing.py`, `retrieval.py`, `qa.py` | ⏳ DB 준비 완료 → `db.py` 함수로 바꾸기 |

`python run.py 애플 30`으로 **검색어 → 의미 선택 → 수집 → (기존) 필터링 → FAISS → 질문**까지 메모리에서 한 번에 돌아갑니다.
검증 키워드 입력은 없어졌고, 의미 선택이 대신합니다. 필터링은 아직 기존 방식입니다.
**DB는 준비됐습니다.** 각 단계가 `qupict/db.py` 함수로 저장·조회하도록 바꾸는 게 다음 일입니다 (아래 7-4).

---

## 2. 폴더 구성

```text
2026_DS_NewsQnA/
├── run.py              실행 파일 (기존 main.py)
├── targets.yaml        평가용 수집 대상 9개 + 고를 의미
├── schema.sql          DB 테이블 6개 (python run.py init-db)
├── docker-compose.yml  로컬 PostgreSQL + pgvector (docker compose up -d)
├── requirements.txt
├── .env.example        API 키 설정 예시 (.env로 복사해서 사용)
├── qupict/             파이프라인 코드: 단계 하나 = 파일 하나
│   ├── config.py       설정값, targets.yaml 읽기
│   ├── schemas.py      단계 사이에 주고받는 데이터 형태
│   ├── llm.py          LLM·임베딩 연결 (여기 한 곳에서만 만듦)
│   ├── db.py           DB 연결 + 단계별 저장·조회 함수
│   ├── sense.py        [0단계] 검색어 → 의미 후보 → 선택
│   ├── collect.py      [1단계] 의미 기반 검색 + 본문·언론사·이미지 수집
│   ├── filtering.py    [2·3단계] 중복 제거 + 관련성 필터
│   ├── indexing.py     [4단계] (예정)
│   ├── retrieval.py    [5단계] 검색 (현재 FAISS)
│   └── qa.py           [5단계] 답변 생성
├── tests/
│   └── check_db.py     DB 점검 (python tests/check_db.py)
├── old/                실험 단계에서 다시 쓸 평가·라벨링 스크립트 (지금은 실행 안 됨)
└── docs/
    ├── PROGRESS.md     이 문서
    └── STRUCTURE.md    폴더 설계, 기존→새 파일 대응표, DB 저장 형태
```

---

## 3. 내 컴퓨터에서 실행하기

### 3-1. 설치

- **Python 3.11 또는 3.12**를 쓰세요. 3.13은 `numpy==1.26.4` 설치가 실패합니다.

```bash
py -3.12 -m venv .venv
```

```bash
source .venv/Scripts/activate
```
(PowerShell은 `.venv\Scripts\Activate.ps1`, PyCharm은 인터프리터를 `.venv\Scripts\python.exe`로 지정)

```bash
pip install -r requirements.txt
```

### 3-2. `.env` 만들기

`.env.example`을 `.env`로 복사한 뒤 값을 채웁니다. **`.env`는 git에 올라가지 않습니다. 키를 채팅방·이슈·PR에 붙여 넣지 마세요.**

| 변수 | 어디서 받나 | 비고 |
| --- | --- | --- |
| `OPENAI_API_KEY` | **팀 공용 키** (서연 계정의 학교 팩트챗에서 발급) | 서연에게 직접 받기. git·공개 채팅방에 올리지 않기 |
| `OPENAI_BASE_URL` | 그대로 둠 | `https://factchat-cloud.mindlogic.ai/v1/gateway` |
| `OPENAI_CHAT_MODEL` | 그대로 둠 | `claude-haiku-5-5` |
| `OPENAI_EMBEDDING_MODEL` | 그대로 둠 | `text-embedding-3-small` (1536차원) |
| `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET` | 네이버 클라우드 플랫폼 (아래 4-1) | 팀에서 하나를 같이 써도 됨 |
| `POSTGRES_PASSWORD` | 각자 정함 | 로컬 Docker DB 비밀번호 (아래 3-4) |
| `DATABASE_URL` | 위 비밀번호로 작성 | `postgresql://qupict:<비밀번호>@localhost:5432/qupict` |

### 3-3. 잘 되는지 확인

```bash
python -m qupict.sense 애플
```
→ "기업 Apple / 과일 사과" 후보가 나오고 번호로 고를 수 있으면 성공입니다.

```bash
python -m qupict.sense --targets
```
→ `targets.yaml`의 대상들을 한 번에 처리해 결과를 보여 줍니다.

### 3-4. DB 띄우기 (Docker)

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) 설치 후 실행
2. `.env`에 `POSTGRES_PASSWORD`와 `DATABASE_URL`을 채움 (두 곳 비밀번호 같게)
3. 컨테이너 실행 (pgvector가 들어 있는 PostgreSQL 17)

```bash
docker compose up -d
```

4. 패키지 다시 설치 (`psycopg`, `pgvector` 추가됨) → 테이블 만들기

```bash
pip install -r requirements.txt
```

```bash
python run.py init-db
```

5. 점검 (LLM·네이버 API는 안 부름, 크레딧 0)

```bash
python tests/check_db.py
```
→ 마지막에 `🎉 전부 통과 (20개)`가 나오면 성공입니다. 점검용 가짜 데이터는 자동으로 지워집니다.

| 증상 | 확인할 것 |
| --- | --- |
| `DB에 연결하지 못했습니다` / connection refused | Docker Desktop이 켜져 있는지, `docker compose ps`에서 `qupict-postgres`가 `healthy`인지 |
| password authentication failed | `DATABASE_URL` 비밀번호 = `POSTGRES_PASSWORD`인지. 컨테이너를 처음 만든 뒤 비밀번호를 바꿨다면 볼륨에 예전 비밀번호가 남아 있음 → `docker compose down -v` 후 다시 `up -d` (**데이터가 지워짐**) |
| `No module named 'psycopg'` | `pip install -r requirements.txt` |
| 5432 포트 사용 중 | PC에 다른 PostgreSQL이 켜져 있음 → 끄거나 `docker-compose.yml` 포트를 `5433:5432`로 바꾸고 URL도 5433으로 |

DB 안을 직접 보고 싶으면:

```bash
docker exec -it qupict-postgres psql -U qupict -d qupict
```
→ `\dt`(테이블 목록), `\dx`(확장 — `vector`가 보이면 정상), `SELECT count(*) FROM articles;`, `\q`(나가기)

---

## 4. 사용하는 API

### 4-1. 네이버 뉴스 검색 — 네이버 클라우드 플랫폼(NCP) **API HUB**

> 네이버 **개발자센터(developers.naver.com)가 아닙니다.** 개발자센터 키를 넣으면 401 오류가 납니다.

1. [ncloud.com](https://www.ncloud.com) 가입 → 콘솔 로그인 (결제수단 등록을 요구할 수 있음)
2. 오른쪽 위 **리전 & 플랫폼**: 한국 → [적용]
3. `Menu > All Services > Application Services > NAVER API HUB > Subscription` → **[서비스 이용 신청]** → 약관 동의
4. 같은 메뉴의 **Application** → **[Application 등록]** → **NAVER 검색 > 뉴스** 선택 → 이름 입력(영문·숫자·하이픈, 예: `qupict-news`) → [완료]
5. 목록에서 **[인증 정보]** → Client ID, Client Secret을 `.env`에 입력

- 요금: **현재 한시적 무료** (나중에 종량제 전환 예정, 사전 공지)
- 한도: 검색 API 합산 월 77.5만 건, 키당 초당 50건 — 우리 사용량(수백 건)엔 충분
- 친구와 공유: 같은 키를 같이 쓰거나, `서비스 > Management & Governance > Sub Account`로 콘솔 권한 부여 (무료)

### 4-2. LLM·임베딩 — 학교 팩트챗 Gateway

- OpenAI 호환 API라서 코드에서는 주소(`OPENAI_BASE_URL`)만 바꿔서 씁니다.
- **ChatGPT 구독과는 별개**이고, 학교 계정에 **월 3,000 크레딧**이 있습니다 (매월 1일 갱신).
- **팀은 서연 계정의 키 하나를 같이 씁니다.** 크레딧 3,000을 팀 전체가 나눠 쓰므로, 큰 작업(파이프라인 전체 실행, 실험)은 돌리기 전에 팀에 말해 주세요.
- 같은 계정 안에서 사람별로 키를 따로 만들어 나눠 쓰면, 크레딧은 그대로 공유하면서 대시보드의 **"키별 사용량"** 으로 누가 얼마나 썼는지 볼 수 있고, 키가 새도 그 키만 지우면 됩니다.
- OpenAI 공식 API로 바꾸려면 `.env`에서 `OPENAI_BASE_URL`을 비우고 OpenAI 키를 넣으면 됩니다. 코드는 바꿀 필요 없습니다.

**실제로 측정한 크레딧 (1회 호출 기준)**

| 모델 | 1회당 | 메모 |
| --- | --- | --- |
| `claude-haiku-5-5` | 약 0.31 | **기본값**. 가장 빠르고(2~3초) 결과가 깔끔 |
| `gpt-5.6-luna` | 약 0.36 | 사용 가능 |
| `gemini-3.7-flash` | 약 1.44 | 느리고(9초) 비쌈 — 사용하지 않음 |
| `text-embedding-3-small` | 약 0 | 임베딩은 거의 무료 |

**예상 사용량**: 파이프라인 1회(9개 대상 × 기사 100건) 약 50~100 크레딧.
**주의**: 5단계 답변은 기사 본문을 많이 넣어서 1회 3~5 크레딧 — 실험에서 답변을 수백 번 만들면 500~1,000 크레딧이 들 수 있습니다. 답변 결과는 저장해서 재사용하고, 실험은 몰아서 하지 않는 게 좋습니다.

---

## 5. 0단계: 검색어 의미 선택 (`qupict/sense.py`)

### 동작

```text
검색어 "애플"
  → LLM이 의미 후보 생성 (자주 쓰이는 순, 최대 3개)
      1. 기업 Apple   별칭 [Apple Inc.]  연관어 [아이폰, 맥북, 팀 쿡, iOS …]
      2. 과일 사과     연관어 [과수원, 농가, 수확, 사과값 …]
  → 하나 선택 (번호 / targets.yaml 이름 / "기타" 직접 입력)
  → 결과(SelectedSense)
      기사에서 찾을 표기 = 검색어 + 별칭          → 1단계 수집
      배제어 = 고르지 않은 의미의 연관어          → 2단계 규칙 필터
      고르지 않은 의미 목록                       → 3단계 LLM 검증의 대조 기준
```

### 실제 결과 (`python -m qupict.sense --targets`)

**① 한 가지 의미를 찾는 검색어**

| 검색어 | 고른 의미 | 결과 |
| --- | --- | --- |
| 서울여대 | 서울여자대학교 | ✅ 후보 1개라 자동 선택 (별칭: 서울여자대학교, SWU) |
| 삼전 | 기업 삼성전자 | ✅ 별칭에 '삼성'이 섞이는 경우가 있어 `targets.yaml`에 별칭 고정 |
| 베라 | 배스킨라빈스 | ✅ LLM이 모름 → **직접 입력**으로 처리 |
| 강남 | 방송인 강남 | ✅ 지역/방송인 구분 잘 됨. LLM이 지어낸 정보는 팀이 확인한 값(이상화의 남편)으로 교체 |
| 애플 | 기업 Apple | ✅ 기업 / 과일 구분 |
| 당근 | 기업 당근(당근마켓) | ✅ 기업 / 채소 구분 |

**② 같은 검색어, 다른 의미 (사용자가 고르는 의미에 따라 결과가 달라지는지)**

| 검색어 | 고른 의미 | 연관어 (찾을 단어) | 배제어 (걸러낼 단어) |
| --- | --- | --- | --- |
| 눈 | 날씨 눈 | 대설, 적설량, 제설, 폭설, 한파 | 시력, 안과, 백내장, 각막 … |
| 눈 | 신체 기관 눈 | 시력, 안과, 백내장, 각막 | 대설, 적설량, 제설, 폭설 … |
| 후드 | 의류 후드 | 후드티, 맨투맨, 패션 | 환기, 가스레인지, 기름때 … |
| 후드 | 주방 후드 | 환기, 가스레인지, 기름때 | 후드티, 맨투맨, 패션 … |
| 마스크 | 보건용 마스크 | KF94, 미세먼지, 식약처 | 짐 캐리(영화), 가면무도회 … |
| 마스크 | 마스크팩 (직접 입력) | 스킨케어, 화장품, 보습 | KF94, 방역, 미세먼지, 짐 캐리 … |

→ 의미를 바꾸면 연관어와 배제어가 **정확히 뒤바뀝니다.**

### 발견한 LLM 문제와 대응

| 문제 | 예시 | 대응 |
| --- | --- | --- |
| 정보를 지어냄 | 방송인 강남 → 별칭 "조남식", 정의 "본명 조성민", 연관어 "비와이" | `targets.yaml`에 사람이 확인한 `definition` / `aliases` / `related_terms`를 적으면 LLM 값 대신 사용 |
| 모르는 단어에 억지 후보 | 베라 → "인물 베라 (확실하지 않음)", 연관어 없음 | **연관어 없는 후보는 버림** → 후보가 없으면 직접 입력으로 |
| 한자·외국 문자 | "出荷", "積雪" | **한자가 섞인 단어는 제거** (iOS, HBM 같은 영문은 유지) |
| 검색어 일부가 연관어로 | 서울여대 → "여대" (성신여대 기사까지 점수를 받게 됨) | **검색어 안에 포함된 단어는 연관어에서 제거** |
| 두 의미에 같은 단어 | 기업 Apple·과일 사과 둘 다 "가격" | 양쪽 모두에서 제거 (원래 있던 규칙) |
| 직접 입력이 LLM 후보와 같은 의미 | 직접 입력 "보건용 마스크" ↔ LLM 후보 "얼굴 마스크" → '방역', '황사'가 배제어가 됨 | **연관어가 2개 이상 겹치는 후보는 같은 의미로 보고 배제어에서 제외** |
| 너무 넓은 별칭 | 삼전 → '삼성' (삼성 라이온즈 기사까지 들어옴) | `targets.yaml`에 별칭 고정 |
| 같은 입력에도 결과가 조금씩 다름 | temperature=0인데도 연관어가 실행마다 다름 | **DB에 첫 결과를 저장해 재사용** — `db.get_sense_candidates` / `save_sense_candidates` 준비됨, `sense.py` 연결 필요 |

**남은 한계**: 배제어에 "인물", "출연" 같은 일반 단어가 섞일 수 있습니다 (예: 베라). 2단계 규칙 필터는 단어 하나로 바로 떨어뜨리지 않고 점수로 판단하고 3단계 LLM 검증도 있지만, 9개 대상은 사람이 결과를 한 번씩 확인하는 게 좋습니다.

---

## 6. 1단계: 뉴스 수집 (`qupict/collect.py`)

### 검색어 만들기

고른 의미로 네이버 검색어를 여러 개 만들어 **동시에** 검색합니다 (네이버 API는 무료).

| 순서 | 검색어 | 예: 눈 → 신체 기관 눈 | 이유 |
| --- | --- | --- | --- |
| 1 | 검색어 + 연관어 3개 | 눈 시력, 눈 안과, 눈 각막 | 의미가 좁혀진 검색. "눈"만 검색하면 날씨 기사가 결과를 다 채움 |
| 2 | 별칭 | 안구, 시력 | 같은 대상의 다른 이름 |
| 3 | 검색어 단독 | 눈 | 연관어가 없는 관련 기사도 놓치지 않기 (섞이는 다른 의미는 2·3단계가 거름) |

검색어마다 최대 100건을 받은 뒤 **검색어별로 돌아가며 하나씩** 뽑아 상한(기본 100건)을 채웁니다. 첫 검색어 결과만으로 상한이 차는 것을 막습니다.

### 기사마다 가져오는 것

| 항목 | 방법 |
| --- | --- |
| 본문 | 네이버 뉴스 주소가 있으면 거기서 먼저 (구조가 일정함), 실패하면 언론사 원문 |
| 언론사 | 네이버 로고 → 네이버 형식 "언론사 \| 네이버" → 언론사 사이트 이름 (기자 이름은 쓰지 않음) |
| 이미지 | 본문 속 사진 주소 + 원래 사진 설명(캡션), 기사당 최대 5장. 이미지 → 텍스트 설명은 4단계에서 |
| 찾은 검색어 | 어떤 검색어로 들어온 기사인지 (분석·실험용) |
| 수집 시각 | 검색 결과가 날짜마다 달라지므로 기록 |

### 실제 확인 (기사 30건씩)

| 대상 | 검색 → 본문 확보 | 언론사 확인 | 이미지 있는 기사 |
| --- | --- | --- | --- |
| 강남 / 방송인 강남 | 30 → 27 | 25 | 13 |
| 눈 / 신체 기관 눈 | 30 → 30 | 27 | 12 |

- **고친 기존 버그:** EUC-KR로 된 언론사 페이지를 UTF-8로 읽어 **본문 글자가 깨진 채** 저장되던 문제 → 페이지 문자셋을 판별해서 읽도록 수정 (깨진 본문 0건 확인)
- **남은 한계:** 일부 언론사는 본문 앞에 "홈 > 방송/TV" 같은 메뉴 글자가 섞입니다. 필요하면 언론사별 본문 위치를 추가하면 됩니다.
- 검색어 단독("눈") 결과에는 다른 의미 기사(미국 하원, 운세 등)가 섞여 들어옵니다. **이걸 걸러 내는 게 2·3단계의 역할**이고, 실험 A의 Baseline(API 결과 그대로)이 됩니다.

## 7. DB (`schema.sql`, `qupict/db.py`)

### 7-1. 테이블

| 테이블 | 내용 | 단계 |
| --- | --- | --- |
| `query_senses` | 검색어별 LLM 의미 후보 (같은 검색어면 재사용) | 0 |
| `topics` | 검색어 + 고른 의미 (사람이 고친 별칭·정의 포함), 배제어, 대조할 다른 의미, 마지막 수집 시각 | 0 |
| `articles` | 제목, 링크, 발행일, 언론사, 본문, 수집 시각, 기사 임베딩 | 1 |
| `article_images` | 이미지 주소, 원래 캡션, LLM 설명 | 1·4 |
| `article_topics` | topic별 기사 판정: 찾은 검색어, 중복 표시, 판정(관련/무관/보류)·단계·근거, 정답 라벨 | 2·3 |
| `article_chunks` | 청크 텍스트(본문/이미지 설명) + 임베딩 (HNSW 인덱스) | 4·5 |

### 7-2. 정한 규칙

| 규칙 | 내용 | 이유 |
| --- | --- | --- |
| 기사는 한 번만 저장 | `articles.link`가 UNIQUE. 여러 topic에 걸린 기사는 `article_topics`로 연결 | 본문·이미지 설명·청크·임베딩을 한 번만 만듦 (크레딧 절약) |
| 기간 = 발행일 | topic에는 기간을 저장하지 않음. 조회할 때 **[질문한 시각 − N일, 질문한 시각]** 안에 발행된 기사만 | 같은 topic을 기간만 바꿔 다시 쓸 수 있음. 과거 시점 기준 재현도 가능 (`now=` 인자) |
| 같은 topic 재사용 | (검색어, 의미 이름)이 같으면 같은 topic. 다시 요청하면 검색은 다시 하되, **DB에 있는 기사는 크롤링 안 하고 새 기사만** 가져와 저장 | 수집은 한 번만, 최신 기사만 추가 |
| 중복 표시 | `duplicate_of`(대표 기사)에 값이 있으면 중복. 그 행의 판정은 비움 (DB 제약으로 강제) | 한 정보를 두 곳에 쓰면 어긋날 수 있음 |
| 판정 ≠ 정답 | `status`(시스템 판정, `stage` = rule/llm)와 `label`(사람이 붙인 정답)은 별도 컬럼 | 실험에서 둘을 비교 |
| 필터 다시 돌리기 | `reset_topic_filter`로 판정만 지우고 다시 실행 (정답 라벨은 유지) | 규칙 수정 → 해당 단계만 재실행 |
| 임베딩 1536차원 | `text-embedding-3-small`. 모델을 바꾸면 `schema.sql`의 `vector(1536)`도 바꿔야 함 | |

### 7-3. 팀 DB 운영

- 코드는 `.env`의 **`DATABASE_URL` 하나만** 봅니다. 주소만 바꾸면 어떤 PostgreSQL이든 같은 코드로 동작합니다.
- **지금 (개발)**: 각자 로컬 Docker. 테스트하다 데이터를 지워도 남에게 영향이 없습니다.
- **팀 공유·서비스 단계**: pgvector를 지원하는 관리형 PostgreSQL(추천: Supabase 서울 리전)을 하나 만들어 팀이 같은 `DATABASE_URL`을 씁니다. 한 명이 수집한 기사를 모두가 쓰므로 "수집은 한 번만"이 지켜집니다. → 시점과 계정은 팀에서 정하기 (아래 8)

### 7-4. 단계별로 쓰는 함수 (각 단계 담당)

기사는 id 대신 **link**로 가리킵니다. 지금의 `Article` 객체를 그대로 넘기면 됩니다.

| 단계 | 함수 | 하는 일 |
| --- | --- | --- |
| 공통 | `init_db()` | 테이블 만들기 (`python run.py init-db`) |
| 0 | `get_sense_candidates(query)` / `save_sense_candidates(c)` | 의미 후보 캐시 읽기 / 저장 |
| 0 | `save_topic(selected)` → topic id, `get_topic(query, sense_name)` | topic 만들기·재사용 / 읽기 |
| 1 | `collect.collect_to_db(selected, period_days)` → topic id | **1단계 DB 버전 (완성)**: topic 저장 → 검색 → 이미 있는 기사는 연결만 → 새 기사만 크롤링·저장 |
| 1 | `link_known_articles(topic_id, metas)`, `save_articles(topic_id, articles)` | 위 함수가 안에서 씀 |
| 2·3 | `get_topic_articles(topic_id, period_days=, statuses=, include_duplicates=, unjudged_only=)` | 조건에 맞는 기사 (임베딩·이미지 포함) |
| 2·3 | `save_article_embeddings(articles)` | 기사 임베딩 저장 (다음에 다시 계산 안 함) |
| 2·3 | `save_duplicates(topic_id, {중복 link: 대표 link})` | 중복 표시 |
| 2·3 | `save_judgments(topic_id, [Judgment(link, status, stage, reason, model)])` | 관련/무관/보류 판정 |
| 2·3 | `save_labels(topic_id, {link: 라벨})`, `reset_topic_filter(topic_id)` | 정답 라벨 / 판정 초기화 |
| 4 | `get_images_to_describe(topic_id)`, `save_image_description(image_id, text)` | 설명 없는 이미지 / 설명 저장 |
| 4 | `get_articles_without_chunks(topic_id)`, `save_chunks(link, [Chunk(...)])` | 청크 없는 기사 / 청크 교체 저장 |
| 5 | `search_chunks(topic_id, 질문 임베딩, k, period_days=)` | topic·판정·기간 조건 + 벡터 유사도 top-k (출처 기사 포함) |

> **필터링 담당 참고:** 같은 topic에 새 기사가 들어오면 `unjudged_only=True`로 새 기사만 꺼내고, 중복 비교는 이미 있는 대표 기사(`include_duplicates=False`)와도 해야 합니다. 기사 임베딩은 DB에 있으니 다시 계산하지 않아도 됩니다.

---

## 8. 정해야 할 것

- [x] 눈·후드·마스크 — 의미별로 2개씩 넣어 "고르는 의미에 따라 결과가 달라지는지" 확인 (위 ②)
- [x] 강남 정보 확인 — 이상화의 남편 맞음
- [ ] **학교 팩트챗 크레딧을 공모전 프로젝트에 써도 되는지** 규정 확인
- [ ] **언론사 기사 본문·이미지 수집이 허용되는지** 확인 (기획안의 "확인 필요" 항목)
- [ ] PR #1 리뷰 후 머지 여부
- [x] DB 구조 — 기사 한 번만 저장, 기간 = 발행일, topic 재사용 시 새 기사만 추가, 중복은 `duplicate_of`로만 표시
- [ ] **팀 공용 DB(Supabase 등)로 언제 옮길지, 누구 계정으로 만들지**
- [ ] 줄바꿈(LF/CRLF) 통일 — 지금 `git diff`에 내용이 안 바뀐 파일도 전부 바뀐 걸로 나옴. `.gitattributes` 추가 필요

---

## 9. 다음 작업

1. **`filtering.py` 개편** — 배제어 점수 규칙 필터(방법 1-c) + LLM 관련/무관/보류 판정과 근거 한 줄 → 결과는 `db.save_duplicates` / `save_judgments`로 저장
2. **`sense.py`에 의미 후보 캐시 연결** — `get_sense_candidates`로 먼저 찾고, 없을 때만 LLM 호출 후 `save_sense_candidates`
3. **`indexing.py`(이미지 설명·청크·임베딩), `retrieval.py`(FAISS → `db.search_chunks`)** 작성
4. **`run.py` 단계별 명령** (`senses`, `collect`, `filter`, `index`, `ask`) — 지금은 `init-db`만 있음
