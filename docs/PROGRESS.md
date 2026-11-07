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
| 0단계 의미 선택 | `qupict/sense.py` | ✅ 구현 + 실제 9개 검색어로 확인 + **DB 캐싱 연결** |
| DB | `schema.sql`, `qupict/db.py` | ✅ 테이블 6개 + 단계별 저장·조회 함수 + 점검 스크립트 (아래 7) |
| 1단계 수집 개편 | `qupict/collect.py` | ✅ 의미 기반 검색 + 본문·언론사·이미지 수집 |
| 2·3단계 필터링 개편 | `qupict/filtering.py` | ✅ 중복 제거 → 규칙 점수 → LLM 관련/무관/보류 + 근거, DB 저장 |
| 4·5단계 인덱싱·검색·답변 | `indexing.py`, `retrieval.py`, `qa.py` | ✅ 청크·이미지 설명·임베딩 저장, pgvector 검색, 청크 기반 답변 |
| 실행 명령 | `run.py`, `qupict/pipeline.py` | ✅ `ask`(DB 버전), `collect-all` · `filter-all` · `index-all` · `label` · `status` |
| 실험 코드 (7단계) | `experiments/` | ✅ 실험 A·B·C + 질문 세트 도구 (`exp-a` · `make-qa` · `exp-b` · `exp-c`), 강남으로 시험 실행 |
| 오프라인 테스트 | `tests/offline/` | ✅ 9개 묶음, `python tests/run_offline.py` (크레딧 0, DB 없이도 됨) |

**데이터 수집 전에 만들 수 있는 코드는 모두 만들었습니다.** 남은 일은 **모여서 12개 대상 수집 + 정답 라벨링 + 질문 세트 만들기**, 그리고 실험 실행입니다 (아래 3-6, 12).

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
│   ├── filtering.py    [2·3단계] 중복 제거 → 1차 규칙 필터 → 2차 LLM 검증
│   ├── indexing.py     [4단계] 이미지 설명 → 청크 → 임베딩 저장
│   ├── retrieval.py    [5단계] pgvector 검색 (+HyDE·Multi-Query), FAISS는 비교 실험용
│   └── qa.py           [5단계] 답변 생성
├── experiments/        [7단계] 실험 A·B·C, 질문 세트(qa_set.yaml), 결과(results/)
├── tests/
│   ├── check_db.py     DB 점검 (python tests/check_db.py, 실제 DB 사용)
│   ├── run_offline.py  오프라인 테스트 전체 실행 (크레딧 0)
│   └── offline/        단계별 오프라인 테스트
├── old/                예전 평가·라벨링 스크립트 (실행 안 됨, experiments/가 대신함 → 지워도 됨)
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
| `DATABASE_URL` | **DB 담당에게 개인 메시지로 받기** (팀 공용 Supabase, 아래 9-3) | 혼자 실험할 땐 로컬 Docker 주소 `postgresql://qupict:<비밀번호>@localhost:5432/qupict` |
| `POSTGRES_PASSWORD` | 각자 정함 | 로컬 Docker를 쓸 때만 (아래 3-4) |

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

> 팀 공용 DB(9-3)를 쓰면 1~3번은 건너뛰고 `pip install`과 점검(5번)만 하면 됩니다. 로컬 Docker는 혼자 실험할 때 씁니다.

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

### 3-5. 실행 명령 모음 (`run.py`)

| 명령 | 하는 일 |
| --- | --- |
| `python run.py 애플 30` | 검색어 + 기간 → 의미 고르기 → 수집 → 필터 → 인덱싱 → **질문** (DB 버전). `--max 30`으로 수집 건수 제한 |
| `python run.py ask 애플 30 --memory` | DB 없이 메모리에서 (예전 방식, FAISS) |
| `python run.py collect-all` | `targets.yaml` 대상 전부 수집 (0·1단계) |
| `python run.py filter-all` | 전부 필터링 (2·3단계). `--redo`: 판정을 지우고 처음부터 (정답 라벨은 유지) |
| `python run.py label` | **사람이 정답 라벨 붙이기** (1 관련 / 2 무관 / 3 보류 / s 건너뛰기 / b 이전 / q 끝) |
| `python run.py index-all` | 전부 인덱싱 (4단계). `--no-images`: 이미지 설명 없이 (LLM 비용 0) |
| `python run.py status` | topic별 현황: 기사·중복·판정·라벨·청크 수 |
| `python run.py init-db` | DB 테이블 만들기 |

- 일괄 명령은 모두 `--target 강남`처럼 **검색어 하나만** 할 수 있습니다.
- 모든 명령은 **아직 안 한 것만** 처리합니다 → 중간에 끊겨도 다시 실행하면 이어서 하고, 비용이 다시 들지 않습니다.
- `ask`에서 고른 의미가 이미 DB에 topic으로 있으면 **저장된 기준(사람이 고친 정의·별칭 포함)** 을 그대로 씁니다.
- 답변의 출처에는 **답변이 실제로 인용한 기사만** 보여 줍니다 (`Answer.cited`, 실험 C의 출처 정확도용).
- `label`은 시스템 판정을 보여 주지 않습니다 (정답이 시스템 판정에 끌려가지 않게). 5건마다 모아서 저장하고, 끝낼 때(q)도 저장합니다.

### 3-6. 모이는 날 순서

```bash
python run.py collect-all
```
```bash
python run.py filter-all
```
```bash
python run.py label
```
```bash
python run.py status
```

1. **collect-all**: 12개 항목 × 최대 100건. 네이버 검색은 무료, 처음 보는 검색어만 의미 후보 LLM 호출 (약 0.3 크레딧씩)
2. **filter-all**: 새 기사를 LLM으로 판정 (100건당 약 10~15 크레딧 예상)
3. **label**: 사람이 나눠서 라벨링 — topic마다 번호를 골라서 하므로 **사람마다 다른 topic**을 맡으면 됩니다
4. **status**로 남은 라벨 수 확인
5. (나중에) **index-all**: 질문 답변·실험 B·C용. 이미지 설명 때문에 크레딧이 가장 많이 드는 단계라 따로 실행
6. **make-qa**: 라벨을 붙이며 기사를 읽은 김에 실험 B·C용 질문을 만든다 (`--suggest`면 LLM 초안). 대상마다 5~10개 정도

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
  → DB(query_senses)에 후보가 있으면 그대로 사용 (LLM 호출 없음)
    없으면 LLM이 의미 후보 생성 (자주 쓰이는 순, 최대 3개) → DB에 저장
      1. 기업 Apple   별칭 [Apple Inc.]  연관어 [아이폰, 맥북, 팀 쿡, iOS …]
      2. 과일 사과     연관어 [과수원, 농가, 수확, 사과값 …]
  → 하나 선택 (번호 / targets.yaml 이름 / "기타" 직접 입력)
  → 결과(SelectedSense)
      기사에서 찾을 표기 = 검색어 + 별칭          → 1단계 수집
      배제어 = 고르지 않은 의미의 연관어          → 2단계 규칙 필터
      고르지 않은 의미 목록                       → 3단계 LLM 검증의 대조 기준
```

### DB 캐싱 (`load_sense_candidates`)

같은 검색어는 **처음 만든 후보만** 씁니다. LLM은 temperature=0이어도 실행마다 연관어가 조금씩 달라서, 캐싱이 없으면 배제어(2단계 기준)가 계속 바뀌어 실험을 재현할 수 없습니다.

| 상황 | 동작 |
| --- | --- |
| DB에 후보 있음 | DB 것 사용 (LLM 비용 0) |
| DB에 없음 | LLM으로 만들고 DB에 저장 |
| LLM이 후보를 못 만듦 (예: 베라) | 저장하지 않음 → 다음에 다시 시도 |
| `DATABASE_URL` 없음 / DB 연결 실패 | 멈추지 않고 LLM만 사용 (경고 출력) |

- 실제 확인 (팀 공용 DB, "애플" 3번): 1회차 LLM → DB 저장, 2·3회차 DB에서 읽음. LLM 호출 1번, 세 번 모두 같은 결과
- `run.py`, `python -m qupict.sense`, `--targets` 모두 이 함수를 씁니다. DB 없이 LLM만 부르려면 `generate_sense_candidates`
- 후보를 **다시 만들고 싶을 때**: 지금은 DB의 해당 검색어 행을 지워야 합니다 (같은 이름은 덮어쓰지만, 이름이 바뀐 옛 후보는 남아서 섞이기 때문)

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
| 같은 입력에도 결과가 조금씩 다름 | temperature=0인데도 연관어가 실행마다 다름 | **DB에 첫 결과를 저장해 재사용** (`load_sense_candidates`) ✅ |

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
| 이미지 | 본문 속 사진 주소 + 원래 사진 설명(캡션), 기사당 최대 5장. 아이콘·로고·기자 프로필 사진은 제외 |
| 이미지 파일 | **DB에 저장하지 않음** (팀 결정). 4단계에서 관련 기사의 이미지 설명을 만들 때만 `download_images()`로 받아서 씀 (5KB 미만·5MB 초과는 버림) |
| 찾은 검색어 | 어떤 검색어로 들어온 기사인지 (분석·실험용) |
| 수집 시각 | 검색 결과가 날짜마다 달라지므로 기록 |

### 이미지 파일 내려받기

- **팀 결정: 이미지 파일은 DB에 저장하지 않습니다.** Supabase 무료 용량(500MB)을 아끼기 위해서입니다. DB(`article_images`)에는 주소·캡션·LLM 설명만 저장합니다.
- 1단계(수집)는 주소·캡션만 저장하고, 파일은 **4단계에서 "관련" 기사의 이미지 설명을 만들 때만** `download_images()`로 받아 LLM에 보내고 버립니다.
- 받은 파일은 `ArticleImage.data`(바이트)와 `content_type`에 들어 있습니다 (메모리에서만 사용).
- 실제로 받아 보니 5KB 미만 파일은 기자 프로필 사진, 작은 그래픽, 미리보기 자리 이미지였습니다 → 버리도록 함.
- 확인 (기사 20건씩): 강남 39개 중 31개, 눈 34개 중 33개 받음 / 한 장 5~73KB, 대상 하나당 합계 약 0.7MB

### DB에 저장하기 (`collect_to_db`)

```python
result = await collect_to_db(selected, period_days=30)
print(result.summary())   # 검색 30건 → DB 재사용 12건 + 새로 저장 17건 (본문 실패 1건, 이미지 40개)
```

1. 고른 의미를 `topics`에 저장 (같은 검색어·의미면 기존 topic 재사용)
2. 네이버 검색 (무료라 매번 다시 함)
3. **DB에 이미 있는 기사는 크롤링하지 않고** `article_topics`로 연결만 함
4. 새 기사만 크롤링 → `articles`, `article_images`, `article_topics`에 저장
5. 돌려주는 `CollectResult`: topic id, 검색·재사용·새로 저장·본문 실패 건수, 이미지 수

이후 단계는 `db.get_topic_articles(topic_id, period_days=...)`로 기사를 꺼내 씁니다.

### 실제 확인 (기사 30건씩)

| 대상 | 검색 → 본문 확보 | 언론사 확인 | 이미지 있는 기사 |
| --- | --- | --- | --- |
| 강남 / 방송인 강남 | 30 → 27 | 25 | 13 |
| 눈 / 신체 기관 눈 | 30 → 30 | 27 | 12 |

- **고친 기존 버그:** EUC-KR로 된 언론사 페이지를 UTF-8로 읽어 **본문 글자가 깨진 채** 저장되던 문제 → 페이지 문자셋을 판별해서 읽도록 수정 (깨진 본문 0건 확인)
- **남은 한계:** 일부 언론사는 본문 앞에 "홈 > 방송/TV" 같은 메뉴 글자가 섞입니다. 필요하면 언론사별 본문 위치를 추가하면 됩니다.
- 검색어 단독("눈") 결과에는 다른 의미 기사(미국 하원, 운세 등)가 섞여 들어옵니다. **이걸 걸러 내는 게 2·3단계의 역할**이고, 실험 A의 Baseline(API 결과 그대로)이 됩니다.

## 7. 2·3단계: 필터링 (`qupict/filtering.py`)

### 흐름

| 순서 | 단계 | 하는 일 | DB 저장 |
| --- | --- | --- | --- |
| 1 | 중복 제거 | 임베딩 유사도 0.9 이상이면 같은 사안 → 본문이 가장 긴 기사만 대표로 | `article_topics.duplicate_of` |
| 2 | 1차 규칙 (`stage=rule`) | **분명히 아닌 것만** 무관 처리 (아래 표) | `status='무관'`, `reason` |
| 3 | 2차 LLM (`stage=llm`) | 고른 의미 vs 고르지 않은 의미를 대조해 **관련 / 무관 / 보류 + 근거 한 줄** | `status`, `reason`, `judged_model` |

**규칙 점수** (기획안 방법 1-c) = 연관어 점수 − 배제어 점수, 단어마다 제목에 있으면 2점·본문에만 있으면 1점

| 경우 | 판정 |
| --- | --- |
| 검색어·별칭이 기사에 아예 없음 (예: "서울여대" 검색에 걸린 성신여대 기사) | 무관 (rule) |
| 점수 ≤ −2 (다른 의미 단어가 분명히 더 많음) | 무관 (rule) |
| 나머지 | LLM으로 |

- 규칙으로 "관련"은 주지 않습니다. 단어만으로는 실제 주제와 단순 언급을 구분할 수 없어서입니다.
- 기준값(−2)은 정답 라벨이 생기면 실험으로 조정합니다 (`filtering.RULE_REJECT_SCORE`).
- 질문 답변(run.py)에는 **관련 + 보류** 기사를 씁니다. 확신 없는 기사를 버리면 놓치는 정보가 생기기 때문입니다.

### 사용

```python
outcome = await filter_articles(articles, selected)        # 메모리 버전 (run.py)
outcome = await filter_topic(topic_id, selected)           # DB 버전: 새 기사만 판정해서 저장
outcome = await filter_topic(topic_id, selected, redo=True) # 규칙·프롬프트를 바꿨을 때 전부 다시 (정답 라벨은 유지)
print(outcome.summary())   # 중복 0건 | 규칙 무관 0건 | LLM 관련 6 · 무관 3 · 보류 0건
```

- DB 버전은 **아직 판정 안 한 기사만** 처리하고, 이미 판정한 기사와 비슷한 새 기사는 중복으로 표시합니다. 기사 임베딩도 DB에 저장해 다시 계산하지 않습니다.
- LLM이 답하지 못한 기사는 판정하지 않고 남겨서 다음 실행 때 다시 시도합니다.

### 실제 확인 (팀 공용 DB, 방송인 강남 9건)

| 판정 | 기사 | 근거 (LLM) |
| --- | --- | --- |
| 관련 6 | 괌 관광객 구조, 이서진 밥값 내기, 드라마 OST 등 | "가수 강남이 … 구한 내용이 주 소재" 등 |
| 무관 3 | 로또 판매점 목록 | "'서울 강남구 삼성로' 등 지역명 언급" |
| | 박서진 성형 시술 | "'강남'은 피부과 시술을 다녀온 '지역 강남'을 뜻함" |
| | 김수찬 공연 | "'성남과 인천, 강남 등' 지역 공연 일정 나열" |

- **고친 점:** 처음엔 LLM이 판정을 먼저 쓰고 근거를 나중에 써서, 근거는 "지역 의미, 무관 쪽"인데 판정은 '관련'인 경우가 있었습니다. **근거를 먼저 쓰고 판정하게** 답 형식 순서를 바꿔서 해결했습니다.
- 비용: 9건 판정 약 1~2 크레딧, 13초

## 8. 4·5단계: 인덱싱·검색·답변 (`indexing.py`, `retrieval.py`, `qa.py`)

### 4단계 인덱싱 (`index_topic`)

| 순서 | 하는 일 | 저장 |
| --- | --- | --- |
| 1 | **이미지 설명**: 관련·보류 기사의 이미지를 LLM이 2~3문장으로 설명 (기사당 최대 2장) | `article_images.llm_description` |
| 2 | **청크**: 본문을 약 500자씩 문장 단위로 자름 (앞 청크의 마지막 문장을 겹쳐서 시작). 이미지 설명은 1장 = 청크 1개 | |
| 3 | **임베딩**: 청크마다 (기사 제목 + 청크)로 임베딩 | `article_chunks` (source = body / image) |

- 청크가 없는 기사 + 이번에 이미지 설명이 새로 생긴 기사만 처리 → 여러 번 실행해도 안전
- `describe_images=False`: 이미지 설명 없이 (LLM 비용 0) / `rebuild=True`: 청크 규칙을 바꿨을 때 전부 다시
- **이미지는 주소를 LLM에 그대로 넘기고, 실패하면 파일을 받아서 보냅니다.** 실제로 해 보니 게이트웨이가 WebP·일부 JPEG 파일을 직접 올리면 거절했지만 주소로는 됐습니다.
- **비용 주의**: 이미지 1장 설명 ≈ 의미 후보 1번과 비슷 (입력 약 470토큰). 12개 대상 × 관련 기사 50건 × 2장이면 수백 크레딧이 될 수 있어 기사당 2장으로 제한

### 5단계 검색 (`search_topic`) + 답변 (`generate_answer_from_chunks`)

```text
질문 → [원 질문 + HyDE 가상 문장, Multi-Query 변형 질문 2개]
     → 각 문장으로 pgvector 검색: SQL 조건(topic · 관련/보류 · 최근 N일) 안에서 청크 유사도
     → 같은 청크는 가장 높은 유사도로 합침, 0.5 미만 버림 → 상위 5개
     → 기사별로 묶어 (기사 N) 번호 → LLM 답변 + 출처 (근거 종류: 본문 / 이미지 설명)
```

- 기사 전체 대신 **청크만** 넣어서 답변 LLM에 들어가는 글이 크게 줄었습니다 (예전 기사 10건 × 6,000자 → 청크 5개 × 약 500자)
- 이미지 설명을 근거로 쓰면 답변에 "(기사 N, 이미지)", 출처에 `[이미지 설명]`으로 표시 (기획안 요구사항)
- FAISS 메모리 버전(`build_in_memory_store`, `retrieve_relevant_articles`)은 실험 B 비교용으로 남겨 둠

### 실제 확인 (팀 공용 DB, 방송인 강남 관련 6건)

| 단계 | 결과 |
| --- | --- |
| 인덱싱 | 본문 청크 22개 + 이미지 청크 9개 |
| "강남이 괌에서 무슨 일을 했어?" | 괌 동굴에서 관광객 구조 — 두 기사 인용, 정답 |
| "밥값 내기에서 얼마를 냈어?" | 1,519싱가포르달러(150만 원대) — 정답 |
| "괌 영상 사진에는 어떤 모습이?" | 이미지 청크 5개 검색 → "(기사 1, 이미지)" 인용, 출처 `[이미지 설명]` |

- 질문 1개당 약 8~14초 (HyDE·Multi-Query LLM 2번 + DB 검색 3번 + 답변). DB가 싱가포르 리전이라 DB 검색이 느린 편

### DB 담당에게 (4단계)

- `indexing.described_images()`: "기사별로 이미 설명이 있는 이미지" 조회가 `db.py`에 없어서 `indexing.py`에 SELECT 하나를 넣었습니다. 원하면 `db.py`로 옮겨 주세요.

## 9. DB (`schema.sql`, `qupict/db.py`)

### 9-1. 테이블

| 테이블 | 내용 | 단계 |
| --- | --- | --- |
| `query_senses` | 검색어별 LLM 의미 후보 (같은 검색어면 재사용) | 0 |
| `topics` | 검색어 + 고른 의미 (사람이 고친 별칭·정의 포함), 배제어, 대조할 다른 의미, 마지막 수집 시각 | 0 |
| `articles` | 제목, 링크, 발행일, 언론사, 본문, 수집 시각, 기사 임베딩 | 1 |
| `article_images` | 이미지 주소, 원래 캡션, LLM 설명 | 1·4 |
| `article_topics` | topic별 기사 판정: 찾은 검색어, 중복 표시, 판정(관련/무관/보류)·단계·근거, 정답 라벨 | 2·3 |
| `article_chunks` | 청크 텍스트(본문/이미지 설명) + 임베딩 (HNSW 인덱스) | 4·5 |

### 9-2. 정한 규칙

| 규칙 | 내용 | 이유 |
| --- | --- | --- |
| 기사는 한 번만 저장 | `articles.link`가 UNIQUE. 여러 topic에 걸린 기사는 `article_topics`로 연결 | 본문·이미지 설명·청크·임베딩을 한 번만 만듦 (크레딧 절약) |
| 기간 = 발행일 | topic에는 기간을 저장하지 않음. 조회할 때 **[질문한 시각 − N일, 질문한 시각]** 안에 발행된 기사만 | 같은 topic을 기간만 바꿔 다시 쓸 수 있음. 과거 시점 기준 재현도 가능 (`now=` 인자) |
| 같은 topic 재사용 | (검색어, 의미 이름)이 같으면 같은 topic. 다시 요청하면 검색은 다시 하되, **DB에 있는 기사는 크롤링 안 하고 새 기사만** 가져와 저장 | 수집은 한 번만, 최신 기사만 추가 |
| 중복 표시 | `duplicate_of`(대표 기사)에 값이 있으면 중복. 그 행의 판정은 비움 (DB 제약으로 강제) | 한 정보를 두 곳에 쓰면 어긋날 수 있음 |
| 판정 ≠ 정답 | `status`(시스템 판정, `stage` = rule/llm)와 `label`(사람이 붙인 정답)은 별도 컬럼 | 실험에서 둘을 비교 |
| 필터 다시 돌리기 | `reset_topic_filter`로 판정만 지우고 다시 실행 (정답 라벨은 유지) | 규칙 수정 → 해당 단계만 재실행 |
| 임베딩 1536차원 | `text-embedding-3-small`. 모델을 바꾸면 `schema.sql`의 `vector(1536)`도 바꿔야 함 | |

### 9-3. 팀 공용 DB (Supabase)

프로젝트를 다 같이 하므로 **DB는 팀 공용 하나**를 씁니다. 한 명이 수집한 기사·판정·라벨을 모두가 같이 봅니다 ("수집은 한 번만").
로컬 Docker(3-4)는 혼자 실험할 때만 씁니다. 코드는 `.env`의 `DATABASE_URL`만 보므로 **주소만 바꾸면 공용 ↔ 로컬 전환**이 됩니다.

**왜 Supabase**: pgvector 지원, 서울 리전, 무료(500MB — 9개 대상 × 100건이면 임베딩 포함 약 100~200MB 예상), 웹 화면(Table Editor)에서 표로 데이터를 보고 라벨도 직접 고칠 수 있음.

**만들기 (DB 담당 1명이 한 번)**

1. [supabase.com](https://supabase.com) 가입 → **New project**
   - Region: **Northeast Asia (Seoul)**
   - Database Password: 강한 비밀번호 (팀에 전달할 값, 따로 보관)
2. 프로젝트 화면 위쪽 **Connect** → **Session pooler** 주소 복사 (`...pooler.supabase.com:5432/postgres`)
   - Direct connection은 IPv6 전용이라 학교·집 네트워크에서 안 될 수 있음 → **Session pooler 사용**
3. 자기 `.env`의 `DATABASE_URL`을 그 주소로 바꾸고 (`[YOUR-PASSWORD]` 자리에 비밀번호, 끝에 `?sslmode=require`)

```bash
python run.py init-db
```

```bash
python tests/check_db.py
```
→ `🎉 전부 통과`면 완료. (점검 스크립트는 `__DB점검__` 표시가 붙은 가짜 데이터만 만들고 지우므로 공용 DB에서 돌려도 안전합니다)

4. 팀원 초대: **Organization Settings → Team → Invite** (웹 화면에서 데이터 보기용)

**팀원 (각자)**

1. DB 담당에게 `DATABASE_URL`을 **개인 메시지로** 받기 (단톡방·이슈·PR·코드에 올리지 않기)
2. `.env`의 `DATABASE_URL`을 바꾸고 `python tests/check_db.py`로 확인

**주의할 것**

| 항목 | 내용 |
| --- | --- |
| 일시 정지 | 무료 플랜은 **7일 동안 접속이 없으면 정지**됩니다. 대시보드에서 Restore하면 데이터 그대로 돌아옵니다 |
| 같이 쓰는 DB | `reset_topic_filter`, 테이블 삭제(`DROP`), `docker compose down -v` 같은 지우는 작업은 **팀에 먼저 말하기**. 실험은 가능하면 로컬 DB에서 |
| 테이블 구조 변경 | `schema.sql`은 `IF NOT EXISTS`라서 **이미 있는 테이블의 컬럼 추가·변경은 반영 안 됩니다.** 바꿀 때는 DB 담당이 `ALTER TABLE`을 공용 DB에 직접 실행하고 `schema.sql`도 같이 고칩니다 |
| 비용·크레딧 | 같은 topic을 두 사람이 동시에 수집하지 않기 (DB는 안전하지만 LLM 크레딧이 두 번 듦) |
| 로컬 데이터 옮기기 | 주소를 바꿔도 데이터는 안 옮겨집니다. 필요하면 `pg_dump`로 내보내 공용 DB에 넣기 (DB 담당에게 요청) |
| 서비스 단계 | 사용자가 많아지면 커넥션 풀(`psycopg_pool`)과 마이그레이션 도구(Alembic 등)를 추가해야 합니다. 지금 팀 개발에는 필요 없음 |

### 9-4. 단계별로 쓰는 함수 (각 단계 담당)

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

## 10. 7단계: 실험 (`experiments/`)

| 명령 | 실험 | 비교 | 지표 | 필요한 것 |
| --- | --- | --- | --- | --- |
| `python run.py exp-a` | A. 필터링 | baseline / 1-a 별칭 / 1-b 검증어 / 1-c 규칙 점수 / 2-b LLM만 / **제안(규칙+LLM)** | Precision·Recall·F1·Accuracy·무관 유입률·LLM 호출 수 | 정답 라벨 |
| `python run.py make-qa` | B·C 질문 세트 | — | — | 관련 기사 |
| `python run.py exp-b` | B. 검색 | base / HyDE / Multi-Query / both (`--faiss`: 예전 FAISS도) | Recall@5·MRR·검색 시간 | 질문 세트 + `index-all` |
| `python run.py exp-c` | C. 답변 | LLM 단독 / 기본 RAG(필터 없음) / **QuPicT** | 정답률·근거 일치율·환각 비율·출처 정확도 (LLM 채점) | 질문 세트 + `index-all` |

- 모두 `--target 강남`처럼 검색어 하나만 할 수 있고, 결과는 `experiments/results/`에 `실험_날짜시각.json/.csv`로 저장됩니다 (CSV는 엑셀에서 바로 열림).
- **실험 A**: 정답 '보류'는 지표에서 빼고, 시스템 판정 '보류'는 관련으로 셉니다 (서비스가 보류 기사도 답변에 쓰므로). 1-c는 기준값별 결과도 같이 저장 → 규칙 기준값 정할 때 사용. 2-b는 규칙으로 걸러진 기사만 LLM을 추가로 부르고 캐시에 저장 (다시 돌려도 비용 0).
- **1-b 검증어**: `targets.yaml`의 `validation_term` (예: 강남 → "방송인"). 제가 넣은 값이니 **팀 확인 필요**.
- **질문 세트**: `experiments/qa_set.yaml` (git에 올려 팀이 같이 씀). 지금은 강남 질문 3개가 예시로 들어 있습니다.
- **실험 C 채점**: LLM이 정답 근거 기사만 보고 주장별로 판정 (근거를 먼저 쓰고 판정). 결과 JSON에 주장별 판정이 남아 있어 **일부를 사람이 검수**하면 됩니다. 출처 정확도는 "정답 근거로 지정한 기사"만 맞다고 보므로 같은 내용의 다른 기사를 인용하면 낮게 나올 수 있습니다.
- **비용**: 실험 C는 질문 1개당 LLM 약 10번 → `--limit`로 나눠 돌리기.

### 강남으로 시험 실행한 결과 (질문 3개, 참고용)

| 실험 | 결과 |
| --- | --- |
| A | 임시 라벨(시스템 판정 복사)로 흐름만 확인 — 1-b(검증어 "방송인")는 관련 8건 중 3건만 찾음 (Recall 0.375) |
| B | 유사도 기준 0.2일 때 모든 방식 Recall@5 = 1.0 → **0.5로 올리니 일부 방식에서 0.667** (OST 질문의 최고 유사도가 0.53이라 아슬아슬) |
| C | LLM 단독은 3개 모두 "확인 불가"(정답 0), 기본 RAG 2/2 정답(1개 채점 실패 → 재시도 추가), **QuPicT 2/3** — OST 질문은 0.5 기준 때문에 근거가 하나도 안 남아 "확인 불가" |

### 유사도 기준 0.5 (팀 결정)

- `RELEVANCE_SIMILARITY_THRESHOLD` (기본 0.5, `.env`로 바꿀 수 있음): 질문과 청크의 유사도가 이 값 미만이면 답변 근거로 쓰지 않습니다.
- 엄격하게 확실한 근거만 쓰지만, **질문 표현에 따라 관련 청크도 0.48~0.54 근처**라 "기사를 찾지 못했습니다"로 끝나는 질문이 생깁니다 (위 OST 질문).
- 질문 세트가 쌓이면 **실험 B를 기준값별로 돌려서**(예: 0.4 / 0.45 / 0.5) 확인하는 걸 추천합니다.

## 11. 정해야 할 것

- [ ] **실험 A 1-b 검증어** — `targets.yaml`의 `validation_term` 12개가 적당한지 (예: 강남 → "방송인", 애플 → "아이폰")
- [ ] **유사도 기준 0.5 유지 여부** — 질문 세트가 생기면 실험 B로 0.4·0.45·0.5 비교 후 결정

- [x] **이미지 파일 DB 저장 안 함** — 4단계에서 관련 기사의 이미지만 그때 받아서 LLM에 보냄 (Supabase 용량 절약). DB 변경 없음
- [x] **네이버 뉴스 링크는 DB에 저장하지 않음** — 크롤링할 때만 쓰고(검색 결과에 있음), 출처·중복 확인은 `link`로 함. `articles.naver_link` 칸은 쓰지 않으므로 지워도 됨 (DB 담당 선택)

- [x] 눈·후드·마스크 — 의미별로 2개씩 넣어 "고르는 의미에 따라 결과가 달라지는지" 확인 (위 ②)
- [x] 강남 정보 확인 — 이상화의 남편 맞음
- [x] **학교 팩트챗 크레딧을 공모전 프로젝트에 써도 되는지** 규정 확인
- [ ] **언론사 기사 본문·이미지 수집이 허용되는지** 확인 (기획안의 "확인 필요" 항목)
- [ ] **정답 라벨 종류 맞추기 (DB 담당)** — 기획안 실험은 정답을 `관련 / 무관 / 동음이의어`로 나누는데 ("동음이의어 유입률" 지표), 지금 `article_topics.label`은 `관련 / 무관 / 보류`만 허용. 라벨에 `동음이의어`를 추가할지 정하기
- [ ] PR #1 리뷰 후 머지 여부
- [x] DB 구조 — 기사 한 번만 저장, 기간 = 발행일, topic 재사용 시 새 기사만 추가, 중복은 `duplicate_of`로만 표시
- [x] 팀 공용 DB 사용 — Supabase (9-3)
- [ ] **Supabase 프로젝트 만들 사람(계정) 정하기**, 만든 뒤 팀원에게 `DATABASE_URL` 전달
- [ ] 줄바꿈(LF/CRLF) 통일 — 지금 `git diff`에 내용이 안 바뀐 파일도 전부 바뀐 걸로 나옴. `.gitattributes` 추가 필요

---

## 12. 다음 작업

1. ~~`filtering.py` 개편~~ ✅, ~~`sense.py` 의미 후보 캐시 연결~~ ✅
2. **12개 대상 일괄 수집 + 정답 라벨링** (팀이 같이 모여서)
3. ~~`indexing.py`, `retrieval.py`~~ ✅
4. ~~`run.py` DB 버전 + 일괄 명령~~ ✅ (`ask`, `collect-all`, `filter-all`, `index-all`, `label`, `status`)
5. ~~실험 코드~~ ✅ (A·B·C + 질문 세트 도구), ~~테스트 저장소로 옮기기~~ ✅
6. **(모여서)** `collect-all` → `filter-all` → `label` → `make-qa` → `index-all`
7. **실험 실행** `exp-a` → `exp-b` → `exp-c` → 결과 정리 (README 성능 평가 갱신)
8. (나중에) 웹사이트·대시보드
