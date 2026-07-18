# AI News Q&A Agent

관심 있는 주제를 입력하면 최근 뉴스를 긁어와서, 그 뉴스 안에서만 답을 찾아주는 챗봇입니다.
"할루시네이션 없는 뉴스 Q&A"를 만들어보고 싶어서 시작한 프로젝트예요.

주제 한 번 입력 → 뉴스 수집 → 그 다음부터는 계속 질문만 하면 됩니다. 질문할 때마다 다시 크롤링하지 않아요.

```
관심 분야를 입력하세요: AI 기업 제논
기간을 숫자로 입력하세요 (최근 며칠, 예: 7): 7
...
질문> 제논이 최근 집중하는 사업은 뭐야?
```

## 왜 만들었나

일반 LLM한테 최신 이슈를 물어보면 모르거나(학습 데이터 컷오프), 아는 척 지어내는 경우가 많잖아요.
그래서 "질문 들어오면 실시간으로 뉴스부터 찾고, 그 안에서만 답하게" 만들었습니다. DB에 쌓아두지 않고
그때그때 필요한 만큼만 모아서 씁니다 (인메모리).

## 어떻게 동작하나

1. 주제 입력받고 (예: "AI 기업 제논")
2. 며칠치 뉴스 볼지 입력받고 (예: 7일)
3. LLM이 검색어를 뽑음 — 이때 고유명사는 무조건 그대로 살리고, "AI 기업", "최근 이슈" 같은 애매한 검색어는 못 만들게 막아놨어요 (안 그러면 검색 예산이 이상한 데 다 씀)
4. 네이버 뉴스 API로 실검색 → 본문 크롤링
5. 동음이의어/무관 기사 걸러내고(예: "제논"이 실제로 본문에 있는지 확인), 중복 기사도 임베딩 유사도로 하나만 남김
6. 남은 기사들로 FAISS 인메모리 벡터스토어 구축
7. 이제부터 질문 루프. 질문마다 관련 기사 다시 찾아서(코사인 유사도 임계값 넘는 것만, 개수 고정 아님) 답변 생성
   - 기사에 답 있으면 → 기사 근거로 답하고 몇 번 기사인지 표시
   - 기사에 없으면 → "기사엔 없다"고 먼저 밝히고, 그 다음 LLM 일반 지식으로 보충 (구분해서 표시)

## 기술 스택

Python · LangChain(LCEL) · OpenAI(`gpt-4o-mini`, `text-embedding-3-small`) · FAISS(인메모리) · aiohttp/BeautifulSoup · Naver 뉴스 검색 API(NCP)

## 폴더 구조

```
main.py                      # 엔트리포인트 (수집 세션 + 질문 루프)
config.py                    # .env 로더
app/
  schemas.py                  # 데이터 모델
  query_understanding.py      # 주제 → 검색어/검증어 뽑기
  news_search.py               # 네이버 뉴스 검색
  crawler.py                   # 본문 크롤링
  relevance_filter.py          # 동음이의어/무관 기사 제거
  deduplication.py             # 중복 기사 제거
  vector_store.py              # FAISS 구축 + 유사도 기반 검색
  rag_chain.py                 # 답변 생성
```

## 시작하기

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env.example.txt .env   # 열어서 키 채우기
python main.py
```

`.env`에 필요한 건 3가지: `OPENAI_API_KEY`, `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`.

네이버 키는 [console.ncloud.com](https://console.ncloud.com)에서 발급받으세요. 예전 개발자센터
(`developers.naver.com`) 방식이랑 인증 헤더가 달라서 헷갈리기 쉬운데, 지금 코드는 새 방식(NCP API
Gateway)에 맞춰져 있습니다.

## 설정값 (config.py / .env)

| 변수 | 기본값 | 뭐 하는 애인지 |
|---|---|---|
| `RELEVANCE_SIMILARITY_THRESHOLD` | 0.30 | 이거보다 유사도 낮으면 출처로 안 씀 |
| `TOP_K_ARTICLES` | 5 | 출처 최대 개수 (고정 아니고 상한) |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.90 | 이거보다 비슷하면 같은 기사로 취급 |
| `MAX_CRAWL_ARTICLES` | 20 | 검색할 때 최대 몇 건 가져올지 |
| `CRAWL_TIMEOUT_SECONDS` | 8 | 크롤링 타임아웃 |

## 하다가 겪은 삽질들

- **`httpx proxies` 에러** → `openai`랑 최신 `httpx` 버전 충돌. `httpx==0.27.2`로 고정해둠.
- **네이버 API 401** → 옛날 방식(Client ID/Secret + openapi.naver.com)이랑 요즘 방식(NCP, X-NCP-APIGW-*)이 완전 다름. 지금은 새 방식으로 맞춰놓음.
- **콘솔에 한글 입력이 깨짐** (`서울ㅇ뎌서울여대ㄷ` 이런 식으로) → PyCharm 콘솔 + 한글 IME 타이밍 문제. UTF-8 강제 고정 + 깨진 입력 감지해서 재입력받게 처리함. 그래도 안 되면 `python main.py "질문"`처럼 인자로 바로 넘기는 게 제일 확실함.
- **검색어가 너무 뭉뚱그려져서 엉뚱한 기사만 잡힘** → LLM이 "AI 기업" 같은 광범위한 키워드를 같이 뽑아버려서 검색 예산을 낭비했던 게 원인. 고유명사는 그대로 유지, 범용 단어는 키워드로 못 쓰게 프롬프트 수정.

## 나중에 하고 싶은 것

- LangGraph로 멀티에이전트 (검색용/검증용 에이전트 분리)
- 낚시성 기사, 신뢰도 낮은 도메인 걸러내는 팩트체크 필터
- 세션 중에 주제 바꾸는 기능 (지금은 프로그램 재시작해야 함)
