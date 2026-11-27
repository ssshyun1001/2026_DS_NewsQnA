# 코드 정리 목록 (2026-11-27 확인)

불필요한 파일 · 코드를 정리하기 전에 팀이 같이 보는 체크리스트입니다.
확인 방법: git에 올라간 모든 파일을 대상으로 import · 파일 이름 참조를 검색했고, 깨진 import는 없습니다.
**서비스 코드(`qupict/`, `web.py`, `run.py`, `admin.py`, `web/`)에는 지울 파일이 없습니다.** 정리 대상은 실험 · 예전 코드 쪽입니다.

정리한 뒤에는 꼭 `python tests/run_offline.py`로 확인해 주세요.

---

## 1. 지워도 되는 파일

| 파일 | 이유 | 같이 할 일 |
| --- | --- | --- |
| `old/` 4개 (`_common.py`, `debug_similarity.py`, `eval_filter.py`, `eval_retrieval.py`) | 어디서도 import하지 않음. 다른 코드의 "여기서 옮겼다" 주석에만 이름이 남아 있음. 기능은 `evalset/` · `experiments/`로 옮겨짐 | `evalset/build_labeled_set.py` · `experiments/similarity.py` 주석, `docs/STRUCTURE.md`의 `old/` 설명 정리 |
| `tests/offline/check_old.py` | `old/`만 시험함 | 테스트가 19 → 18묶음 → README · PROGRESS 숫자 수정 |
| `tools/migrate_supabase_to_firestore.py` | Supabase → Firestore 이전용 일회성 도구. 이전은 끝났고 `psycopg`가 requirements에 없어 지금은 실행도 안 됨 | README 구조 · `.dockerignore`의 `tools/` 줄 정리 |
| `experiments/qa_set_hyde_hybrid.csv` | `qa_set.csv`와 **내용이 완전히 같음** (BOM · 줄바꿈만 다름) | `exp_b_hyde_hybrid.py`가 `qa_set.csv`를 읽도록 바꾼 뒤 삭제 |
| `evalset/results/exp_a_filtering_20261111_0009.csv/json` | 3개 검색어만 라벨링했을 때의 중간 결과. 최종 결과는 `…20261113_0456` | 없음 |

## 2. 정해야 할 파일

| 파일 | 상황 | 제안 |
| --- | --- | --- |
| `evalset/results/verdict_reasons_20261112.csv` | 어디서도 읽거나 언급하지 않음 (판정 근거 분석용으로 보임) | 쓸 일 없으면 삭제 |
| `evalset/results/llm_cache.json` | 실험 A를 다시 돌릴 때 LLM 비용을 0으로 만드는 캐시 | **유지** |
| `experiments/results/exp_b_retrieval_20261116_0026` · `_1624` | PROGRESS 실험 B 표 / README Reranker 수치의 출처 | **유지** |
| `experiments/exp_b_hyde_hybrid.py` | 최종 검색 방식을 정한 실험. 다만 BM25를 직접 구현(`BM25Index`, `tokens`, `fuse`)해서 `qupict/bm25.py`와 겹치고, `exp_b_retrieval.py`에도 `hybrid` 방식이 이미 있음 | 유지하되 `qupict/bm25.py`를 쓰도록 정리 검토 |
| `experiments/benchmark.py` | 11-10 속도 측정 도구. `qupict/concurrency.py` 주석이 측정 근거로 인용 | 더 쓸 일 없으면 삭제 (주석도 같이) |
| `experiments/check_dedup.py` | 중복 제거 확인용 단독 도구 | 유지 |

## 3. 파일 안에서 정리할 코드

| 위치 | 내용 |
| --- | --- |
| `web.py:542` | `_open_from_db`의 `when` 변수를 만들기만 하고 안 씀 (11-19 문구 수정 때 남음). 바로 아래 `log(f"...")`의 불필요한 `f`도 같이 |
| `web/app.js:234` | 상태 줄에 아직 `DB에 저장된 기사 N건으로 먼저 답합니다.` — 답변 시점 줄 · 로그처럼 "N건"을 뺄지 결정 |
| `qupict/db.py:105` `close_pool()` | 예전 PostgreSQL 호환용 빈 함수, 호출하는 곳 없음 |
| `qupict/ocr.py:44` `uses_gpu()` | 호출하는 곳 없음 |
| `qupict/db.py` `save_image_description` · `save_chunks` | 서비스는 `save_image_descriptions` · `save_chunks_many`만 씀. 테스트만 부르므로 테스트를 바꾼 뒤 삭제 |
| `qupict/concurrency.py` 맨 위 주석 | 아직 Supabase 기준 설명 → Firestore 기준으로 수정 (`DB_PARALLEL` 값도 다시 보기) |
| `qupict/retrieval.py` `retrieve_relevant_articles` | 메모리 모드(`run.py --memory`, `web.py`)의 기본값이 아직 HyDE + Multi-Query 켜짐. 서비스는 Hybrid만 쓰기로 했으므로 맞출지 결정 |
| `requirements.txt` `pyyaml` | import하는 곳 없음 (`targets.yaml` · `qa_set.yaml` 삭제 뒤 남음) |

## 4. 설정

| 위치 | 내용 |
| --- | --- |
| `.dockerignore` | `.venv/`만 빼고 있는데 로컬 가상환경 폴더는 `venv/` → 로컬에서 `docker build`하면 `venv/`가 이미지에 들어감. `venv/` 한 줄 추가 (Cloud Build는 git 기준이라 배포에는 영향 없음) |

---

## 체크리스트

- [x] 1. 지워도 되는 파일 삭제 + 같이 할 일
- [ ] 2. 정해야 할 파일 결정
- [ ] 3. 코드 정리
- [ ] 4. `.dockerignore`에 `venv/` 추가
- [ ] `python tests/run_offline.py` 통과 확인
- [ ] README · PROGRESS의 테스트 수 · 폴더 구조 갱신
