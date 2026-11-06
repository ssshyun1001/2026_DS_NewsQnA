-- QuPicT DB 테이블 (PostgreSQL + pgvector)
-- TODO: 컬럼 확정 후 CREATE TABLE 작성

CREATE EXTENSION IF NOT EXISTS vector;

-- query_senses    검색어별 의미 후보 (같은 검색어면 재사용)
-- topics          수집 대상 + 선택한 의미 + 기간
-- articles        제목, 링크, 게시일, 언론사, 본문, 수집 일시
-- article_images  이미지 원본, 캡션, LLM 설명
-- article_topics  기사별 판정(관련/무관/보류), 근거 한 줄, 정답 라벨
-- article_chunks  청크 텍스트 + 임베딩 벡터
