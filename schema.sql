-- QuPicT DB 테이블 (PostgreSQL + pgvector)
-- 실행: python run.py init-db   (여러 번 실행해도 안전: IF NOT EXISTS)
--
-- 설계 원칙
--   - 기사(articles)는 link 기준으로 한 번만 저장한다. 여러 topic에 걸린 기사는 article_topics로 연결한다.
--   - topic에는 기간을 저장하지 않는다. 기간은 질문 시점 기준으로 articles.pub_date(발행일)로 거른다.
--     같은 topic을 다시 쓰면 이미 있는 기사는 그대로 쓰고, DB에 없는 기사만 새로 수집한다.
--   - 중복 여부는 article_topics.duplicate_of 하나로만 표시한다 (값이 있으면 중복, status는 비움).
--   - 임베딩 차원 1536 = text-embedding-3-small. 모델을 바꾸면 vector(N)과 인덱스를 함께 바꿔야 한다.

CREATE EXTENSION IF NOT EXISTS vector;

-- [0단계] 검색어별 의미 후보. 같은 검색어면 LLM을 다시 부르지 않고 재사용한다.
CREATE TABLE IF NOT EXISTS query_senses (
    id             BIGSERIAL PRIMARY KEY,
    query          TEXT        NOT NULL,
    rank           SMALLINT,                         -- LLM 후보 순서 (1 = 가장 흔한 의미). 직접 입력이면 NULL
    name           TEXT        NOT NULL,
    definition     TEXT        NOT NULL DEFAULT '',
    aliases        TEXT[]      NOT NULL DEFAULT '{}',
    related_terms  TEXT[]      NOT NULL DEFAULT '{}',
    is_custom      BOOLEAN     NOT NULL DEFAULT FALSE,
    model          TEXT        NOT NULL DEFAULT '',  -- 후보를 만든 LLM (재현성)
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (query, name)
);

-- 수집 대상 = 검색어 + 고른 의미. 의미 내용은 그 시점 값을 그대로 복사해 둔다
-- (targets.yaml에서 별칭·정의를 사람이 고쳐 쓰는 경우가 있어서 query_senses와 다를 수 있음).
CREATE TABLE IF NOT EXISTS topics (
    id                 BIGSERIAL PRIMARY KEY,
    query              TEXT        NOT NULL,
    sense_id           BIGINT      REFERENCES query_senses(id) ON DELETE SET NULL,
    sense_name         TEXT        NOT NULL,
    definition         TEXT        NOT NULL DEFAULT '',
    aliases            TEXT[]      NOT NULL DEFAULT '{}',
    related_terms      TEXT[]      NOT NULL DEFAULT '{}',
    is_custom          BOOLEAN     NOT NULL DEFAULT FALSE,
    excluded_terms     TEXT[]      NOT NULL DEFAULT '{}',   -- 2단계 규칙 필터 배제어
    other_senses       JSONB       NOT NULL DEFAULT '[]',   -- 3단계 LLM 검증 대조용 (Sense 목록)
    last_collected_at  TIMESTAMPTZ,                          -- 마지막 수집 시각
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (query, sense_name)
);

-- [1단계] 기사. link(언론사 원문 주소)가 기준.
CREATE TABLE IF NOT EXISTS articles (
    id               BIGSERIAL PRIMARY KEY,
    link             TEXT        NOT NULL UNIQUE,
    naver_link       TEXT,
    title            TEXT        NOT NULL,
    press            TEXT,
    pub_date         TIMESTAMPTZ,                -- 발행일: 기간 조건은 이 값으로 건다
    description      TEXT        NOT NULL DEFAULT '',
    content          TEXT        NOT NULL DEFAULT '',
    collected_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding        vector(1536),               -- 기사 전체 임베딩 (2·3단계 중복 제거·관련성 필터용)
    embedding_model  TEXT
);
CREATE INDEX IF NOT EXISTS articles_pub_date_idx ON articles (pub_date);

-- [1단계 수집 / 4단계 설명] 기사 속 이미지
CREATE TABLE IF NOT EXISTS article_images (
    id               BIGSERIAL PRIMARY KEY,
    article_id       BIGINT      NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    position         SMALLINT    NOT NULL DEFAULT 0,  -- 기사 안 순서
    url              TEXT        NOT NULL,
    caption          TEXT        NOT NULL DEFAULT '', -- 기사에 원래 달린 사진 설명
    llm_description  TEXT,                            -- 4단계: LLM이 만든 이미지 설명
    described_model  TEXT,
    described_at     TIMESTAMPTZ,
    UNIQUE (article_id, url)
);

-- [2·3단계] topic별 기사 판정. 같은 기사라도 topic마다 판정이 다르다.
CREATE TABLE IF NOT EXISTS article_topics (
    article_id     BIGINT      NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    topic_id       BIGINT      NOT NULL REFERENCES topics(id)   ON DELETE CASCADE,
    search_term    TEXT        NOT NULL DEFAULT '',  -- 이 기사를 찾은 네이버 검색어
    duplicate_of   BIGINT      REFERENCES articles(id) ON DELETE SET NULL,  -- 2단계: 대표 기사 (NULL = 중복 아님)
    status         TEXT        CHECK (status IN ('관련', '무관', '보류')),  -- 시스템 판정 (NULL = 아직 안 함)
    stage          TEXT        CHECK (stage IN ('rule', 'llm')),           -- 어느 단계에서 판정했나
    reason         TEXT,                                                   -- 근거 한 줄
    judged_model   TEXT,
    judged_at      TIMESTAMPTZ,
    label          TEXT        CHECK (label IN ('관련', '무관', '보류')),   -- 사람이 붙인 정답 라벨 (평가용)
    labeled_at     TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, topic_id),
    CHECK (duplicate_of IS NULL OR status IS NULL),     -- 중복이면 판정하지 않는다
    CHECK (duplicate_of IS DISTINCT FROM article_id)
);
CREATE INDEX IF NOT EXISTS article_topics_topic_idx ON article_topics (topic_id, status);

-- [4단계] 청크 + 임베딩. 기사 단위로 저장하고, 검색할 때 article_topics로 topic·판정을 건다.
CREATE TABLE IF NOT EXISTS article_chunks (
    id               BIGSERIAL PRIMARY KEY,
    article_id       BIGINT      NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    chunk_index      INT         NOT NULL,
    source           TEXT        NOT NULL DEFAULT 'body' CHECK (source IN ('body', 'image')),
    image_id         BIGINT      REFERENCES article_images(id) ON DELETE CASCADE,  -- source = 'image'일 때
    content          TEXT        NOT NULL,
    embedding        vector(1536),
    embedding_model  TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS article_chunks_embedding_idx
    ON article_chunks USING hnsw (embedding vector_cosine_ops);
