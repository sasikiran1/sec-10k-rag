-- Enable pgvector. We don't use vector columns yet (that's session 3), but the
-- extension lives with the database, so turn it on now.
CREATE EXTENSION IF NOT EXISTS vector;

-- Every call to an LLM gets one row here. On the free tier cost_usd is 0, but we
-- still record token counts and latency: those are the numbers the eval harness
-- reasons about later (how expensive is this retrieval strategy, how slow).
CREATE TABLE IF NOT EXISTS llm_calls (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at        timestamptz NOT NULL DEFAULT now(),
    provider          text        NOT NULL,           -- 'groq' | 'gemini'
    model             text        NOT NULL,
    tag               text,                           -- free-text: what this call was for
    prompt_tokens     integer     NOT NULL,
    completion_tokens integer     NOT NULL,
    total_tokens      integer     NOT NULL,
    cost_usd          numeric(12, 6) NOT NULL DEFAULT 0,
    latency_ms        integer     NOT NULL,
    temperature       real        NOT NULL,
    response_id       text                            -- provider's id for the response, for tracing
);

CREATE INDEX IF NOT EXISTS llm_calls_created_at_idx ON llm_calls (created_at);
CREATE INDEX IF NOT EXISTS llm_calls_tag_idx        ON llm_calls (tag);

-- A chunk of source text plus its embedding. `source` + `ord` identify a chunk
-- within a document; the filing-metadata columns are NULL for ad-hoc test data
-- and filled in by sec10k.ingest for real 10-Ks.
CREATE TABLE IF NOT EXISTS chunks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source      text        NOT NULL,         -- accession number, or a label for ad-hoc data
    ord         integer     NOT NULL,         -- position of this chunk within the document
    text        text        NOT NULL,
    embedding   vector(384) NOT NULL,         -- must match sec10k.embeddings.EMBEDDING_DIM
    company     text,
    cik         integer,
    accession   text,
    fiscal_year integer,
    section     text,                         -- "Item 7" etc.
    kind        text,                         -- "prose" | "table"
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- pgvector does a sequential scan unless there's an ANN index. HNSW with cosine
-- ops means queries must compare with the <=> (cosine distance) operator.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_accession_idx ON chunks (accession);

-- Full-text search vector for hybrid (keyword + vector) retrieval.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_text_tsv_idx ON chunks USING gin (text_tsv);

-- One row per answer() call: ties retrieval (which chunks, what scores) to the
-- generation (answer, refusal, cost). The observability view over the pipeline.
CREATE TABLE IF NOT EXISTS traces (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at   timestamptz NOT NULL DEFAULT now(),
    question     text        NOT NULL,
    company      text,                          -- retrieval scope, if any
    fiscal_year  integer,
    hybrid       boolean     NOT NULL DEFAULT false,
    reranked     boolean     NOT NULL DEFAULT false,
    retrieved    jsonb       NOT NULL,          -- [{chunk_id, score, section, kind}, ...] in final order
    answer       text        NOT NULL,
    refused      boolean     NOT NULL,
    llm_call_id  bigint      REFERENCES llm_calls(id),
    latency_ms   integer     NOT NULL           -- end-to-end: retrieve + rerank + generate
);
CREATE INDEX IF NOT EXISTS traces_created_at_idx ON traces (created_at);
