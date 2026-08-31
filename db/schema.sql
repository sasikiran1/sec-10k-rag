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
