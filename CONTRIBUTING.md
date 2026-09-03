# Contributing to Filing Analyst

Thanks for considering a contribution.

Filing Analyst is an evaluation-driven RAG project. Changes that affect retrieval, ranking, chunking, prompting, or judging should include evidence showing whether the change improves or degrades the relevant evaluation metrics.

## Good contributions

Examples include:

- retrieval improvements
- true BM25 or alternative lexical baselines
- embedding/reranker comparisons
- additional hand-verified golden questions
- independent human annotation
- improved table parsing
- numerical reasoning
- evaluation methodology improvements
- bug fixes
- tests
- documentation and reproducibility improvements

## Development setup

Follow the setup instructions in `README.md`, then run:

```bash
pytest -m "not live"
```

before opening a pull request.

## Pull requests

Please keep pull requests focused.

For behavior-changing PRs, include:

1. the problem being addressed
2. the proposed change
3. tests added or updated
4. evaluation impact, when applicable
5. known limitations or regressions

If an experiment makes performance worse but reveals something useful, that is still valuable. Please report the negative result rather than hiding it.

## Evaluation contributions

When adding or changing golden questions:

- verify the answer against the original SEC filing
- record the filing/company/fiscal year
- specify the evidence required for retrieval evaluation
- avoid ambiguous questions
- distinguish intentionally unanswerable/refusal examples clearly

## Code quality

Please:

- keep responsibilities separated by module
- add tests for new behavior
- avoid committing secrets or API keys
- preserve deterministic behavior where practical
- document non-obvious design decisions

## Security

Do not open a public issue containing credentials or other sensitive information.
