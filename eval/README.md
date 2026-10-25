# Evaluation

The task files use the shared `id`, `topic`, `input`, `rubric_axes`, and
`key_points` contract. The set contains the original example plus eight
paper-grounded critique prompts across the two topics represented in the
project corpus. They are not copied from the instructor's seminar-topics
handout; replace or validate them against its "Key defense points" and
"Required" lists before treating them as an official course evaluation set.

Run the evaluation and summarize its CSV with:

```text
python -m eval.runner --repeats 1
python -m eval.summarize
```

The runner defaults to `FakeLLM`; its output is for checking the pipeline, not
for reporting model quality. Use `--llm gemini` only when the real client is
available and `GEMINI_API_KEY` is configured. The Gemini client uses
`gemini-3.1-flash-lite` when a config still has its `fake-model` placeholder;
set `GEMINI_MODEL` to override that default. Explicit model names in a config
are passed through unchanged. The runner writes one CSV row per task, config,
and repeat, plus traces under `traces/`.

`key_point_recall` and `rubric_axis_coverage` are deterministic phrase-coverage
proxies: text is Unicode-normalized, case-folded, and matched by word sequence.
They do not establish semantic correctness, and paraphrases may be missed.
Validate evaluation claims with human review and report the sample size.
