# Threat model (Owner: Tin) - Report Part 6

Scope: the Seminar Critique Agent as it runs today. Update this file when the system changes.
D2 does not require running injection tests. Attack tests go into the final report.

## 1. Assets to protect

| Asset | Why it matters |
|---|---|
| API key (GEMINI_API_KEY) | Cost and abuse if leaked |
| Knowledge graph and paper corpus | Wrong or poisoned data gives wrong critiques |
| Quality of the critique output | Students rely on it |
| Repository and benchmark data | Evidence for grading and reproducibility |

## 2. Entry points for untrusted content

| # | Entry point | Example of untrusted content |
|---|---|---|
| 1 | Seminar material given to the Summarizer | A paper or slide that contains "ignore previous instructions" |
| 2 | Retrieved chunks from GraphRAG | A poisoned paper added to the corpus |
| 3 | Tool results (benchmark lookup) | A tampered benchmark file |
| 4 | (add more if the system grows) | |

## 3. Attacker capabilities (assumptions)

- Can control: text of a paper or slide that a student feeds in, content of a file added to the corpus.
- Cannot control: our source code, our API key, the model weights.

## 4. Mitigations per entry point

| # | Entry point | Attack scenario | Planned mitigation | Status (done / planned) |
|---|---|---|---|---|
| 1 | Seminar material | Instructions hidden in the paper steer the Critic | Wrap as data with delimiters, system prompt states that quoted text is never instructions | planned |
| 2 | Retrieved chunks | Poisoned chunk changes a conclusion | Only ingest from a reviewed paper list, show the source with every chunk | planned |
| 3 | Tool results | Tampered benchmark number | Read-only local file, value always shown with its source | planned |

Link to code: `seminar_critique/tools/__init__.py` (allowlist) and `seminar_critique/security/sanitize.py`.
