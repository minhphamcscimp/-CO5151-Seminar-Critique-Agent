# Papers for retrieval (public papers only)

Owner: Minh. `ingest.py` reads this table. Keep the 5 columns in this order.

- `id`: short slug, unique, used in chunk ids
- `topic`: topic id that tasks and `retrieve(query, topic, k)` use
- `cite`: label shown to the LLM and in the report (becomes `Chunk.paper`)
- `title`: exact paper title (ingest warns if it is not found on page 1)
- `arxiv`: arXiv id (PDF is downloaded to `data/raw/`, which is not committed)

| id | topic | cite | title | arxiv |
|---|---|---|---|---|
| huang2024 | self-improvement-reflection | Huang et al. 2024 | Large Language Models Cannot Self-Correct Reasoning Yet | 2310.01798 |
| shinn2023 | self-improvement-reflection | Shinn et al. 2023 | Reflexion: Language Agents with Verbal Reinforcement Learning | 2303.11366 |
| madaan2023 | self-improvement-reflection | Madaan et al. 2023 | Self-Refine: Iterative Refinement with Self-Feedback | 2303.17651 |
| gou2024 | self-improvement-reflection | Gou et al. 2024 | CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing | 2305.11738 |
| kamoi2024 | self-improvement-reflection | Kamoi et al. 2024 | When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs | 2406.01297 |
| chen2024 | self-improvement-reflection | Chen et al. 2024 | Teaching Large Language Models to Self-Debug | 2304.05128 |
| yao2023 | tool-use-mcp | Yao et al. 2023 | ReAct: Synergizing Reasoning and Acting in Language Models | 2210.03629 |
| schick2023 | tool-use-mcp | Schick et al. 2023 | Toolformer: Language Models Can Teach Themselves to Use Tools | 2302.04761 |
| patil2023 | tool-use-mcp | Patil et al. 2023 | Gorilla: Large Language Model Connected with Massive APIs | 2305.15334 |
| qin2024 | tool-use-mcp | Qin et al. 2024 | ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs | 2307.16789 |
