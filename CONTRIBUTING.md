# Hướng dẫn làm việc nhóm (G7)

Mục tiêu: 4 người cùng viết code mà không dẫm chân nhau, và lịch sử commit phản ánh đúng đóng góp của từng người (thầy đối chiếu với bảng đóng góp ở Phần 8 của report).

## 1. Ai phụ trách thư mục nào

| Người | Thư mục sở hữu | Phần report |
|---|---|---|
| Vân | `seminar_critique/llm/`, `agents/`, `orchestrator.py`, `configs.py` | 1, 2, 7 |
| Minh | `seminar_critique/retrieval/`, `third_party/`, README | 4 |
| Tín | `seminar_critique/tools/`, `security/` | 3, 6 |
| Tâm | `eval/` | 5 |

Phần 8 do mỗi người tự điền dòng của mình. Muốn sửa thư mục của người khác thì nhắn người đó trước, hoặc mở PR và gắn tên họ vào review.

## 2. Bốn giao diện đã chốt (không đổi chữ ký nếu chưa bàn)

| Giao diện | Chủ | Vị trí |
|---|---|---|
| `retrieve(query, topic, k) -> list[Chunk]` | Minh | `seminar_critique/retrieval/retriever.py` |
| `lookup_benchmark(name, metric) -> dict` (chỉ đọc) | Tín | `seminar_critique/tools/benchmark_lookup.py` |
| `run_pipeline(task, config, llm=None) -> Result` | Vân | `seminar_critique/orchestrator.py` |
| File task JSON: `id`, `topic`, `input`, `rubric_axes`, `key_points` | Tâm | `eval/tasks/` |

Các kiểu dữ liệu dùng chung nằm trong `seminar_critique/schemas.py`. Muốn đổi file này hoặc đổi chữ ký ở bảng trên, mở PR có tiêu đề `[contract] ...` và cần cả 4 người đồng ý.

Mỗi giao diện hiện là **stub chạy được**. Bạn thay phần ruột, giữ nguyên chữ ký, thì các module khác không bị vỡ. Nhờ `FakeLLM` nên không ai cần API key để phát triển phần của mình.

## 3. Quy trình Git

1. Lấy code mới nhất: `git checkout main && git pull`.
2. Tạo nhánh: `git checkout -b feature/<tên>-<module>`, ví dụ `feature/tin-tools`.
3. Commit nhỏ và thường xuyên, có tiền tố module:
   - `[retrieval] add chunker by section`
   - `[tools] add read-only allowlist test`
   - `[eval] add 5 tasks for reflection topic`
4. Trước khi mở PR: `pytest` phải xanh.
5. Mở Pull Request vào `main`. Vân review và merge. Với PR của Vân thì một bạn khác review.
6. Mỗi người push bằng **tài khoản GitHub của chính mình**. Không để một người đẩy code thay cả nhóm.

Mục tiêu tối thiểu cho D2: mỗi người có ít nhất 5 commit có nội dung thật vào thư mục của mình.

## 4. Chạy thử

```
pip install -r requirements.txt
pytest
python -m seminar_critique.run --task eval/tasks/example_task.json --config C
python -m eval.runner --repeats 3
```

Mặc định dùng `FakeLLM` (không cần key). Để chạy model thật: đặt biến môi trường `GEMINI_API_KEY` rồi thêm `--llm gemini`. Không bao giờ commit key (xem `.env.example`).

## 5. Quy tắc nhỏ

- Số liệu trong report phải tái lập được: lưu kết quả chạy vào `eval/results/*.csv` và commit file này. Thư mục `traces/` không commit, chỉ trích một trace đẹp vào phụ lục.
- Mỗi task trong `eval/tasks/` là một file JSON riêng.
- Code lấy từ MAD để trong `third_party/MAD/`, giữ nguyên LICENSE. Nếu tái dùng đoạn nào thì ghi nguồn trong docstring.
- Dùng công cụ AI thì ghi lại để điền vào AI-use statement (Phần 8): dùng vào việc gì, và bạn đã tự kiểm chứng gì.
