set -e
set -u

MAD_PATH=$(cd "$(dirname "$0")" && pwd)

python3 "$MAD_PATH/code/debate4tran.py" \
    -i "$MAD_PATH/data/CommonMT/input.example.txt" \
    -o "$MAD_PATH/data/CommonMT/output" \
    -lp zh-en \
    --gemini-api-key "${GEMINI_API_KEY:?Set GEMINI_API_KEY first}" \
