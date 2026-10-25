"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --temperature 0.2 "LoRAとは何ですか？"
    uv run python -m app.ask --chat
    AIE_MODEL=gemini-3.5-flash uv run python -m app.ask --chat

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from time import perf_counter

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _generation_config(temperature: float | None) -> types.GenerateContentConfig | None:
    if temperature is None:
        return None
    return types.GenerateContentConfig(temperature=temperature)


def _print_metrics(response: types.GenerateContentResponse, elapsed: float) -> None:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    input_display = input_tokens if input_tokens is not None else "不明"
    output_display = output_tokens if output_tokens is not None else "不明"
    print(
        f"[計測] 応答時間: {elapsed:.2f}秒, "
        f"入力: {input_display} tokens, 出力: {output_display} tokens"
    )


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    config = _generation_config(temperature)
    started_at = perf_counter()
    response = client.models.generate_content(
        model=MODEL, contents=question, config=config
    )
    _print_metrics(response, perf_counter() - started_at)
    return response.text


def chat(temperature: float | None = None, first_question: str | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    conversation = client.chats.create(
        model=MODEL, config=_generation_config(temperature)
    )
    question = first_question

    print("会話を開始します。終了するには :q または Ctrl-D を入力してください。")
    while True:
        if question is None:
            try:
                question = input("あなた> ")
            except EOFError:
                print()
                return

        if question.strip().lower() in {":q", "quit", "exit", "終了"}:
            return
        if not question.strip():
            question = None
            continue

        started_at = perf_counter()
        response = conversation.send_message(question)
        elapsed = perf_counter() - started_at
        print(f"あなた> {question}")
        print(f"Gemini> {response.text}")
        _print_metrics(response, elapsed)
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument(
        "question",
        nargs="?",
        help="質問文（省略時は通常モードで既定の質問を使用）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="会話履歴を保って対話する（質問文を指定すると最初の質問になる）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Gemini API に渡す temperature（通常 0.0〜2.0）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat(temperature=args.temperature, first_question=args.question)
        else:
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question, temperature=args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
