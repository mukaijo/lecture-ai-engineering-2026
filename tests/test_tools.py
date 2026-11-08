"""ツールのテスト（LLMもAPIキーも使わない）。"""

import json

from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} の related に存在しない id: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start タグ"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("エージェント"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "見つかりませんでした" in tools.search_glossary("存在しない用語xyz")


def test_lecture_summary():
    assert "エージェント" in tools.get_lecture_summary(3)
    assert "まだありません" in tools.get_lecture_summary(99)


def test_save_learning_log_appends_markdown_entries(tmp_path, monkeypatch):
    log_path = tmp_path / "learning_log" / "learning_log.md"
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)

    tools.save_learning_log("今日はLoRAの仕組みを学んだ。")
    tools.save_learning_log("次にQLoRAとの違いを学んだ。")

    content = log_path.read_text(encoding="utf-8")
    assert content.count("## 学習記録") == 2
    assert content.count("保存日時:") == 2
    assert "今日はLoRAの仕組みを学んだ。" in content
    assert "次にQLoRAとの違いを学んだ。" in content
    assert "\n\n---\n\n" in content


def test_save_learning_log_preserves_existing_file_content(tmp_path, monkeypatch):
    log_path = tmp_path / "learning_log" / "learning_log.md"
    original = "# 以前の学習記録\n\n既存内容\n"
    log_path.parent.mkdir()
    log_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)

    tools.save_learning_log("新しい学習内容")

    content = log_path.read_text(encoding="utf-8")
    assert content.startswith(original)
    assert "\n\n---\n\n" in content
    assert "新しい学習内容" in content


def test_save_learning_log_does_not_store_credentials(tmp_path, monkeypatch):
    log_path = tmp_path / "learning_log" / "learning_log.md"
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)

    result = tools.save_learning_log("APIキー：secret-value")

    assert "保存しませんでした" in result
    assert not log_path.exists()


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """第1回の既定の質問（app.ask）は、第3回のエージェントが用語集で正しく答えられること。"""
    from app.ask import DEFAULT_QUESTION

    assert "3レベル" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3レベル"))
    assert hits[0]["term"] == "宿題の3レベル"
