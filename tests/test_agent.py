"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import sys

import pytest
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, main, run_with_approval

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定
    assert "save_learning_log" in offered

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert isinstance(result.output, str)
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_normal_question_does_not_request_log_approval(tmp_path, monkeypatch):
    log_path = tmp_path / "learning_log" / "learning_log.md"
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)
    model = TestModel(call_tools=["search_glossary"])
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("unexpected approval prompt"))

    with agent.override(model=model):
        result = run_with_approval(agent, "LoRAとは？", model)

    assert isinstance(result.output, str)
    assert not log_path.exists()


def test_learning_log_is_saved_only_after_approval(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "learning_log" / "learning_log.md"
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)
    model = TestModel(call_tools=["save_learning_log"])

    def approve(_prompt):
        model.call_tools = []
        return "y"

    monkeypatch.setattr("builtins.input", approve)
    with agent.override(model=model):
        result = run_with_approval(agent, "学習記録を保存して", model)

    assert isinstance(result.output, str)
    assert log_path.exists()
    assert "学習記録" in log_path.read_text(encoding="utf-8")
    assert "保存予定の内容" in capsys.readouterr().out


def test_learning_log_is_not_saved_when_rejected(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "learning_log" / "learning_log.md"
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)
    model = TestModel(call_tools=["save_learning_log"])

    def reject(_prompt):
        model.call_tools = []
        return "n"

    monkeypatch.setattr("builtins.input", reject)
    with agent.override(model=model):
        result = run_with_approval(agent, "学習記録を保存して", model)

    assert isinstance(result.output, str)
    assert "保存を拒否しました" in capsys.readouterr().out
    assert not log_path.exists()


def test_cli_trace_includes_approved_log_call(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "learning_log" / "learning_log.md"
    monkeypatch.setattr(tools, "LEARNING_LOG_PATH", log_path)
    model = TestModel(call_tools=["save_learning_log"])
    monkeypatch.setattr("app.agent.build_model", lambda: model)
    monkeypatch.setattr(sys, "argv", ["app.agent", "--trace", "学習記録を保存して"])

    def approve(_prompt):
        model.call_tools = []
        return "y"

    monkeypatch.setattr("builtins.input", approve)
    main()

    output = capsys.readouterr().out
    assert "[tool call]   save_learning_log(" in output
    assert "[tool return] save_learning_log ->" in output
    assert log_path.exists()


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"
