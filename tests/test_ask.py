from types import SimpleNamespace

import app.ask as ask_module


class FakeModels:
    def __init__(self):
        self.arguments = None

    def generate_content(self, **arguments):
        self.arguments = arguments
        return SimpleNamespace(text="answer", usage_metadata=None)


class FakeChat:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.questions = []

    def send_message(self, question):
        self.questions.append(question)
        return next(self.responses)


def _mock_client(monkeypatch):
    models = FakeModels()
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    def create_client(**arguments):
        assert arguments["api_key"] == "test-key"
        return SimpleNamespace(models=models)

    monkeypatch.setattr(
        ask_module.genai, "Client", create_client
    )
    return models


def test_ask_passes_temperature_in_generation_config(monkeypatch):
    models = _mock_client(monkeypatch)

    assert ask_module.ask("question", temperature=0.2) == "answer"

    assert models.arguments["config"].temperature == 0.2


def test_ask_leaves_temperature_unset_by_default(monkeypatch):
    models = _mock_client(monkeypatch)

    assert ask_module.ask("question") == "answer"

    assert models.arguments["config"] is None


def test_main_passes_cli_temperature_to_ask(monkeypatch, capsys):
    arguments = {}

    def fake_ask(question, *, temperature):
        arguments["question"] = question
        arguments["temperature"] = temperature
        return "answer"

    monkeypatch.setattr(ask_module, "ask", fake_ask)
    monkeypatch.setattr(
        "sys.argv", ["app.ask", "--temperature", "0.4", "LoRAとは何ですか？"]
    )

    ask_module.main()

    assert arguments == {"question": "LoRAとは何ですか？", "temperature": 0.4}
    assert "answer" in capsys.readouterr().out


def test_chat_reuses_conversation_and_reports_response_metrics(monkeypatch, capsys):
    chat = FakeChat(
        [
            SimpleNamespace(
                text="first answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10, candidates_token_count=5
                ),
            ),
            SimpleNamespace(
                text="second answer",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=20, candidates_token_count=7
                ),
            ),
        ]
    )
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    create_arguments = {}

    def create_chat(**arguments):
        create_arguments.update(arguments)
        return chat

    def create_client(**arguments):
        assert arguments["api_key"] == "test-key"
        return SimpleNamespace(chats=SimpleNamespace(create=create_chat))

    monkeypatch.setattr(
        ask_module.genai,
        "Client",
        create_client,
    )
    answers = iter(["最初の質問", "続きの質問", ":q"])

    def read_input(prompt):
        assert prompt == "あなた> "
        return next(answers)

    monkeypatch.setattr("builtins.input", read_input)
    monkeypatch.setattr(
        ask_module, "perf_counter", iter([1.0, 2.5, 3.0, 3.25]).__next__
    )

    ask_module.chat(temperature=0.3)

    assert create_arguments["model"] == ask_module.MODEL
    assert create_arguments["config"].temperature == 0.3
    assert chat.questions == ["最初の質問", "続きの質問"]
    output = capsys.readouterr().out
    assert "first answer" in output
    assert "second answer" in output
    assert "入力: 10 tokens, 出力: 5 tokens" in output
    assert "入力: 20 tokens, 出力: 7 tokens" in output


def test_main_starts_chat_mode(monkeypatch):
    arguments = {}

    def fake_chat(*, temperature, first_question):
        arguments["temperature"] = temperature
        arguments["first_question"] = first_question

    monkeypatch.setattr(ask_module, "chat", fake_chat)
    monkeypatch.setattr(
        "sys.argv", ["app.ask", "--chat", "--temperature", "0.5", "最初の質問"]
    )

    ask_module.main()

    assert arguments == {"temperature": 0.5, "first_question": "最初の質問"}