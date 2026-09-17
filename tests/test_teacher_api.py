from analysis import teacher_api


class FakeResponse:
    def __init__(self, body):
        self._body = body
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_parse_env_file(tmp_path):
    env_file = tmp_path / ".secrets.env"
    env_file.write_text(
        "# comment\n"
        "ORD_API_ENDPOINT = 'https://gateway.example'\n"
        'export ORD_API_KEY="secret-value"\n'
        "IGNORED_LINE\n"
        "EXTRA=a=b\n",
        encoding="utf-8",
    )

    parsed = teacher_api._parse_env_file(env_file)

    assert parsed == {
        "ORD_API_ENDPOINT": "https://gateway.example",
        "ORD_API_KEY": "secret-value",
        "EXTRA": "a=b",
    }


def test_luna_payload_and_response_with_mocked_http(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        teacher_api, "_load_key_ring",
        lambda path=None: ("https://gateway.example", ["test-key"]),
    )

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse({
            "model": "gpt-5.6-luna-2026-08-01",
            "choices": [{
                "message": {"content": "OK"},
                "logprobs": {"content": [{"token": "OK"}]},
            }],
            "usage": {"completion_tokens": 1},
        })

    monkeypatch.setattr(teacher_api.requests, "post", fake_post)
    result = teacher_api.chat(
        "gpt-5.6-luna", [{"role": "user", "content": "Say OK"}],
        max_tokens=17, temperature=0.2, logprobs=True,
    )

    assert captured["url"] == (
        "https://gateway.example/openai/v1/chat/completions")
    assert captured["headers"]["api-key"] == "test-key"
    assert captured["json"] == {
        "model": "gpt-5.6-luna",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_completion_tokens": 17,
        "logprobs": True,
        "top_logprobs": 5,
        "temperature": 0.2,
    }
    assert result["text"] == "OK"
    assert result["usage"] == {"completion_tokens": 1}
    assert result["logprobs"]["content"][0]["token"] == "OK"


def test_sonnet_payload_and_response_with_mocked_http(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        teacher_api, "_load_key_ring",
        lambda path=None: ("https://gateway.example", ["test-key"]),
    )

    def fake_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse({
            "model": "claude-sonnet-4-6-20260801",
            "content": [{"type": "text", "text": "OK"}],
            "usage": {"output_tokens": 1},
        })

    monkeypatch.setattr(teacher_api.requests, "post", fake_post)
    result = teacher_api.chat(
        "claude-sonnet-4-6",
        [{"role": "user", "content": "Say OK"}],
        max_tokens=23,
    )

    assert captured["url"] == (
        "https://gateway.example/anthropic/v1/messages")
    assert captured["headers"]["api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"] == {
        "model": "claude-sonnet-4-6",
        "max_tokens": 23,
        "messages": [{"role": "user", "content": "Say OK"}],
    }
    assert result["text"] == "OK"
    assert result["logprobs"] is None
