import json
import sys
import types
from unittest.mock import MagicMock, patch

import agents


def _fake_openai_module(reply="provider response"):
    module = types.ModuleType("openai")
    sdk_client = MagicMock()
    sdk_client.chat.completions.create.return_value = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=reply))]
    )
    module.OpenAI = MagicMock(return_value=sdk_client)
    return module, sdk_client


def test_deepseek_uses_official_openai_compatible_endpoint_and_current_default():
    fake_openai, sdk_client = _fake_openai_module()
    with patch.dict(sys.modules, {"openai": fake_openai}), \
         patch.dict("os.environ", {}, clear=False):
        client = agents.create_llm_client("deepseek", "sk-deepseek", "")
        result = client.complete("system", [{"role": "user", "content": "hello"}])

    fake_openai.OpenAI.assert_called_once_with(
        api_key="sk-deepseek",
        base_url="https://api.deepseek.com",
    )
    assert sdk_client.chat.completions.create.call_args.kwargs["model"] == "deepseek-v4-flash"
    assert result == "provider response"


def test_qwen_uses_dashscope_endpoint_and_supports_region_override():
    fake_openai, sdk_client = _fake_openai_module()
    singapore_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    with patch.dict(sys.modules, {"openai": fake_openai}), \
         patch.dict("os.environ", {"QWEN_BASE_URL": singapore_url}):
        client = agents.create_llm_client("qwen", "sk-qwen", "")
        client.complete("system", [{"role": "user", "content": "hello"}])

    fake_openai.OpenAI.assert_called_once_with(api_key="sk-qwen", base_url=singapore_url)
    assert sdk_client.chat.completions.create.call_args.kwargs["model"] == "qwen3.7-plus"


def test_provider_key_validation_and_registry():
    assert {"deepseek", "qwen"}.issubset(agents.VALID_PROVIDERS)
    assert agents.validate_api_key_for_provider("deepseek", "bad-key")
    assert agents.validate_api_key_for_provider("qwen", "bad-key")
    assert agents.validate_api_key_for_provider("deepseek", "sk-valid") == ""
    assert agents.validate_api_key_for_provider("qwen", "sk-valid") == ""


def test_openrouter_retries_malformed_api_json_without_changing_messages():
    fake_openai, sdk_client = _fake_openai_module()
    malformed = json.JSONDecodeError("Expecting value", "", 0)
    successful_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content="recovered response")
            )
        ]
    )
    sdk_client.chat.completions.create.side_effect = [
        malformed,
        malformed,
        successful_response,
    ]
    messages = [{"role": "user", "content": "hello"}]

    with patch.dict(sys.modules, {"openai": fake_openai}), \
         patch("agents.time.sleep") as sleep:
        client = agents.create_llm_client(
            "openrouter",
            "sk-openrouter",
            "deepseek/deepseek-chat-v3.1",
        )
        result = client.complete("system", messages)

    assert result == "recovered response"
    assert sdk_client.chat.completions.create.call_count == 3
    assert sleep.call_count == 2
    sent_messages = [
        call.kwargs["messages"]
        for call in sdk_client.chat.completions.create.call_args_list
    ]
    assert sent_messages == [
        [{"role": "system", "content": "system"}, *messages],
        [{"role": "system", "content": "system"}, *messages],
        [{"role": "system", "content": "system"}, *messages],
    ]


def test_openrouter_does_not_retry_non_parse_errors():
    fake_openai, sdk_client = _fake_openai_module()
    sdk_client.chat.completions.create.side_effect = RuntimeError("invalid model")

    with patch.dict(sys.modules, {"openai": fake_openai}), \
         patch("agents.time.sleep") as sleep:
        client = agents.create_llm_client(
            "openrouter",
            "sk-openrouter",
            "deepseek/deepseek-chat-v3.1",
        )
        try:
            client.complete("system", [{"role": "user", "content": "hello"}])
        except RuntimeError as exc:
            assert str(exc) == "invalid model"
        else:
            raise AssertionError("Expected the non-retryable error to propagate.")

    assert sdk_client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


def test_openrouter_stops_after_bounded_malformed_json_retries():
    fake_openai, sdk_client = _fake_openai_module()
    errors = [
        json.JSONDecodeError("Expecting value", "", attempt)
        for attempt in range(4)
    ]
    sdk_client.chat.completions.create.side_effect = errors

    with patch.dict(sys.modules, {"openai": fake_openai}), \
         patch("agents.time.sleep") as sleep:
        client = agents.create_llm_client(
            "openrouter",
            "sk-openrouter",
            "deepseek/deepseek-chat-v3.1",
        )
        try:
            client.complete("system", [{"role": "user", "content": "hello"}])
        except json.JSONDecodeError as exc:
            assert exc is errors[-1]
        else:
            raise AssertionError("Expected malformed JSON retries to be exhausted.")

    assert sdk_client.chat.completions.create.call_count == 4
    assert sleep.call_count == 3


def test_agent_discards_invalid_content_json_and_retries_identical_prompt():
    agent = agents.Agent.__new__(agents.Agent)
    agent.name = "Test Agent"
    agent.persona = "system"
    agent.messages = [{"role": "user", "content": "paper"}]
    agent.client = MagicMock()
    agent.client.complete.side_effect = [
        "not json",
        "[1, 2, 3]",
        '{"decision": "accept"}',
    ]

    with patch("agents.time.sleep") as sleep:
        result = agent.call("same prompt", required_json=True)

    assert result == '{"decision": "accept"}'
    assert agent.client.complete.call_count == 3
    assert sleep.call_count == 2
    assert agent.client.complete.call_args_list == [
        (("system", [
            {"role": "user", "content": "paper"},
            {"role": "user", "content": "same prompt"},
        ]),) for _ in range(3)
    ]
    assert agent.messages[-1] == {
        "role": "assistant", "content": '{"decision": "accept"}'
    }


def test_agent_gives_up_invalid_json_without_saving_invalid_return():
    agent = agents.Agent.__new__(agents.Agent)
    agent.name = "Test Agent"
    agent.persona = "system"
    agent.messages = []
    agent.client = MagicMock()
    agent.client.complete.return_value = "invalid"

    with patch("agents.time.sleep"):
        result = agent.call("same prompt", required_json=True)

    assert result is None
    assert agent.client.complete.call_count == 4
    assert agent.messages == [{"role": "user", "content": "same prompt"}]
