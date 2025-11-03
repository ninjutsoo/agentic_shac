import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agentic.prompts import get_prompt, format_for_llama, parse_model_output, letter_to_label


def test_get_prompt_and_format_for_llama():
    prompt = get_prompt("status_v1", note="abc", trigger="IVDU")
    assert "You classify Drug StatusTime" in prompt["system"]
    assert "Drug trigger:" in prompt["user"]

    formatted = format_for_llama(prompt["system"], prompt["user"])
    assert "<|start_header_id|>system" in formatted
    assert "<|start_header_id|>user" in formatted


def test_parse_model_output_prefers_last_parenthesized():
    out = "(a) blah ... (c) final"
    letter = parse_model_output(out)
    assert letter == "c"
    assert letter_to_label(letter) == "past"


def test_parse_model_output_fallback_last_letter():
    out = "answer is b"
    letter = parse_model_output(out)
    assert letter == "b"
    assert letter_to_label(letter) == "current"
