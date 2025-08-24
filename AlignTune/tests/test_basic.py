from scripts.utils import format_sft, safety_flag, looks_like_refusal
def test_format():
    ex = {"instruction":"Define PEFT","input":"","output":"x"}
    assert "[RESPONSE]" in format_sft(ex)
def test_safety():
    assert safety_flag("I hate this") is True
    assert safety_flag("I like this") is False
def test_refusal():
    assert looks_like_refusal("Sorry, I can't help with that.") is True
