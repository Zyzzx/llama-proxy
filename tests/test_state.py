"""Unit tests for the /metrics parser (both llama.cpp metric families)."""
from llm_proxy.state import _parse_busy_metrics_sync as _parse_busy_metrics

NEW_SAMPLE = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 0
# HELP llamacpp:requests_processing Number of requests processing
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
# HELP llamacpp:requests_deferred Number of requests deferred
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 3
# HELP llamacpp:n_busy_slots_per_decode Average number of busy slots per llama_decode() call
# TYPE llamacpp:n_busy_slots_per_decode gauge
llamacpp:n_busy_slots_per_decode 1.5
"""

OLD_SAMPLE = """# HELP llama_slots_busy Number of busy slots
# TYPE llama_slots_busy gauge
llama_slots_busy 4
"""


def test_new_format_sums_processing_and_deferred():
    assert _parse_busy_metrics(NEW_SAMPLE) == 5


def test_new_format_zero_deferred():
    sample = "llamacpp:requests_processing 1\nllamacpp:requests_deferred 0\n"
    assert _parse_busy_metrics(sample) == 1


def test_new_format_processing_only_line():
    # deferred line absent entirely
    assert _parse_busy_metrics("llamacpp:requests_processing 2\n") == 2


def test_old_format_still_supported():
    assert _parse_busy_metrics(OLD_SAMPLE) == 4


def test_no_busy_metric_returns_none():
    assert _parse_busy_metrics("llamacpp:prompt_tokens_total 5\n") is None
    assert _parse_busy_metrics("") is None


def test_json_wrapped_body_tolerated():
    import json
    assert _parse_busy_metrics(json.dumps(OLD_SAMPLE)) == 4


def test_help_lines_ignored():
    # '# HELP llama_slots_busy ...' must not be parsed as a value
    assert _parse_busy_metrics(OLD_SAMPLE) == 4
    assert _parse_busy_metrics(
        "# HELP llamacpp:requests_processing Number of requests processing\n"
        "llamacpp:requests_processing 0\n"
    ) == 0
