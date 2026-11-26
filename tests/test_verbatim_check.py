"""Tests for verbatim storyboard coverage detection."""

from src.apps.comic_gen.verbatim_check import coverage_ratio, missing_blocks


def test_paraphrased_line_counts_as_covered():
    text = "韩立睁大眼睛望着屋顶，心里想着白天发生的事情"
    frames = [{"action_summary": "韩立睁大双眼望着屋顶，心里思索白天发生的事"}]
    assert missing_blocks(text, frames) == []
    assert coverage_ratio(text, frames) == 1.0


def test_truly_missing_line_is_detected():
    text = "韩立说我要去七玄门\n张三说好的我等你"
    frames = [{"action_summary": "韩立说我要去七玄门"}]
    missing = missing_blocks(text, frames)
    assert len(missing) == 1
    assert "张三" in missing[0]


def test_dialogue_field_counts_toward_coverage():
    text = "韩立说我要去七玄门"
    frames = [{"dialogue": "我要去七玄门", "action_summary": "韩立开口"}]
    assert missing_blocks(text, frames) == []
