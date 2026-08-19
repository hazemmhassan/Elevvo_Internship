import pytest


def test_highlight_context_marks_answer_and_escapes_user_html():
    from question_answering.ui import highlight_context

    rendered = highlight_context("<b>Paris</b> is here", 3, 8)

    assert "&lt;b&gt;<mark>Paris</mark>&lt;/b&gt; is here" == rendered


@pytest.mark.parametrize(("start", "end"), [(-1, 2), (2, 2), (3, 20)])
def test_highlight_context_rejects_invalid_span(start, end):
    from question_answering.ui import highlight_context

    with pytest.raises(ValueError, match="invalid highlight span"):
        highlight_context("short", start, end)
