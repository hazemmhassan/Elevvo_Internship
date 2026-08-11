from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app.py"


def test_streamlit_app_loads_without_calling_openai():
    assert APP_PATH.exists(), "Streamlit app has not been implemented"

    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Autonomous Text-to-SQL Agent"
    assert len(app.chat_input) == 1
    assert any("read-only" in item.value.lower() for item in app.markdown)
