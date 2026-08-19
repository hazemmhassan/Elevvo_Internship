from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app.py"


def test_streamlit_app_loads_without_downloading_model():
    assert APP_PATH.exists(), "Streamlit app has not been implemented"

    app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Extractive Question Answering"
    assert len(app.text_area) == 2
    assert app.text_area[0].label == "Context passage"
    assert app.text_area[1].label == "Question"
    assert app.button[0].label == "Extract answer"
