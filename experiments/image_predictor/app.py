from service.model import load_learner
from service.webapp import WebApp
from pathlib import Path

model = load_learner(Path.cwd() / "service/models/fine_tuned_rhino_resnet34.pkl")
streamlit_app = WebApp(model)
streamlit_app.main()
