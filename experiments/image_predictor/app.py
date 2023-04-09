import streamlit as st
from PIL import Image
from fastai.vision.all import *
from service.model import predict, load_learner
from service.utils import inject_custom_css
from service.webapp import WebApp
from pathlib import Path
import pandas as pd
from pprint import pprint

model = load_learner(Path.cwd() / "service/models/fine_tuned_rhino_resnet34.pkl")
streamlit_app = WebApp(model)
streamlit_app.main()

# image_url = streamlit_app.search_images("black rhino")

