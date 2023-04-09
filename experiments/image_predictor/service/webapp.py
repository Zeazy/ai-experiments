from service.utils import inject_custom_css, render_header
from service.constants import CUSTOM_CSS
from service.model import predict
from duckduckgo_search import ddg_images
from fastai.vision.all import *
from random import choice, randint
import streamlit as st
from PIL import Image
import pandas as pd


st.set_page_config(page_title="Rhino Guessing Game 🦏", page_icon="🦏", layout="wide")


class WebApp:
    """
    Simple Streamlit web app for image classification
    """

    def __init__(self, model, css=CUSTOM_CSS):
        self.rhino_species = ["black", "white", "javan", "sumatran"]
        self.model = model
        self.css = css

    def search_image(self, term, max_results=5):
        """
        Searches duckduckgo for an image of the given term
        """
        print(f"Searching for random images of '{term} rhino...'")
        results = ddg_images(f"{term} rhino", max_results=max_results)
        results = results[randint(0, max_results - 1)]["image"]
        return results

    def display_results(self, pred_class, class_probs):
        st.markdown(f'<p class="prediction">Predicted class: {pred_class}</p>', unsafe_allow_html=True)
        st.markdown('<p class="probabilities">Probabilities for each class:</p>', unsafe_allow_html=True)
        probs_df = pd.DataFrame(class_probs, columns=["Class", "Probability"])
        probs_df = probs_df.sort_values(by="Probability", ascending=False)
        st.table(probs_df)

    def search_and_display_rhino_image(self, species):
        result = self.search_image(f"{species} rhino")
        if result:
            response = requests.get(result, stream=True)
            response.raw.decode_content = True
            img = Image.open(response.raw)
            st.image(img, caption=f"A rhino - guess which species", use_column_width=True)
            return img
        else:
            st.error("No image found for the given species. Try again.")
            return None

    def render_search(self):
        random_rhino = choice(self.rhino_species)
        st.session_state.image = self.search_and_display_rhino_image(random_rhino)

    def render_guess(self, user_guess: str):
        if st.session_state.image:
            model_prediction = predict(self.model, st.session_state.image)[0]
            if user_guess == model_prediction:
                st.success(
                    f"Congratulations! You guessed correctly! The rhino is a {model_prediction.capitalize()} Rhino."
                )
            else:
                st.error(f"Oops! Your guess was incorrect. The rhino is a {model_prediction.capitalize()} Rhino.")
            self.display_results(model_prediction, predict(self.model, st.session_state.image)[1])
        else:
            st.warning("Please search for a rhino image first.")

    def render_guess_form(self):
        with st.form(key="rhino_guess_form"):
            user_guess = st.radio("Select the species:", options=self.rhino_species, key="user_guess")
            submit_button = st.form_submit_button(label="Submit")
        return user_guess, submit_button

    def main(self):
        """
        Main entry point for the app
        """
        render_header()
        inject_custom_css(self.css)

        if "search_clicked" not in st.session_state:
            st.session_state.search_clicked = False

        if "image" not in st.session_state:
            st.session_state.image = None

        if st.button("Search for a rhino"):
            st.session_state.search_clicked = True
            self.render_search()

        if st.session_state.search_clicked:
            st.subheader("What species of rhino is this?")
            user_guess, submit_button = self.render_guess_form()

            if submit_button:
                if st.session_state.image:
                    self.render_guess(user_guess)
                else:
                    st.warning("Please search for a rhino image first.")

        st.markdown("</div>", unsafe_allow_html=True)
