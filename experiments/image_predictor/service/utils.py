from service.constants import CUSTOM_CSS
import streamlit as st

def inject_custom_css(css: str = CUSTOM_CSS):
    st.markdown(f"""
    <style>
        {css}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("Rhino Guessing Game 🦏")
    st.subheader("""Generate a random picture of a rhino and see if you can guess the species 
                \n Species include: Black, White, Javan and Sumatran.
                \n This is a simple image classification app - the model was trained on a small dataset of rhino images.""")  
