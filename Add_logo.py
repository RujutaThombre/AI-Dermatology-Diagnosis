import streamlit as st

def add_logo(logo_url, width=100):
    """
    Adds a logo to the Streamlit sidebar.

    Parameters:
    logo_url (str): The URL or path of the logo image file.
    width (int): The width of the logo image. Default is 100.
    """
    st.sidebar.image(logo_url, width=width)
