import streamlit as st

def apply_theme():

    st.markdown("""
    <style>

    body{
    background:#0E1117;
    color:white;
    }

    .stApp{
    background:linear-gradient(
    135deg,#0f2027,#203a43,#2c5364
    );
    }

    h1,h2,h3{
    color:#00E0FF;
    }

    </style>
    """,unsafe_allow_html=True)