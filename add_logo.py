import streamlit as st
import base64

def add_sidebar_logo():
    with open("./img/alex.png", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

        st.sidebar.markdown(
            f"""
            <div style="display:table;margin-top:-20%;margin-left:-3%;">
                <img src="data:image/png;base64,{data}" width="170" height="60">
            </div>
            """,
            unsafe_allow_html=True)

add_sidebar_logo()
