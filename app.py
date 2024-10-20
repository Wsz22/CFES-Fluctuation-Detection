import streamlit as st
from streamlit_navigation_bar import st_navbar
from pages import playground as pg
from pages import home
from pages import tutorials

# 设置页面布局为宽屏,设置页面标题为"CFES signal wave detection",设置页面图标为火箭,设置初始侧边栏状态为折叠
st.set_page_config(page_title="CFES signal wave detection",
                       layout="wide",
                       page_icon=":rocket:", 
                       initial_sidebar_state='collapsed',
)
# 创建一个导航栏
pages = ["Home", "Playground", "Tutorials", "Development", "Download"]
styles = {
    "nav": {
        "background-color": "rgb(20,140,253)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
    }
options = {
    "show_menu": True,
    "show_sidebar": True,
    }
page = st_navbar(pages,styles=styles,options=options)
if page == "Home":
    home.home()
elif page == "Playground":
    pg.palyground()
elif page == "Tutorials":
    tutorials.tutorials()