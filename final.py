import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import altair as alt
import streamlit.components.v1 as components
from datetime import datetime
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re 


# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------
st.set_page_config(
    page_title="Disaster Prediction & Alert Hub",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# INITIALIZE SESSION STATE ATTRIBUTES
# -------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Red"  # default theme

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  # is user logged in?

if "username" not in st.session_state:
    st.session_state.username = None  # store the username

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False  # to check if the user is admin

# -------------------------------------------------------
# INJECT CUSTOM CSS BASED ON THE THEME
# -------------------------------------------------------
def apply_custom_theme():
    # We keep the same color sets from your original code:
    if st.session_state.theme == "Red":
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono&display=swap');
        
        /* Global Settings - Red Theme */
        html, body {
            margin: 0;
            padding: 0;
            background: #330000;
            color: #fefefe;
            font-family: 'Roboto Mono', monospace;
            scroll-behavior: smooth;
            overflow-x: hidden;
        }
        body {
            background: radial-gradient(circle at center, #330000 0%, #660000 60%, #330000 100%) fixed;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #660000; }
        ::-webkit-scrollbar-thumb { background-color: #ff4d4d; border-radius: 10px; }
        
        /* Rolling Alert Marquee */
        .rolling-alert {
            width: 100%;
            background: #660000;
            overflow: hidden;
            position: relative;
            height: 40px;
            display: flex;
            align-items: center;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px #ff4d4d;
            z-index: 9999;
        }
        .rolling-alert span {
            position: absolute;
            white-space: nowrap;
            animation: scroll-left 10s linear infinite;
            color: #fefefe;
            text-shadow: 0 0 5px rgba(255,77,77,0.8), 0 0 10px rgba(255,77,77,0.6);
        }
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        /* Parallax Header */
        .parallax-header {
            position: relative;
            height: 400px;
            background-image: url('https://source.unsplash.com/1600x900/?fire,disaster');
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px rgba(255,77,77,0.4);
        }
        .parallax-header::after {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(135deg, rgba(255,77,77,0.2), rgba(150,0,0,0.5));
            z-index: 1;
        }
        .header-text {
            position: relative;
            z-index: 2;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
        }
        .header-text h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 60px;
            margin: 0;
            letter-spacing: 2px;
            color: #ff4d4d;
            cursor: default;
        }
        .header-text p { font-size: 20px; margin-top: 10px; color: #ffe6e6; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #330000;
            border-right: 2px solid #ff4d4d;
            box-shadow: 0 0 15px rgba(255,77,77,0.5);
            transition: all 0.4s ease-in-out;
        }
        [data-testid="stSidebar"]:hover {
            transform: translateX(3px);
            box-shadow: inset 0 0 10px rgba(255,77,77,0.4);
        }
        
        /* Glassmorphism Cards */
        .card {
            background: rgba(51,0,0,0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,77,77,0.3);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 8px 20px rgba(255,77,77,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            color: #fefefe;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(255,77,77,0.3);
        }
        
        /* Buttons & Inputs */
        .stButton>button {
            background-color: #ff4d4d;
            color: #330000;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: transform 0.2s ease;
            box-shadow: 0 0 8px rgba(255,77,77,0.4);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 12px rgba(255,77,77,0.7);
        }
        input, .stTextInput>div>div>input {
            background: #330000 !important;
            border: 1px solid #ff4d4d !important;
            color: #fefefe !important;
            border-radius: 4px;
            padding: 8px;
        }
        input:focus, .stTextInput>div>div>input:focus {
            outline: none;
            box-shadow: 0 0 5px 2px rgba(255,77,77,0.5);
        }
        .stDownloadButton button {
            background-color: #ff6666;
            color: #330000;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: transform 0.2s ease;
        }
        .stDownloadButton button:hover {
            transform: scale(1.03);
        }
        
        /* Footer & Back-to-Top */
        .footer {
            background: #330000;
            border-top: 2px solid #ff4d4d;
            text-align: center;
            padding: 20px;
            margin: 40px 20px;
            border-radius: 10px;
        }
        .footer p { margin: 0; font-size: 15px; color: #fefefe; }
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #ff4d4d;
            color: #330000;
            border-radius: 50%;
            padding: 12px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10000;
        }
        </style>
        """
    elif st.session_state.theme == "Blue":
        custom_css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono&display=swap');
        /* ... (Code for Blue theme styles) ... */
        </style>
        """
        # Replacing placeholder with the original detailed Blue theme:
        custom_css = custom_css.replace("/* ... (Code for Blue theme styles) ... */", """
        /* Global Settings - Blue Theme */
        html, body {
            margin: 0;
            padding: 0;
            background: #001f3f;
            color: #f0f8ff;
            font-family: 'Roboto Mono', monospace;
            scroll-behavior: smooth;
            overflow-x: hidden;
        }
        body {
            background: radial-gradient(circle at center, #001f3f 0%, #0074D9 60%, #001f3f 100%) fixed;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0074D9; }
        ::-webkit-scrollbar-thumb { background-color: #7FDBFF; border-radius: 10px; }
        
        /* Rolling Alert Marquee */
        .rolling-alert {
            width: 100%;
            background: #0074D9;
            overflow: hidden;
            position: relative;
            height: 40px;
            display: flex;
            align-items: center;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px #7FDBFF;
            z-index: 9999;
        }
        .rolling-alert span {
            position: absolute;
            white-space: nowrap;
            animation: scroll-left 10s linear infinite;
            color: #f0f8ff;
            text-shadow: 0 0 5px rgba(127,219,255,0.8), 0 0 10px rgba(127,219,255,0.6);
        }
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        /* Parallax Header */
        .parallax-header {
            position: relative;
            height: 400px;
            background-image: url('https://source.unsplash.com/1600x900/?ocean,storm');
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px rgba(127,219,255,0.4);
        }
        .parallax-header::after {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(135deg, rgba(127,219,255,0.2), rgba(0,102,204,0.5));
            z-index: 1;
        }
        .header-text {
            position: relative;
            z-index: 2;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
        }
        .header-text h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 60px;
            margin: 0;
            letter-spacing: 2px;
            color: #7FDBFF;
            cursor: default;
        }
        .header-text p { font-size: 20px; margin-top: 10px; color: #d0ebff; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #001f3f;
            border-right: 2px solid #7FDBFF;
            box-shadow: 0 0 15px rgba(127,219,255,0.5);
            transition: all 0.4s ease-in-out;
        }
        [data-testid="stSidebar"]:hover {
            transform: translateX(3px);
            box-shadow: inset 0 0 10px rgba(127,219,255,0.4);
        }
        
        /* Glassmorphism Cards */
        .card {
            background: rgba(0,31,63,0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(127,219,255,0.3);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 8px 20px rgba(127,219,255,0.1);
            transition: transform 0.3s ease;
            color: #f0f8ff;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(127,219,255,0.3);
        }
        
        /* Buttons & Inputs */
        .stButton>button {
            background-color: #7FDBFF;
            color: #001f3f;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: transform 0.2s ease;
            box-shadow: 0 0 8px rgba(127,219,255,0.4);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 12px rgba(127,219,255,0.7);
        }
        input, .stTextInput>div>div>input {
            background: #001f3f !important;
            border: 1px solid #7FDBFF !important;
            color: #f0f8ff !important;
            border-radius: 4px;
            padding: 8px;
        }
        input:focus, .stTextInput>div>div>input:focus {
            outline: none;
            box-shadow: 0 0 5px 2px rgba(127,219,255,0.5);
        }
        .stDownloadButton button {
            background-color: #89CFF0;
            color: #001f3f;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: transform 0.2s ease;
        }
        .stDownloadButton button:hover {
            transform: scale(1.03);
        }
        
        /* Footer & Back-to-Top */
        .footer {
            background: #001f3f;
            border-top: 2px solid #7FDBFF;
            text-align: center;
            padding: 20px;
            margin: 40px 20px;
            border-radius: 10px;
        }
        .footer p { margin: 0; font-size: 15px; color: #f0f8ff; }
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #7FDBFF;
            color: #001f3f;
            border-radius: 50%;
            padding: 12px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10000;
        }
        """)
    elif st.session_state.theme == "Green":
        custom_css = """<style>/* ... Green theme code ... */</style>"""
        custom_css = custom_css.replace("/* ... Green theme code ... */", """
        /* Global Settings - Green Theme */
        html, body {
            margin: 0;
            padding: 0;
            background: #0b3d0b;
            color: #e6ffe6;
            font-family: 'Roboto Mono', monospace;
            scroll-behavior: smooth;
            overflow-x: hidden;
        }
        body {
            background: radial-gradient(circle at center, #0b3d0b 0%, #2ecc40 60%, #0b3d0b 100%) fixed;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #2ecc40; }
        ::-webkit-scrollbar-thumb { background-color: #7bed9f; border-radius: 10px; }
        
        /* Rolling Alert Marquee */
        .rolling-alert {
            width: 100%;
            background: #2ecc40;
            overflow: hidden;
            position: relative;
            height: 40px;
            display: flex;
            align-items: center;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px #7bed9f;
            z-index: 9999;
        }
        .rolling-alert span {
            position: absolute;
            white-space: nowrap;
            animation: scroll-left 10s linear infinite;
            color: #e6ffe6;
            text-shadow: 0 0 5px rgba(123,237,159,0.8), 0 0 10px rgba(123,237,159,0.6);
        }
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        /* Parallax Header */
        .parallax-header {
            position: relative;
            height: 400px;
            background-image: url('https://source.unsplash.com/1600x900/?forest,storm');
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px rgba(123,237,159,0.4);
        }
        .parallax-header::after {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(135deg, rgba(123,237,159,0.2), rgba(0,102,51,0.5));
            z-index: 1;
        }
        .header-text {
            position: relative;
            z-index: 2;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
        }
        .header-text h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 60px;
            margin: 0;
            letter-spacing: 2px;
            color: #7bed9f;
            cursor: default;
        }
        .header-text p { font-size: 20px; margin-top: 10px; color: #d0f0d0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #0b3d0b;
            border-right: 2px solid #7bed9f;
            box-shadow: 0 0 15px rgba(123,237,159,0.5);
            transition: all 0.4s ease-in-out;
        }
        [data-testid="stSidebar"]:hover {
            transform: translateX(3px);
            box-shadow: inset 0 0 10px rgba(123,237,159,0.4);
        }
        
        /* Glassmorphism Cards */
        .card {
            background: rgba(11,61,11,0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(123,237,159,0.3);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 8px 20px rgba(123,237,159,0.1);
            transition: transform 0.3s ease;
            color: #e6ffe6;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(123,237,159,0.3);
        }
        
        /* Buttons & Inputs */
        .stButton>button {
            background-color: #7bed9f;
            color: #0b3d0b;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: transform 0.2s ease;
            box-shadow: 0 0 8px rgba(123,237,159,0.4);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 12px rgba(123,237,159,0.7);
        }
        input, .stTextInput>div>div>input {
            background: #0b3d0b !important;
            border: 1px solid #7bed9f !important;
            color: #e6ffe6 !important;
            border-radius: 4px;
            padding: 8px;
        }
        input:focus, .stTextInput>div>div>input:focus {
            outline: none;
            box-shadow: 0 0 5px 2px rgba(123,237,159,0.5);
        }
        .stDownloadButton button {
            background-color: #9be7a0;
            color: #0b3d0b;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: transform 0.2s ease;
        }
        .stDownloadButton button:hover {
            transform: scale(1.03);
        }
        
        /* Footer & Back-to-Top */
        .footer {
            background: #0b3d0b;
            border-top: 2px solid #7bed9f;
            text-align: center;
            padding: 20px;
            margin: 40px 20px;
            border-radius: 10px;
        }
        .footer p { margin: 0; font-size: 15px; color: #e6ffe6; }
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #7bed9f;
            color: #0b3d0b;
            border-radius: 50%;
            padding: 12px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10000;
        }
        """)
    elif st.session_state.theme == "Orange":
        custom_css = """<style>/* ... Orange theme ... */</style>"""
        custom_css = custom_css.replace("/* ... Orange theme ... */", """
        /* Global Settings - Orange Theme */
        html, body {
            margin: 0;
            padding: 0;
            background: #663300;
            color: #fff8e1;
            font-family: 'Roboto Mono', monospace;
            scroll-behavior: smooth;
            overflow-x: hidden;
        }
        body {
            background: radial-gradient(circle at center, #663300 0%, #ff851b 60%, #663300 100%) fixed;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #ff851b; }
        ::-webkit-scrollbar-thumb { background-color: #ffb84d; border-radius: 10px; }
        
        /* Rolling Alert Marquee */
        .rolling-alert {
            width: 100%;
            background: #ff851b;
            overflow: hidden;
            position: relative;
            height: 40px;
            display: flex;
            align-items: center;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px #ffb84d;
            z-index: 9999;
        }
        .rolling-alert span {
            position: absolute;
            white-space: nowrap;
            animation: scroll-left 10s linear infinite;
            color: #fff8e1;
            text-shadow: 0 0 5px rgba(255,184,77,0.8), 0 0 10px rgba(255,184,77,0.6);
        }
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        /* Parallax Header */
        .parallax-header {
            position: relative;
            height: 400px;
            background-image: url('https://source.unsplash.com/1600x900/?sunset,disaster');
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px rgba(255,184,77,0.4);
        }
        .parallax-header::after {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(135deg, rgba(255,184,77,0.2), rgba(204,85,0,0.5));
            z-index: 1;
        }
        .header-text {
            position: relative;
            z-index: 2;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
        }
        .header-text h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 60px;
            margin: 0;
            letter-spacing: 2px;
            color: #ffb84d;
            cursor: default;
        }
        .header-text p { font-size: 20px; margin-top: 10px; color: #ffe6cc; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #663300;
            border-right: 2px solid #ffb84d;
            box-shadow: 0 0 15px rgba(255,184,77,0.5);
            transition: all 0.4s ease-in-out;
        }
        [data-testid="stSidebar"]:hover {
            transform: translateX(3px);
            box-shadow: inset 0 0 10px rgba(255,184,77,0.4);
        }
        
        /* Glassmorphism Cards */
        .card {
            background: rgba(102,51,0,0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,184,77,0.3);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 8px 20px rgba(255,184,77,0.1);
            transition: transform 0.3s ease;
            color: #fff8e1;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(255,184,77,0.3);
        }
        
        /* Buttons & Inputs */
        .stButton>button {
            background-color: #ffb84d;
            color: #663300;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: transform 0.2s ease;
            box-shadow: 0 0 8px rgba(255,184,77,0.4);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 12px rgba(255,184,77,0.7);
        }
        input, .stTextInput>div>div>input {
            background: #663300 !important;
            border: 1px solid #ffb84d !important;
            color: #fff8e1 !important;
            border-radius: 4px;
            padding: 8px;
        }
        input:focus, .stTextInput>div>div>input:focus {
            outline: none;
            box-shadow: 0 0 5px 2px rgba(255,184,77,0.5);
        }
        .stDownloadButton button {
            background-color: #ffcc80;
            color: #663300;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: transform 0.2s ease;
        }
        .stDownloadButton button:hover {
            transform: scale(1.03);
        }
        
        /* Footer & Back-to-Top */
        .footer {
            background: #663300;
            border-top: 2px solid #ffb84d;
            text-align: center;
            padding: 20px;
            margin: 40px 20px;
            border-radius: 10px;
        }
        .footer p { margin: 0; font-size: 15px; color: #fff8e1; }
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #ffb84d;
            color: #663300;
            border-radius: 50%;
            padding: 12px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10000;
        }
        """)
    elif st.session_state.theme == "Purple":
        custom_css = """<style>/* ... Purple theme ... */</style>"""
        custom_css = custom_css.replace("/* ... Purple theme ... */", """
        /* Global Settings - Purple Theme */
        html, body {
            margin: 0;
            padding: 0;
            background: #2e004f;
            color: #f3e5f5;
            font-family: 'Roboto Mono', monospace;
            scroll-behavior: smooth;
            overflow-x: hidden;
        }
        body {
            background: radial-gradient(circle at center, #2e004f 0%, #6a1b9a 60%, #2e004f 100%) fixed;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #6a1b9a; }
        ::-webkit-scrollbar-thumb { background-color: #d1c4e9; border-radius: 10px; }
        
        /* Rolling Alert Marquee */
        .rolling-alert {
            width: 100%;
            background: #6a1b9a;
            overflow: hidden;
            position: relative;
            height: 40px;
            display: flex;
            align-items: center;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px #d1c4e9;
            z-index: 9999;
        }
        .rolling-alert span {
            position: absolute;
            white-space: nowrap;
            animation: scroll-left 10s linear infinite;
            color: #f3e5f5;
            text-shadow: 0 0 5px rgba(209,196,233,0.8), 0 0 10px rgba(209,196,233,0.6);
        }
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        /* Parallax Header */
        .parallax-header {
            position: relative;
            height: 400px;
            background-image: url('https://source.unsplash.com/1600x900/?purple,abstract');
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px rgba(209,196,233,0.4);
        }
        .parallax-header::after {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(135deg, rgba(209,196,233,0.2), rgba(103,58,183,0.5));
            z-index: 1;
        }
        .header-text {
            position: relative;
            z-index: 2;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
        }
        .header-text h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 60px;
            margin: 0;
            letter-spacing: 2px;
            color: #d1c4e9;
            cursor: default;
        }
        .header-text p { font-size: 20px; margin-top: 10px; color: #e1bee7; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #2e004f;
            border-right: 2px solid #d1c4e9;
            box-shadow: 0 0 15px rgba(209,196,233,0.5);
            transition: all 0.4s ease-in-out;
        }
        [data-testid="stSidebar"]:hover {
            transform: translateX(3px);
            box-shadow: inset 0 0 10px rgba(209,196,233,0.4);
        }
        
        /* Glassmorphism Cards */
        .card {
            background: rgba(46,0,79,0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(209,196,233,0.3);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 8px 20px rgba(209,196,233,0.1);
            transition: transform 0.3s ease;
            color: #f3e5f5;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(209,196,233,0.3);
        }
        
        /* Buttons & Inputs */
        .stButton>button {
            background-color: #d1c4e9;
            color: #2e004f;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: transform 0.2s ease;
            box-shadow: 0 0 8px rgba(209,196,233,0.4);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 12px rgba(209,196,233,0.7);
        }
        input, .stTextInput>div>div>input {
            background: #2e004f !important;
            border: 1px solid #d1c4e9 !important;
            color: #f3e5f5 !important;
            border-radius: 4px;
            padding: 8px;
        }
        input:focus, .stTextInput>div>div>input:focus {
            outline: none;
            box-shadow: 0 0 5px 2px rgba(209,196,233,0.5);
        }
        .stDownloadButton button {
            background-color: #9575cd;
            color: #2e004f;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: transform 0.2s ease;
        }
        .stDownloadButton button:hover {
            transform: scale(1.03);
        }
        
        /* Footer & Back-to-Top */
        .footer {
            background: #2e004f;
            border-top: 2px solid #d1c4e9;
            text-align: center;
            padding: 20px;
            margin: 40px 20px;
            border-radius: 10px;
        }
        .footer p { margin: 0; font-size: 15px; color: #f3e5f5; }
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #d1c4e9;
            color: #2e004f;
            border-radius: 50%;
            padding: 12px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10000;
        }
        """)
    elif st.session_state.theme == "Yellow":
        custom_css = """<style>/* ... Yellow theme ... */</style>"""
        custom_css = custom_css.replace("/* ... Yellow theme ... */", """
        /* Global Settings - Yellow Theme */
        html, body {
            margin: 0;
            padding: 0;
            background: #665c00;
            color: #fff9c4;
            font-family: 'Roboto Mono', monospace;
            scroll-behavior: smooth;
            overflow-x: hidden;
        }
        body {
            background: radial-gradient(circle at center, #665c00 0%, #ffeb3b 60%, #665c00 100%) fixed;
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #ffeb3b; }
        ::-webkit-scrollbar-thumb { background-color: #fff176; border-radius: 10px; }
        
        /* Rolling Alert Marquee */
        .rolling-alert {
            width: 100%;
            background: #ffeb3b;
            overflow: hidden;
            position: relative;
            height: 40px;
            display: flex;
            align-items: center;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px #fff176;
            z-index: 9999;
        }
        .rolling-alert span {
            position: absolute;
            white-space: nowrap;
            animation: scroll-left 10s linear infinite;
            color: #fff9c4;
            text-shadow: 0 0 5px rgba(255,243,0,0.8), 0 0 10px rgba(255,243,0,0.6);
        }
        @keyframes scroll-left {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        /* Parallax Header */
        .parallax-header {
            position: relative;
            height: 400px;
            background-image: url('https://source.unsplash.com/1600x900/?sunflower,yellow');
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px rgba(255,243,0,0.4);
        }
        .parallax-header::after {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(135deg, rgba(255,243,0,0.2), rgba(255,235,59,0.5));
            z-index: 1;
        }
        .header-text {
            position: relative;
            z-index: 2;
            text-align: center;
            animation: fadeIn 2s ease-in-out;
        }
        .header-text h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 60px;
            margin: 0;
            letter-spacing: 2px;
            color: #fff176;
            cursor: default;
        }
        .header-text p { font-size: 20px; margin-top: 10px; color: #fff9c4; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #665c00;
            border-right: 2px solid #fff176;
            box-shadow: 0 0 15px rgba(255,243,0,0.5);
            transition: all 0.4s ease-in-out;
        }
        [data-testid="stSidebar"]:hover {
            transform: translateX(3px);
            box-shadow: inset 0 0 10px rgba(255,243,0,0.4);
        }
        
        /* Glassmorphism Cards */
        .card {
            background: rgba(102,92,0,0.7);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,243,0,0.3);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 40px;
            box-shadow: 0 8px 20px rgba(255,243,0,0.1);
            transition: transform 0.3s ease;
            color: #fff9c4;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(255,243,0,0.3);
        }
        
        /* Buttons & Inputs */
        .stButton>button {
            background-color: #fff176;
            color: #665c00;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            transition: transform 0.2s ease;
            box-shadow: 0 0 8px rgba(255,243,0,0.4);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 12px rgba(255,243,0,0.7);
        }
        input, .stTextInput>div>div>input {
            background: #665c00 !important;
            border: 1px solid #fff176 !important;
            color: #fff9c4 !important;
            border-radius: 4px;
            padding: 8px;
        }
        input:focus, .stTextInput>div>div>input:focus {
            outline: none;
            box-shadow: 0 0 5px 2px rgba(255,243,0,0.5);
        }
        .stDownloadButton button {
            background-color: #fff59d;
            color: #665c00;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: transform 0.2s ease;
        }
        .stDownloadButton button:hover {
            transform: scale(1.03);
        }
        
        /* Footer & Back-to-Top */
        .footer {
            background: #665c00;
            border-top: 2px solid #fff176;
            text-align: center;
            padding: 20px;
            margin: 40px 20px;
            border-radius: 10px;
        }
        .footer p { margin: 0; font-size: 15px; color: #fff9c4; }
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #fff176;
            color: #665c00;
            border-radius: 50%;
            padding: 12px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 10000;
        }
        """)
    else:
        # fallback no style
        custom_css = ""
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_theme()

# -------------------------------------------------------
# SCROLLING ALERT (Marquee)
# -------------------------------------------------------
st.markdown(
    """
    <div class="rolling-alert">
        <span>⚠️ ALERT: Severe Weather Warning in Your Region – Stay Informed & Prepared! ⚠️</span>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# PARALLAX HEADER (Image from Unsplash)
# -------------------------------------------------------
st.markdown(
    """
    <div class="parallax-header">
        <div class="header-text">
            <h1>Disaster Alert Hub</h1>
            <p>AI-Powered Disaster Predictions</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# MAIN DATABASE (Users, History)
# -------------------------------------------------------
def init_db():
    """
    Initialize the main database (dpah_database.db) for user accounts and history.
    Using unique table names: 'users_new', 'history_new'.
    """
    try:
        conn = sqlite3.connect('dpah_database.db')
        c = conn.cursor()
        # Create table for user accounts, including 'approved' and 'is_admin'
        c.execute('''
            CREATE TABLE IF NOT EXISTS users_new (
                username TEXT PRIMARY KEY,
                password TEXT,
                name TEXT,
                location TEXT,
                approved INTEGER DEFAULT 0,
                is_admin INTEGER DEFAULT 0
            )
        ''')
        # Historical predictions
        c.execute('''
            CREATE TABLE IF NOT EXISTS history_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                disaster TEXT,
                prediction TEXT,
                probability REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Insert a default admin (username='admin') if not present
        c.execute("SELECT * FROM users_new WHERE username='admin'")
        admin_exists = c.fetchone()
        if not admin_exists:
            c.execute("""
                INSERT INTO users_new (username, password, name, location, approved, is_admin)
                VALUES ('admin', 'admin123', 'System Administrator', 'HQ', 1, 1)
            """)
        conn.commit()
    except Exception as e:
        st.error("DB Error: " + str(e))
    finally:
        conn.close()

# -------------------------------------------------------
# MESSAGES DATABASE
# -------------------------------------------------------
def init_msg_db():
    """
    Initialize the messages database (dpah_messages.db) for user-admin messages.
    Table: 'messages_new'
    """
    try:
        conn = sqlite3.connect('dpah_messages.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_user TEXT,
                to_user TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    except Exception as e:
        st.error("Message DB Error: " + str(e))
    finally:
        conn.close()

# -------------------------------------------------------
# NOTIFICATIONS DATABASE
# -------------------------------------------------------
def init_notif_db():
    """
    Initialize the notifications database (dpah_notifications.db).
    Table: 'notifications_new'
    """
    try:
        conn = sqlite3.connect('dpah_notifications.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS notifications_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender TEXT,
                recipient TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
    except Exception as e:
        st.error("Notification DB Error: " + str(e))
    finally:
        conn.close()


# Call initialization
init_db()
init_msg_db()
init_notif_db()

# -------------------------------------------------------
# AUTH / USER FUNCTIONS (in dpah_database.db)
# -------------------------------------------------------
def create_user(username, password, name, location):
    """
    Create a new user in dpah_database.db with approved=0 by default (pending admin approval).
    """
    try:
        conn = sqlite3.connect('dpah_database.db')
        c = conn.cursor()
        c.execute(
            '''
            INSERT INTO users_new (username, password, name, location, approved, is_admin)
            VALUES (?, ?, ?, ?, 0, 0)
            ''',
            (username, password, name, location)
        )
        conn.commit()
        st.success("User account created successfully! Waiting for admin approval.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    except Exception as e:
        st.error("Error creating user: " + str(e))
    finally:
        conn.close()

def authenticate_user(username, password):
    """
    Return (True, is_admin) if user is found in dpah_database.db, password matches, and approved=1.
    Else return (False, False).
    """
    try:
        conn = sqlite3.connect('dpah_database.db')
        c = conn.cursor()
        c.execute('SELECT password, approved, is_admin FROM users_new WHERE username=?', (username,))
        result = c.fetchone()
        if result:
            db_password, approved, is_admin_val = result
            if db_password == password and approved == 1:
                return (True, bool(is_admin_val))
        return (False, False)
    except Exception as e:
        st.error("Authentication Error: " + str(e))
        return (False, False)
    finally:
        conn.close()

def get_pending_users():
    """
    Get all users where approved=0 and is_admin=0
    """
    conn = sqlite3.connect('dpah_database.db')
    c = conn.cursor()
    c.execute("SELECT username, name, location FROM users_new WHERE approved=0 AND is_admin=0")
    rows = c.fetchall()
    conn.close()
    return rows

def approve_user(username):
    """
    Approve a user (set approved=1)
    """
    try:
        conn = sqlite3.connect('dpah_database.db')
        c = conn.cursor()
        c.execute("UPDATE users_new SET approved=1 WHERE username=?", (username,))
        conn.commit()
        st.success(f"User '{username}' has been approved.")
    except Exception as e:
        st.error("Error approving user: " + str(e))
    finally:
        conn.close()

def remove_user(username):
    """
    Remove a user from dpah_database.db
    """
    try:
        conn = sqlite3.connect('dpah_database.db')
        c = conn.cursor()
        c.execute("DELETE FROM users_new WHERE username=?", (username,))
        conn.commit()
        st.success(f"User '{username}' has been removed.")
    except Exception as e:
        st.error("Error removing user: " + str(e))
    finally:
        conn.close()

def get_all_approved_users():
    """
    Return all (username, name) for approved, non-admin users
    """
    conn = sqlite3.connect('dpah_database.db')
    c = conn.cursor()
    c.execute("SELECT username, name FROM users_new WHERE approved=1 AND is_admin=0")
    rows = c.fetchall()
    conn.close()
    return rows

# -------------------------------------------------------
# HISTORY LOGGING (dpah_database.db)
# -------------------------------------------------------
def log_history(username, disaster, prediction, probability):
    try:
        conn = sqlite3.connect('dpah_database.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO history_new (username, disaster, prediction, probability)
            VALUES (?, ?, ?, ?)
        ''', (username, disaster, prediction, probability))
        conn.commit()
    except Exception as e:
        st.error("History Logging Error: " + str(e))
    finally:
        conn.close()

# -------------------------------------------------------
# NOTIFICATIONS (dpah_notifications.db)
# -------------------------------------------------------
def send_notification(sender, recipient, message):
    try:
        conn = sqlite3.connect('dpah_notifications.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO notifications_new (sender, recipient, message)
            VALUES (?, ?, ?)
        ''', (sender, recipient, message))
        conn.commit()
        st.success("Notification sent!")
    except Exception as e:
        st.error("Error sending notification: " + str(e))
    finally:
        conn.close()

def get_user_notifications(username):
    """
    Return all notifications for a given user (either 'ALL' or the user's username).
    from dpah_notifications.db
    """
    conn = sqlite3.connect('dpah_notifications.db')
    c = conn.cursor()
    c.execute("""
        SELECT id, sender, recipient, message, timestamp 
        FROM notifications_new
        WHERE recipient='ALL' OR recipient=?
        ORDER BY timestamp DESC
        """, (username,))
    rows = c.fetchall()
    conn.close()
    return rows

# -------------------------------------------------------
# MESSAGES (dpah_messages.db)
# -------------------------------------------------------
def send_message(from_user, to_user, msg):
    """
    Insert a new message into dpah_messages.db
    """
    try:
        conn = sqlite3.connect('dpah_messages.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO messages_new (from_user, to_user, message)
            VALUES (?, ?, ?)
        ''', (from_user, to_user, msg))
        conn.commit()
    except Exception as e:
        st.error("Error sending message: " + str(e))
    finally:
        conn.close()

def get_chat_messages(user1, user2):
    """
    Retrieve conversation between user1 and user2 from dpah_messages.db in chronological order.
    """
    conn = sqlite3.connect('dpah_messages.db')
    c = conn.cursor()
    c.execute('''
        SELECT from_user, to_user, message, timestamp
        FROM messages_new
        WHERE (from_user=? AND to_user=?) OR (from_user=? AND to_user=?)
        ORDER BY timestamp ASC
    ''', (user1, user2, user2, user1))
    rows = c.fetchall()
    conn.close()
    return rows

# -------------------------------------------------------
# SIMULATED EMAIL REPORTING
# -------------------------------------------------------
def send_email(report, recipient):
    st.info(f"Simulated email sent to {recipient}:\n\n{report}")

# -------------------------------------------------------
# LOAD MODEL (with caching)
# -------------------------------------------------------
@st.cache_resource
def load_model(model_file):
    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_file}: {e}")
        return None

# -------------------------------------------------------
# PREDICTION LOGIC
# -------------------------------------------------------
def predict_disaster(model, df):
    try:
        prob = model.predict_proba(df)[0][1]
        pred = "Harm" if prob > 0.5 else "No Harm"
        return pred, prob
    except Exception as e:
        st.error("Prediction error: " + str(e))
        return "Unknown", 0.0

def prevention_info(disaster, pred, prob):
    if pred == "No Harm":
        return ("SAFE",
                "Situation is stable. Stay informed with local updates.",
                "Helpline: 123-456-7890")
    else:
        if prob > 0.8:
            return ("CRITICAL",
                    "Evacuate immediately! Follow emergency instructions.",
                    "Emergency: 911")
        elif prob > 0.6:
            return ("WARNING",
                    "Prepare for possible evacuation. Secure valuables.",
                    "Assistance: 1800-XYZ-HELP")
        else:
            return ("CAUTION",
                    "Stay alert and prepare an emergency kit.",
                    "Local Support: 555-123-4567")

# -------------------------------------------------------
# HTML + JS FOR THE ADVISOR CHATBOT
# -------------------------------------------------------
def get_chatbot_html(user_name):
    base_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Advisor Chatbot</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {
      --primary-color: {primary};
      --secondary-color: {secondary};
      --bg-color: {bg};
      --text-color: {text};
      --accent-color: {secondary};
      --container-bg: rgba({container_bg},0.9);
      --header-bg: linear-gradient(135deg, {header1}, {header2});
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Roboto Mono', monospace;
      background: var(--bg-color);
      color: var(--text-color);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .chat-container {
      width: 100%;
      max-width: 700px;
      background: var(--container-bg);
      border-radius: 15px;
      border: 1px solid var(--primary-color);
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      animation: popIn 0.7s ease;
    }
    @keyframes popIn { from { transform: scale(0.8); opacity: 0; } to { transform: scale(1); opacity: 1; } }
    .chat-header {
      background: var(--header-bg);
      padding: 20px;
      text-align: center;
      box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    .chat-header h2 {
      font-size: 2em;
      color: #fff;
      margin-bottom: 5px;
      text-shadow: 0 0 5px rgba(0,0,0,0.4);
    }
    .chat-header p { font-size: 1em; color: #ddd; }
    .chat-box {
      flex: 1;
      padding: 20px;
      background: var(--bg-color);
      overflow-y: auto;
      scrollbar-color: var(--primary-color) var(--bg-color);
      scrollbar-width: thin;
      max-height: 500px;
    }
    .message { display: flex; align-items: flex-start; margin-bottom: 15px; animation: fadeInUp 0.4s ease; }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .message .icon { font-size: 24px; margin-right: 10px; color: var(--primary-color); }
    .user-message, .bot-message {
      max-width: 75%;
      padding: 12px 16px;
      border-radius: 20px;
      position: relative;
      word-wrap: break-word;
      overflow-wrap: break-word;
    }
    .user-message {
      background: var(--secondary-color);
      color: var(--bg-color);
      margin-left: auto;
      border-bottom-right-radius: 0;
      text-align: right;
    }
    .user-message:hover { border: 2px solid var(--secondary-color); }
    .bot-message {
      background: var(--primary-color);
      color: var(--bg-color);
      margin-right: auto;
      border-bottom-left-radius: 0;
      text-align: left;
    }
    .timestamp { font-size: 10px; color: #fdd; margin-top: 5px; }
    .input-section {
      background: var(--bg-color);
      padding: 15px;
      border-top: 1px solid var(--primary-color);
    }
    .input-container { display: flex; gap: 10px; margin-bottom: 10px; }
    .input-container input {
      flex: 1;
      padding: 12px;
      border-radius: 8px;
      border: 2px solid var(--primary-color);
      font-size: 16px;
      background: var(--bg-color);
      color: var(--text-color);
    }
    .input-container button {
      padding: 12px 20px;
      border: none;
      background: var(--primary-color);
      color: var(--bg-color);
      border-radius: 8px;
      cursor: pointer;
      font-size: 16px;
      transition: transform 0.3s ease;
    }
    .input-container button:hover { transform: scale(1.05); background: var(--secondary-color); }
    .quick-replies { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-bottom: 10px; }
    .quick-replies button {
      padding: 8px 12px;
      border: none;
      background: var(--primary-color);
      color: var(--bg-color);
      border-radius: 20px;
      cursor: pointer;
      font-size: 14px;
      transition: transform 0.2s ease, background 0.3s ease;
    }
    .quick-replies button:hover { transform: scale(1.05); background: var(--secondary-color); }
    .clear-btn {
      background: var(--secondary-color);
      width: 100%;
      padding: 12px;
      border: none;
      border-radius: 8px;
      font-size: 16px;
      cursor: pointer;
      transition: background 0.3s ease;
      font-weight: 600;
    }
    .clear-btn:hover { background: var(--primary-color); }
  </style>
</head>
<body>
  <div class="chat-container">
    <div class="chat-header">
      <h2><i class="fa-solid fa-robot icon"></i> Welcome, {USER_NAME}!</h2>
      <p>Your Advisor is at your service.</p>
    </div>
    <div class="chat-box" id="chat-box">
      <div class="message bot-message">
        <i class="fa-solid fa-robot icon"></i>
        <div id="bot-initial-message"></div>
      </div>
    </div>
    <div class="input-section">
      <div class="input-container">
        <input type="text" id="user-input" placeholder="Type your message..." onkeypress="handleEnter(event)">
        <button onclick="sendMessage()"><i class="fa-solid fa-paper-plane"></i></button>
      </div>
      <div class="quick-replies">
        <button onclick="sendQuickReply('Evacuation')"><i class="fa-solid fa-route"></i> Evacuation</button>
        <button onclick="sendQuickReply('Emergency Kit')"><i class="fa-solid fa-briefcase-medical"></i> Emergency Kit</button>
        <button onclick="sendQuickReply('Shelter')"><i class="fa-solid fa-house-flood-water"></i> Shelter</button>
        <button onclick="sendQuickReply('Alert Levels')"><i class="fa-solid fa-triangle-exclamation"></i> Alert</button>
        <button onclick="sendQuickReply('Contact Support')"><i class="fa-solid fa-headset"></i> Support</button>
      </div>
      <button class="clear-btn" onclick="clearChat()"><i class="fa-solid fa-trash"></i> Clear Chat</button>
    </div>
  </div>
  <script>
    const initialMessage = "Hello {USER_NAME}, how may I assist you with disaster preparedness today?";
    const typingSpeed = 40;
    let charIndex = 0;
    function typeWriter() {
      const container = document.getElementById("bot-initial-message");
      if(!container) return;
      if(charIndex < initialMessage.length) {
        container.innerHTML += initialMessage.charAt(charIndex);
        charIndex++;
        setTimeout(typeWriter, typingSpeed);
      } else {
        const timeSpan = document.createElement("div");
        timeSpan.className = "timestamp";
        timeSpan.innerText = getCurrentTime();
        container.appendChild(timeSpan);
      }
    }
    window.onload = typeWriter;
    function getCurrentTime() {
      const now = new Date();
      return now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    }
    function sendMessage() {
      const inputField = document.getElementById("user-input");
      const message = inputField.value.trim();
      if(!message) return;
      const chatBox = document.getElementById("chat-box");
      const userMsgContainer = document.createElement("div");
      userMsgContainer.className = "message user-message";
      userMsgContainer.innerHTML = `
        <i class="fa-solid fa-user icon"></i>
        <div>
          ${message}
          <div class="timestamp">${getCurrentTime()}</div>
        </div>
      `;
      chatBox.appendChild(userMsgContainer);
      inputField.value = "";
      chatBox.scrollTop = chatBox.scrollHeight;
      handleBotResponse(message);
    }
    function sendQuickReply(reply) {
      document.getElementById("user-input").value = reply;
      sendMessage();
    }
    function handleEnter(event) { if(event.key === "Enter") { sendMessage(); } }
    function handleBotResponse(message) {
      const responses = {
        "evacuation": "In case of mandatory evacuation, follow local directives to the nearest safe zone.",
        "emergency": "An emergency kit should include water, food, first aid, flashlights, and important docs.",
        "support": "Contact local helplines or official agencies for urgent assistance.",
        "alert": "Typical alert levels: SAFE, CAUTION, WARNING, CRITICAL. Stay updated with official channels.",
        "thank": "You're welcome! Stay vigilant.",
        "shelter": "Check community guidelines for designated shelters in your area.",
        "hello": "Hello! I'm here to help with any preparedness questions.",
        "hi": "Hi there! Ask me about disasters or prevention tips."
      };
      let response = "Could you clarify that? I'm here to help.";
      const msgLower = message.toLowerCase();
      Object.keys(responses).forEach((key) => { if(msgLower.includes(key)) { response = responses[key]; } });
      showBotResponse(response);
    }
    function showBotResponse(response) {
      const chatBox = document.getElementById("chat-box");
      const typingDiv = document.createElement("div");
      typingDiv.className = "message bot-message typing";
      typingDiv.innerHTML = `
        <i class="fa-solid fa-robot icon"></i>
        <div>Typing...
          <div class="timestamp">${getCurrentTime()}</div>
        </div>
      `;
      chatBox.appendChild(typingDiv);
      chatBox.scrollTop = chatBox.scrollHeight;
      setTimeout(() => {
        chatBox.removeChild(typingDiv);
        const botMsgContainer = document.createElement("div");
        botMsgContainer.className = "message bot-message";
        botMsgContainer.innerHTML = `
          <i class="fa-solid fa-robot icon"></i>
          <div>
            ${response}
            <div class="timestamp">${getCurrentTime()}</div>
          </div>
        `;
        chatBox.appendChild(botMsgContainer);
        chatBox.scrollTop = chatBox.scrollHeight;
      }, 1200);
    }
    function clearChat() {
      const chatBox = document.getElementById("chat-box");
      chatBox.innerHTML = "";
      chatBox.style.backgroundColor = '#111';
      setTimeout(() => { chatBox.style.backgroundColor = 'var(--bg-color)'; }, 200);
      const clearedMsg = document.createElement("div");
      clearedMsg.className = "message bot-message";
      clearedMsg.innerHTML = `
        <i class="fa-solid fa-robot icon"></i>
        <div>
          Chat cleared. How else can I assist you?
          <div class="timestamp">${getCurrentTime()}</div>
        </div>
      `;
      chatBox.appendChild(clearedMsg);
    }
  </script>
</body>
</html>
"""
    # Replace placeholders with color variables
    if st.session_state.theme == "Red":
        color_vars = {
            "primary": "#ff4d4d",
            "secondary": "#f4a0a0",
            "bg": "#330000",
            "text": "#fefefe",
            "container_bg": "51,0,0",
            "header1": "#660000",
            "header2": "#330000"
        }
    elif st.session_state.theme == "Blue":
        color_vars = {
            "primary": "#7FDBFF",
            "secondary": "#89CFF0",
            "bg": "#001f3f",
            "text": "#f0f8ff",
            "container_bg": "0,31,63",
            "header1": "#0074D9",
            "header2": "#001f3f"
        }
    elif st.session_state.theme == "Green":
        color_vars = {
            "primary": "#7bed9f",
            "secondary": "#9be7a0",
            "bg": "#0b3d0b",
            "text": "#e6ffe6",
            "container_bg": "11,61,11",
            "header1": "#2ecc40",
            "header2": "#0b3d0b"
        }
    elif st.session_state.theme == "Orange":
        color_vars = {
            "primary": "#ffb84d",
            "secondary": "#ffcc80",
            "bg": "#663300",
            "text": "#fff8e1",
            "container_bg": "102,51,0",
            "header1": "#ff851b",
            "header2": "#663300"
        }
    elif st.session_state.theme == "Purple":
        color_vars = {
            "primary": "#d1c4e9",
            "secondary": "#9575cd",
            "bg": "#2e004f",
            "text": "#f3e5f5",
            "container_bg": "46,0,79",
            "header1": "#6a1b9a",
            "header2": "#2e004f"
        }
    elif st.session_state.theme == "Yellow":
        color_vars = {
            "primary": "#fff176",
            "secondary": "#fff59d",
            "bg": "#665c00",
            "text": "#fff9c4",
            "container_bg": "102,92,0",
            "header1": "#ffeb3b",
            "header2": "#665c00"
        }
    else:
        color_vars = {
            "primary": "#ff4d4d",
            "secondary": "#f4a0a0",
            "bg": "#330000",
            "text": "#fefefe",
            "container_bg": "51,0,0",
            "header1": "#660000",
            "header2": "#330000"
        }
    for key, value in color_vars.items():
        base_html = base_html.replace("{" + key + "}", value)
    return base_html.replace("{USER_NAME}", user_name)

# -------------------------------------------------------
# MAP FOR LOCATION SELECTION
# -------------------------------------------------------
def location_selector():
    st.markdown("<div class='card'><h3>Select Your Location</h3><p>Drag the marker to your current location.</p></div>", unsafe_allow_html=True)
    m = folium.Map(location=[20, 0], zoom_start=2)
    marker = folium.Marker(location=[20, 0], draggable=True)
    marker.add_to(m)
    map_data = st_folium(m, width=700, height=450)
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        lat = map_data["last_clicked"].get("lat", 20)
        lng = map_data["last_clicked"].get("lng", 0)
        return f"{lat:.4f}, {lng:.4f}"
    return "20.0000, 0.0000"

# -------------------------------------------------------
# ADDITIONAL RESOURCE PAGES & UI COMPONENTS
# -------------------------------------------------------
def emergency_resources():
    st.markdown("<div class='card'><h2>Emergency Resources</h2></div>", unsafe_allow_html=True)
    st.markdown(
        """
        **National Emergency Contacts:**
        - **911:** For immediate assistance
        - **1800-XYZ-HELP:** Disaster Helpline

        **Local Shelters:**
        - Shelter A: 123 Main St, Cityville – Tel: 111-222-3333
        - Shelter B: 456 Elm St, Townsville – Tel: 444-555-6666
        """
    )
    st.image("https://source.unsplash.com/800x600/?emergency", use_container_width=True)

def prevention_guidelines():
    st.markdown("<div class='card'><h2>Prevention Guidelines</h2></div>", unsafe_allow_html=True)
    disaster = st.selectbox("Select Disaster Type for Guidelines", 
                             ["Earthquake", "Flood", "Cyclone", "Wildfire", "Tsunami", "Heatwave"])
    if disaster == "Earthquake":
        st.markdown(
            """
            **Earthquake Guidelines:**
            - Secure heavy furniture.
            - Identify safe spots in your home.
            - Prepare an emergency kit.
            - Practice "Drop, Cover, and Hold On" drills.
            """
        )
    elif disaster == "Flood":
        st.markdown(
            """
            **Flood Guidelines:**
            - Clean gutters and drainage.
            - Elevate valuables.
            - Develop an evacuation plan.
            - Consider flood insurance.
            """
        )
    elif disaster == "Cyclone":
        st.markdown(
            """
            **Cyclone Guidelines:**
            - Reinforce windows and doors.
            - Secure outdoor items.
            - Stock up on emergency supplies.
            - Follow evacuation orders if issued.
            """
        )
    elif disaster == "Wildfire":
        st.markdown(
            """
            **Wildfire Guidelines:**
            - Create a safety buffer around your home.
            - Remove flammable materials.
            - Monitor air quality.
            - Follow evacuation instructions promptly.
            """
        )
    elif disaster == "Tsunami":
        st.markdown(
            """
            **Tsunami Guidelines:**
            - Move to higher ground immediately.
            - Stay away from the coast until official 'all clear'.
            - Follow all local emergency instructions.
            """
        )
    elif disaster == "Heatwave":
        st.markdown(
            """
            **Heatwave Guidelines:**
            - Stay hydrated.
            - Avoid strenuous activities during peak heat.
            - Check on vulnerable individuals.
            - Keep living spaces cool.
            """
        )
    st.image("https://source.unsplash.com/800x600/?safety", use_container_width=True)

def simulated_weather_forecast(location):
    st.markdown("<div class='card'><h2>Local Weather Forecast</h2></div>", unsafe_allow_html=True)
    forecast = {
        "Temperature": np.random.randint(15, 35),
        "Humidity": np.random.randint(40, 90),
        "Wind Speed": np.random.randint(5, 20),
        "Conditions": np.random.choice(["Sunny", "Cloudy", "Rainy", "Stormy"])
    }
    st.markdown(f"""
    - **Temperature:** {forecast["Temperature"]}°C  
    - **Humidity:** {forecast["Humidity"]}%  
    - **Wind Speed:** {forecast["Wind Speed"]} km/h  
    - **Conditions:** {forecast["Conditions"]}
    """)

def faq_section():
    st.markdown("<div class='card'><h2>Frequently Asked Questions</h2></div>", unsafe_allow_html=True)
    faqs = {
        "How do I sign up?": "Use the 'User Onboarding' section to sign up.",
        "Why can't I log in?": "Your account might still be pending admin approval.",
        "How is disaster prediction performed?": "We use machine learning models trained on historical data.",
        "What if I get a high-risk prediction?": "Follow prevention guidelines and local authority instructions."
    }
    for q, a in faqs.items():
        st.markdown(f"**Q: {q}**")
        st.markdown(f"A: {a}")
        st.markdown("---")

def about_section():
    st.markdown("<div class='card'><h2>About This System</h2></div>", unsafe_allow_html=True)
    st.markdown(
        """
        **Disaster Prediction & Alert Hub**

        This platform leverages AI and advanced data analytics to predict disasters—earthquakes, floods, cyclones, wildfires, tsunamis, and heatwaves.

        **Features:**
        - Admin & User login with secure user onboarding
        - Admin approval system for new sign-ups
        - Detailed input forms for multiple disaster types
        - Actionable alert levels and prevention guidelines
        - Chatbot advisor for preparedness
        - User–Admin Chat
        - Emergency resources, weather forecast
        - Alert system (NLP-based) for analyzing text
        - Dashboard with historical prediction trends

        **Developed by:** Your Team
        """
    )
    st.image("https://source.unsplash.com/800x600/?technology", use_container_width=True)

def contact_section():
    st.markdown("<div class='card'><h2>Contact Us</h2></div>", unsafe_allow_html=True)
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Send Message")
        if submitted:
            if name and email and message:
                st.success("Thank you for contacting us! We will get back to you soon.")
            else:
                st.warning("Please fill in all fields.")


def alert_system():
    st.markdown(
        "<div class='card'><h2>Alert System</h2>"
        "<p>Analyze news headlines or social media posts using advanced NLP techniques (simple regex tokenization and lemmatization) "
        "along with a comprehensive list of weighted keywords to detect potential disasters.</p></div>",
        unsafe_allow_html=True
    )
    st.markdown("### Enter News Headline or Social Media Post")
    alert_text = st.text_area("Input Text")
    
    if st.button("Analyze"):
        if alert_text.strip():
            # Initialize NLP tools.
            lemmatizer = WordNetLemmatizer()
            text_lower = alert_text.lower()
            # Use a regex-based tokenizer that splits text into words (alphanumeric characters).
            tokens = re.findall(r'\w+', text_lower)
            tokens_lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
            
            # Define a comprehensive dictionary of 100 disaster-related keywords with weights.
            keywords = {
                "disaster": 1.0,
                "earthquake": 1.5,
                "tremor": 1.5,
                "aftershock": 1.5,
                "flood": 1.2,
                "storm": 1.0,
                "hurricane": 1.3,
                "cyclone": 1.3,
                "typhoon": 1.3,
                "wildfire": 1.4,
                "forest fire": 1.4,
                "heatwave": 1.2,
                "drought": 1.2,
                "landslide": 1.2,
                "avalanche": 1.2,
                "eruption": 1.5,
                "volcano": 1.5,
                "chemical spill": 1.5,
                "radiation leak": 1.5,
                "biohazard": 1.5,
                "nuclear": 1.5,
                "crisis": 1.0,
                "emergency": 1.0,
                "evacuate": 1.0,
                "evacuation": 1.0,
                "alert": 1.0,
                "warning": 1.0,
                "critical": 1.0,
                "tsunami": 1.5,
                "mudslide": 1.2,
                "severe weather": 1.2,
                "blizzard": 1.3,
                "snowstorm": 1.3,
                "hailstorm": 1.1,
                "tornado": 1.4,
                "sandstorm": 1.2,
                "dust storm": 1.2,
                "bushfire": 1.4,
                "incendiary": 1.0,
                "burning": 1.0,
                "conflagration": 1.4,
                "blackout": 1.1,
                "power outage": 1.1,
                "infrastructure collapse": 1.5,
                "structural damage": 1.4,
                "collapse": 1.3,
                "explosion": 1.4,
                "bomb": 1.5,
                "detonation": 1.4,
                "hazard": 1.0,
                "threat": 1.0,
                "emergency services": 1.0,
                "rescue": 1.2,
                "relief": 1.1,
                "tragedy": 1.2,
                "casualty": 1.2,
                "fatality": 1.3,
                "injured": 1.2,
                "stranded": 1.1,
                "panic": 1.0,
                "riot": 1.0,
                "meltdown": 1.5,
                "oil spill": 1.4,
                "contamination": 1.3,
                "toxic": 1.3,
                "hazardous": 1.2,
                "outbreak": 1.3,
                "epidemic": 1.4,
                "pandemic": 1.5,
                "virus": 1.0,
                "bioattack": 1.5,
                "mass casualty": 1.5,
                "disaster relief": 1.2,
                "catastrophe": 1.5,
                "upheaval": 1.2,
                "turmoil": 1.1,
                "emergency alert": 1.0,
                "siren": 1.0,
                "evacuation order": 1.0,
                "aftermath": 1.1,
                "recovery": 1.1,
                "damage": 1.0,
                "destruction": 1.4,
                "inferno": 1.5,
                "engulf": 1.0,
                "smog": 1.1,
                "radioactive": 1.5,
                "crisis management": 1.2,
                "hazard zone": 1.2,
                "emergency response": 1.2,
                "risk": 1.0,
                "alert level": 1.0,
                "containment": 1.2,
                "seismic": 1.4,
                "volcanic": 1.5,
                "thunderstorm": 1.1,
                "lightning": 1.0,
                "rainfall": 1.0,
                "flash flood": 1.3,
                "smoke": 1.1,
                "ash": 1.1,
                "debris": 1.0,
                "wreckage": 1.2,
                "submersion": 1.3,
                "tsunamic": 1.5,
                "earth shaking": 1.5,
                "cataclysm": 1.5,
                "severe": 1.0,
                "violent": 1.0,
                "calamity": 1.5,
                "peril": 1.0
            }
            
            # Calculate a weighted score based on keyword occurrences.
            # For single-word keywords we use lemmatized tokens; for multi-word phrases, we check the raw lowercased text.
            score = 0.0
            for keyword, weight in keywords.items():
                if " " not in keyword:
                    count = tokens_lemmatized.count(keyword)
                else:
                    count = text_lower.count(keyword)
                score += count * weight

            # Compute a probability as a linear function of score (capped at 1.0).
            probability = min(score / 10, 1.0)
            
            # Classify the risk based on the computed score.
            if score >= 3:
                st.error(f"🚨 ALERT: High risk detected! Score: {score:.2f}, Probability: {probability:.2f}", icon="🚨")
            elif score >= 1:
                st.warning(f"Potential risk detected. Score: {score:.2f}, Probability: {probability:.2f}")
            elif score > 0:
                st.info(f"Low risk detected. Score: {score:.2f}, Probability: {probability:.2f}")
            else:
                st.success("No significant alert detected in the input text.")
        else:
            st.warning("Please enter some text to analyze.")


# -------------------------------------------------------
# SIDEBAR REAL-TIME CLOCK
# -------------------------------------------------------
clock_placeholder = st.sidebar.empty()
def update_clock():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clock_placeholder.markdown(f"### Current Time\n\n**{now}**")

update_clock()

# -------------------------------------------------------
# DYNAMIC SIDEBAR NAVIGATION
# -------------------------------------------------------
def get_sidebar_options_and_icons():
    """
    Build a list of (menu option labels) and a parallel list of icons.
    The final call to option_menu can use these lists.
    """
    # We'll create the full set:
    # Common items for both admin and user:
    #   "Home", "User Onboarding"
    # If logged in, add more items.
    # If admin, add Admin Panel
    # Always end with "Settings", "Logout" if logged in
    # Also add "Messages & Notifications" for normal users
    base_options = []
    base_icons = []

    # Always visible:
    base_options.append("Home")
    base_icons.append("house")  # or any icon you like

    base_options.append("User Onboarding")
    base_icons.append("person-add")

    # Public items:
    if not st.session_state.logged_in:
        # Some public items:
        base_options.append("FAQ")
        base_icons.append("question-circle")

        base_options.append("About")
        base_icons.append("info-circle")

        base_options.append("Contact Us")
        base_icons.append("envelope")

        base_options.append("Settings")
        base_icons.append("gear")

    else:
        # Logged in items:
        base_options.append("Disaster Prediction")
        base_icons.append("fire")

        base_options.append("Alert System")
        base_icons.append("bell")

        base_options.append("Advisor Chatbot")
        base_icons.append("robot")

        base_options.append("Dashboard")
        base_icons.append("bar-chart-fill")

        base_options.append("Emergency Resources")
        base_icons.append("life-preserver")

        base_options.append("Prevention Guidelines")
        base_icons.append("shield-check")

        base_options.append("Weather")
        base_icons.append("cloud-sun")

        base_options.append("FAQ")
        base_icons.append("question-circle")

        base_options.append("About")
        base_icons.append("info-circle")

        base_options.append("Contact Us")
        base_icons.append("envelope")

        # If user is NOT admin, let's add "Messages & Notifications"
        # so that normal users can see them
        if not st.session_state.is_admin:
            base_options.append("Messages & Notifications")
            base_icons.append("chat-dots")

        # Admin only
        if st.session_state.is_admin:
            base_options.append("Admin Panel")
            base_icons.append("person-check-fill")

        base_options.append("Settings")
        base_icons.append("gear")

        base_options.append("Logout")
        base_icons.append("box-arrow-right")

    return base_options, base_icons


options_list, icons_list = get_sidebar_options_and_icons()

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=options_list,
        icons=icons_list,
        menu_icon="list",
        default_index=0,
        styles={
            "container": {"padding": "10px"},
            "icon": {"font-size": "22px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": 
                    "#660000" if st.session_state.theme=="Red" 
                    else "#0074D9" if st.session_state.theme=="Blue" 
                    else "#2ecc40" if st.session_state.theme=="Green" 
                    else "#ff851b" if st.session_state.theme=="Orange" 
                    else "#6a1b9a" if st.session_state.theme=="Purple" 
                    else "#ffeb3b",
                "color": 
                    "#fefefe" if st.session_state.theme=="Red" 
                    else "#f0f8ff" if st.session_state.theme=="Blue" 
                    else "#e6ffe6" if st.session_state.theme=="Green" 
                    else "#fff8e1" if st.session_state.theme=="Orange" 
                    else "#f3e5f5" if st.session_state.theme=="Purple" 
                    else "#fff9c4"
            },
            "nav-link-selected": {
                "background-color": 
                    "#ff4d4d" if st.session_state.theme=="Red" 
                    else "#7FDBFF" if st.session_state.theme=="Blue" 
                    else "#7bed9f" if st.session_state.theme=="Green" 
                    else "#ffb84d" if st.session_state.theme=="Orange" 
                    else "#d1c4e9" if st.session_state.theme=="Purple" 
                    else "#fff176",
                "color": 
                    "#330000" if st.session_state.theme=="Red" 
                    else "#001f3f" if st.session_state.theme=="Blue" 
                    else "#0b3d0b" if st.session_state.theme=="Green" 
                    else "#663300" if st.session_state.theme=="Orange" 
                    else "#2e004f" if st.session_state.theme=="Purple" 
                    else "#665c00"
            }
        }
    )

# -------------------------------------------------------
# PAGE ROUTING
# -------------------------------------------------------
if selected == "Home":
    st.markdown("<div class='card'><h2>Welcome!</h2><p>This platform leverages AI to predict disasters and provide alerts, guidelines, and admin-managed communications.</p></div>", unsafe_allow_html=True)
    st.image("https://source.unsplash.com/1600x900/?disaster", use_container_width=True)
    st.markdown("<div class='card'><h3>Daily Safety Tip</h3><p>Always keep a portable emergency kit with essentials like water, snacks, and a flashlight.</p></div>", unsafe_allow_html=True)
    with st.expander("Emergency Preparedness Tips"):
        st.markdown("""
        - **Stay Informed:** Monitor local news and weather alerts.
        - **Plan Ahead:** Establish a family emergency plan.
        - **Prepare Essentials:** Stock up on food, water, and medications.
        - **Practice Drills:** Regularly rehearse evacuation routes.
        """)

    # Show quick location-based weather if logged in
    if st.session_state.logged_in:
        # fetch location
        conn = sqlite3.connect('dpah_database.db')
        c = conn.cursor()
        c.execute("SELECT location FROM users_new WHERE username=?", (st.session_state.username,))
        row = c.fetchone()
        conn.close()
        if row:
            location = row[0]
            st.write(f"**Your Location:** {location}")
            st.write("Simulated weather forecast for your area:")
            simulated_weather_forecast(location)

elif selected == "User Onboarding":
    if not st.session_state.logged_in:
        mode = st.radio("Choose Option", ["Login", "Sign Up"])
        if mode == "Sign Up":
            st.markdown("<div class='card'><h2>Create Account</h2></div>", unsafe_allow_html=True)
            name = st.text_input("Full Name", help="Enter your full name.")
            username = st.text_input("Username", help="Choose a unique username.")
            password = st.text_input("Password", type="password", help="Enter a secure password.")
            st.write("Select your current location on the map:")
            location = location_selector()
            st.write(f"Selected Location: {location}")
            if st.button("Create Account"):
                if name and username and password:
                    create_user(username, password, name, location)
                else:
                    st.warning("Please fill in all fields.")
        else:
            st.markdown("<div class='card'><h2>User Login</h2></div>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                ok, is_adm = authenticate_user(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.is_admin = is_adm
                    st.success("Logged in successfully!")
                else:
                    st.error("Invalid credentials or not approved yet!", icon="🚨")
    else:
        st.markdown(f"<div class='card'><h2>Welcome, {st.session_state.username}!</h2></div>", unsafe_allow_html=True)
        st.write("Your account is already logged in.")

elif selected == "Disaster Prediction":
    if not st.session_state.logged_in:
        st.error("Please log in to access Disaster Prediction.")
    else:
        st.markdown("<div class='card'><h2>Disaster Prediction</h2><p>Enter the required attributes for prediction.</p></div>", unsafe_allow_html=True)
        disaster = st.selectbox("Select Disaster Type", ["Earthquake", "Flood", "Cyclone", "Wildfire", "Tsunami", "Heatwave"])
        input_df = None
        model = None
        # EXACT same attribute sections as before, for each disaster:
        if disaster == "Earthquake":
            st.subheader("Earthquake Attributes")
            col1, col2, col3 = st.columns(3)
            with col1:
                magnitude = st.number_input("Magnitude", min_value=3.0, max_value=9.0, value=5.5, step=0.1)
                shaking_duration = st.number_input("Shaking Duration (sec)", min_value=0, max_value=120, value=30)
                ground_acceleration = st.number_input("Ground Acceleration (g)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
                peak_ground_velocity = st.number_input("Peak Ground Velocity (cm/sec)", min_value=0, max_value=200, value=50)
            with col2:
                depth = st.number_input("Depth (km)", min_value=5.0, max_value=300.0, value=50.0)
                fault_distance = st.number_input("Fault Distance (km)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
                building_density = st.number_input("Building Density (bld/km²)", min_value=0, max_value=10000, value=300)
                liquefaction_index = st.number_input("Soil Liquefaction Index", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            with col3:
                distance = st.number_input("Distance to City (km)", min_value=0.0, max_value=100.0, value=20.0)
                pop_density = st.number_input("Population Density (/km²)", min_value=0, max_value=10000, value=500)
                seismic_gap = st.number_input("Seismic Gap (years)", min_value=0, max_value=500, value=50)
                building_age = st.number_input("Avg. Building Age (years)", min_value=0, max_value=200, value=30)
            features = {
                "magnitude": magnitude,
                "shaking_duration": shaking_duration,
                "ground_acceleration": ground_acceleration,
                "peak_ground_velocity": peak_ground_velocity,
                "depth": depth,
                "fault_distance": fault_distance,
                "building_density": building_density,
                "liquefaction_index": liquefaction_index,
                "distance": distance,
                "pop_density": pop_density,
                "seismic_gap": seismic_gap,
                "building_age": building_age
            }
            input_df = pd.DataFrame([features])
            model = load_model("earthquake_best_model.pkl")

        elif disaster == "Flood":
            st.subheader("Flood Attributes")
            col1, col2, col3 = st.columns(3)
            with col1:
                rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=600.0, value=250.0)
                land_slope = st.number_input("Land Slope (°)", min_value=0.0, max_value=45.0, value=10.0)
                drainage_efficiency = st.number_input("Drainage Efficiency (0-1)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
                urbanization_rate = st.number_input("Urbanization Rate (%)", min_value=0, max_value=100, value=70)
            with col2:
                river_level = st.number_input("River Level (m)", min_value=0.0, max_value=50.0, value=15.0)
                soil_saturation = st.number_input("Soil Saturation (%)", min_value=0.0, max_value=100.0, value=85.0)
                impervious_area = st.number_input("Impervious Area (%)", min_value=0, max_value=100, value=60)
                catchment_area = st.number_input("Catchment Area (km²)", min_value=0.0, max_value=500.0, value=50.0)
            with col3:
                flood_duration = st.number_input("Flood Duration (hours)", min_value=0, max_value=240, value=48)
                water_retention = st.number_input("Water Retention (mm)", min_value=0.0, max_value=200.0, value=80.0)
                vegetation_index = st.number_input("Vegetation Index (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                soil_type_index = st.number_input("Soil Type Index (0-1)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
            features = {
                "rainfall": rainfall,
                "land_slope": land_slope,
                "drainage_efficiency": drainage_efficiency,
                "urbanization_rate": urbanization_rate,
                "river_level": river_level,
                "soil_saturation": soil_saturation,
                "impervious_area": impervious_area,
                "catchment_area": catchment_area,
                "flood_duration": flood_duration,
                "water_retention": water_retention,
                "vegetation_index": vegetation_index,
                "soil_type_index": soil_type_index
            }
            input_df = pd.DataFrame([features])
            model = load_model("flood_model.pkl")

        elif disaster == "Cyclone":
            st.subheader("Cyclone Attributes")
            col1, col2, col3 = st.columns(3)
            with col1:
                wind_speed = st.number_input("Wind Speed (km/h)", min_value=50, max_value=300, value=180)
                storm_surge = st.number_input("Storm Surge (m)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
                central_pressure = st.number_input("Central Pressure (hPa)", min_value=800, max_value=1100, value=930)
                eye_diameter = st.number_input("Eye Diameter (km)", min_value=0.0, max_value=100.0, value=25.0)
            with col2:
                precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=400.0, value=100.0)
                visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=20.0, value=5.0)
                pressure_variation = st.number_input("Pressure Variation (hPa)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
                cyclone_duration = st.number_input("Duration (hours)", min_value=0, max_value=240, value=72)
            with col3:
                sea_surface_temp = st.number_input("Sea Surface Temp (°C)", min_value=20.0, max_value=35.0, value=28.0, step=0.1)
                wave_height = st.number_input("Wave Height (m)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
                ocean_current = st.number_input("Ocean Current (km/h)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
                humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=80)
            features = {
                "wind_speed": wind_speed,
                "storm_surge": storm_surge,
                "central_pressure": central_pressure,
                "eye_diameter": eye_diameter,
                "precipitation": precipitation,
                "visibility": visibility,
                "pressure_variation": pressure_variation,
                "cyclone_duration": cyclone_duration,
                "sea_surface_temp": sea_surface_temp,
                "wave_height": wave_height,
                "ocean_current": ocean_current,
                "humidity": humidity
            }
            input_df = pd.DataFrame([features])
            model = load_model("cyclone_best_model.pkl")

        elif disaster == "Wildfire":
            st.subheader("Wildfire Attributes")
            col1, col2, col3 = st.columns(3)
            with col1:
                temperature = st.number_input("Temperature (°C)", value=30.0)
                humidity = st.number_input("Humidity (%)", value=40)
                wind_speed = st.number_input("Wind Speed (km/h)", value=20)
                vegetation_density = st.number_input("Vegetation Density (%)", value=50)
            with col2:
                fuel_moisture = st.number_input("Fuel Moisture Content (%)", value=20)
                drought_index = st.number_input("Drought Index", value=0.5, step=0.01)
                precipitation = st.number_input("Precipitation (mm)", value=0.0)
                air_quality = st.number_input("Air Quality Index", value=100)
            with col3:
                land_slope = st.number_input("Land Slope (°)", value=5.0)
                ignition_sources = st.number_input("Ignition Sources (count)", value=1)
            features = {
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "vegetation_density": vegetation_density,
                "fuel_moisture": fuel_moisture,
                "drought_index": drought_index,
                "precipitation": precipitation,
                "air_quality": air_quality,
                "land_slope": land_slope,
                "ignition_sources": ignition_sources
            }
            input_df = pd.DataFrame([features])
            model = load_model("wildfire_model.pkl")

        elif disaster == "Tsunami":
            st.subheader("Tsunami Attributes")
            col1, col2, col3 = st.columns(3)
            with col1:
                eq_magnitude = st.number_input("Earthquake Magnitude", value=7.0)
                seafloor_disp = st.number_input("Seafloor Displacement (m)", value=1.0, step=0.1)
                water_depth = st.number_input("Water Depth (m)", value=4000.0)
                distance_epicenter = st.number_input("Distance from Epicenter (km)", value=100.0)
            with col2:
                wave_height = st.number_input("Wave Height (m)", value=5.0, step=0.1)
                wave_period = st.number_input("Wave Period (sec)", value=15.0, step=0.1)
                coastal_elevation = st.number_input("Coastal Elevation (m)", value=10.0)
                tidal_range = st.number_input("Tidal Range (m)", value=2.0, step=0.1)
            with col3:
                shoreline_slope = st.number_input("Shoreline Slope (°)", value=5.0)
                warning_time = st.number_input("Warning Time (min)", value=30)
            features = {
                "eq_magnitude": eq_magnitude,
                "seafloor_disp": seafloor_disp,
                "water_depth": water_depth,
                "distance_epicenter": distance_epicenter,
                "wave_height": wave_height,
                "wave_period": wave_period,
                "coastal_elevation": coastal_elevation,
                "tidal_range": tidal_range,
                "shoreline_slope": shoreline_slope,
                "warning_time": warning_time
            }
            input_df = pd.DataFrame([features])
            model = load_model("tsunami_model.pkl")

        elif disaster == "Heatwave":
            st.subheader("Heatwave Attributes")
            col1, col2, col3 = st.columns(3)
            with col1:
                temperature = st.number_input("Temperature (°C)", value=40.0)
                humidity = st.number_input("Humidity (%)", value=30)
                heat_index = st.number_input("Heat Index", value=45.0)
                duration = st.number_input("Duration (days)", value=5)
            with col2:
                air_quality = st.number_input("Air Quality Index", value=120)
                wind_speed = st.number_input("Wind Speed (km/h)", value=10)
                uv_index = st.number_input("UV Index", value=8)
                drought_index = st.number_input("Drought Index", value=0.6, step=0.01)
            with col3:
                precipitation = st.number_input("Precipitation (mm)", value=0.0)
                night_temp = st.number_input("Nighttime Temperature (°C)", value=30.0)
            features = {
                "temperature": temperature,
                "humidity": humidity,
                "heat_index": heat_index,
                "duration": duration,
                "air_quality": air_quality,
                "wind_speed": wind_speed,
                "uv_index": uv_index,
                "drought_index": drought_index,
                "precipitation": precipitation,
                "night_temp": night_temp
            }
            input_df = pd.DataFrame([features])
            model = load_model("heatwave_model.pkl")

        if st.button("Predict Impact"):
            with st.spinner("Running prediction..."):
                if model is None:
                    st.error("Model not loaded. Please check the model file.")
                else:
                    pred, prob = predict_disaster(model, input_df)
                    st.markdown(f"<div class='card'><h3>Prediction: {pred}</h3><p>Probability of Harm: {prob:.2f}</p></div>", unsafe_allow_html=True)
                    alert, advice, support = prevention_info(disaster, pred, prob)
                    st.markdown(f"<div class='card'><h3>Alert Level: {alert}</h3><p>{advice}</p><p><strong>Support:</strong> {support}</p></div>", unsafe_allow_html=True)
                    log_history(st.session_state.username, disaster, pred, prob)
                    report = (f"Disaster: {disaster}\nPrediction: {pred}\nProbability: {prob:.2f}\nAlert: {alert}\nAdvice: {advice}\nSupport: {support}")
                    st.download_button("Download Report", report, file_name="disaster_report.txt", mime="text/plain")
                    if st.button("Send Report via Email"):
                        recipient = st.text_input("Enter Email Address:")
                        if recipient:
                            send_email(report, recipient)
                        else:
                            st.warning("Please enter a valid email address.")

elif selected == "Alert System":
    if not st.session_state.logged_in:
        st.error("Please log in to access the Alert System.")
    else:
        alert_system()

elif selected == "Advisor Chatbot":
    if not st.session_state.logged_in:
        st.error("Please log in to access the Advisor Chatbot.")
    else:
        st.markdown("<div class='card'><h2>Advisor Chatbot</h2><p>Ask any questions about disaster preparedness.</p></div>", unsafe_allow_html=True)
        user_name = st.session_state.get("username", "Guest")
        chatbot_html = get_chatbot_html(user_name)
        components.html(chatbot_html, height=750)

elif selected == "Dashboard":
    if not st.session_state.logged_in:
        st.error("Please log in to access the Dashboard.")
    else:
        st.markdown("<div class='card'><h2>Prediction Dashboard</h2><p>Review your past disaster predictions and trends.</p></div>", unsafe_allow_html=True)
        try:
            conn = sqlite3.connect('dpah_database.db')
            df_history = pd.read_sql_query(
                "SELECT disaster, prediction, probability, timestamp FROM history_new WHERE username=?",
                conn, params=(st.session_state.username,))
            conn.close()
            if df_history.empty:
                st.info("No history available.")
            else:
                st.dataframe(df_history)
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                st.markdown("### Prediction Probability Trend")
                chart = alt.Chart(df_history).mark_line(point=True).encode(
                    x='timestamp:T',
                    y='probability:Q',
                    color='disaster:N',
                    tooltip=['timestamp:T', 'probability:Q', 'disaster:N']
                ).properties(width=700, height=400)
                st.altair_chart(chart, use_container_width=True)
                st.markdown("### Breakdown by Disaster Type")
                breakdown = df_history.groupby('disaster')['prediction'].count().reset_index()
                fig, ax = plt.subplots()
                ax.pie(breakdown['prediction'], labels=breakdown['disaster'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
        except Exception as e:
            st.error("Error loading dashboard: " + str(e))

elif selected == "Emergency Resources":
    if not st.session_state.logged_in:
        st.error("Please log in to access Emergency Resources.")
    else:
        emergency_resources()

elif selected == "Prevention Guidelines":
    if not st.session_state.logged_in:
        st.error("Please log in to access Prevention Guidelines.")
    else:
        prevention_guidelines()

elif selected == "Weather":
    if not st.session_state.logged_in:
        st.error("Please log in to access Weather.")
    else:
        st.markdown("<div class='card'><h2>Weather Forecast</h2></div>", unsafe_allow_html=True)
        conn = sqlite3.connect('dpah_database.db')
        c = conn.cursor()
        c.execute("SELECT location FROM users_new WHERE username=?", (st.session_state.username,))
        row = c.fetchone()
        conn.close()
        if row:
            location = row[0]
            st.write(f"**Your Location:** {location}")
            simulated_weather_forecast(location)

elif selected == "FAQ":
    faq_section()

elif selected == "About":
    about_section()

elif selected == "Contact Us":
    contact_section()

elif selected == "Messages & Notifications":
    # This menu is available only to non-admin logged-in users
    if not st.session_state.logged_in or st.session_state.is_admin:
        st.error("Access denied or not applicable.")
    else:
        st.markdown("<div class='card'><h2>Your Notifications</h2></div>", unsafe_allow_html=True)
        notifs = get_user_notifications(st.session_state.username)
        if not notifs:
            st.info("No notifications found.")
        else:
            for nid, sender, recipient, msg, ts in notifs:
                st.markdown(f"**From:** {sender}, **Message:** {msg}  \n*{ts}*")
                st.markdown("---")

        st.markdown("<div class='card'><h2>Chat with Admin</h2></div>", unsafe_allow_html=True)
        chat_data = get_chat_messages(st.session_state.username, "admin")
        if chat_data:
            for (from_u, to_u, msg, ts) in chat_data:
                st.markdown(f"**[{from_u}]** {msg}  \n*{ts}*")
        else:
            st.info("No chat messages yet.")

        new_msg = st.text_input("Your Message to Admin")
        if st.button("Send"):
            if new_msg.strip():
                send_message(st.session_state.username, "admin", new_msg)
                st.success("Message sent! Please reload or revisit this page to see updates.")

elif selected == "Admin Panel":
    if not st.session_state.logged_in or not st.session_state.is_admin:
        st.error("Access denied. Admins only.")
    else:
        admin_choice = st.radio("Admin Panel Options", ["Manage Users", "Send Notifications", "User–Admin Chat"])
        if admin_choice == "Manage Users":
            st.subheader("Pending User Approvals")
            pending_users = get_pending_users()
            if pending_users:
                for usr, name, loc in pending_users:
                    st.write(f"**Username**: {usr}, **Name**: {name}, **Location**: {loc}")
                    colA, colB = st.columns(2)
                    with colA:
                        if st.button(f"Approve {usr}"):
                            approve_user(usr)
                    with colB:
                        if st.button(f"Remove {usr}"):
                            remove_user(usr)
            else:
                st.info("No pending sign-up requests.")

            st.subheader("Remove Existing Users")
            approved_users = get_all_approved_users()
            if approved_users:
                user_to_remove = st.selectbox("Select a user to remove", ["None"] + [u[0] for u in approved_users])
                if user_to_remove != "None":
                    if st.button("Remove Selected User"):
                        remove_user(user_to_remove)
            else:
                st.info("No approved users found.")

        elif admin_choice == "Send Notifications":
            st.subheader("Send Notifications")
            recipient_choice = st.radio("Choose notification recipient", ["All Users", "Specific User"])
            if recipient_choice == "All Users":
                message = st.text_area("Notification Message")
                if st.button("Send to ALL"):
                    if message.strip():
                        send_notification("admin", "ALL", message)
            else:
                user_list = get_all_approved_users()
                if user_list:
                    user_dict = {f"{u[0]} ({u[1]})": u[0] for u in user_list}
                    selected_recipient_label = st.selectbox("Select user", list(user_dict.keys()))
                    message = st.text_area("Notification Message")
                    if st.button("Send to Specific User"):
                        if message.strip():
                            send_notification("admin", user_dict[selected_recipient_label], message)
                else:
                    st.info("No approved users to send notifications.")

        elif admin_choice == "User–Admin Chat":
            st.subheader("Admin - Chat with Users")
            user_list = get_all_approved_users()
            if user_list:
                user_dict = {f"{u[0]} ({u[1]})": u[0] for u in user_list}
                selected_user_label = st.selectbox("Select user to chat with", list(user_dict.keys()))
                selected_user = user_dict[selected_user_label]
                st.write(f"Chat with **{selected_user}**")
                # Display chat messages
                chat_data = get_chat_messages("admin", selected_user)
                if chat_data:
                    for (from_u, to_u, msg, ts) in chat_data:
                        st.markdown(f"**[{from_u}]** {msg}  \n*{ts}*")
                else:
                    st.info("No messages yet.")

                new_msg = st.text_input("Your Message")
                if st.button("Send Message to User"):
                    if new_msg.strip():
                        send_message("admin", selected_user, new_msg)
                        st.success("Message sent! Reload or revisit to see new messages.")
            else:
                st.info("No approved users to chat with.")

elif selected == "Settings":
    st.markdown("<div class='card'><h2>Settings</h2></div>", unsafe_allow_html=True)
    st.subheader("UI Color Theme Settings")
    new_theme = st.radio("Select UI Color Theme", options=["Red", "Blue", "Green", "Orange", "Purple", "Yellow"],
                        index=["Red", "Blue", "Green", "Orange", "Purple", "Yellow"].index(st.session_state.theme) 
                            if st.session_state.theme in ["Red", "Blue", "Green", "Orange", "Purple", "Yellow"] else 0)
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.warning("Theme updated! Please reload the page to see changes.")

elif selected == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.is_admin = False
    st.success("Logged out successfully!")
    time.sleep(1)
    # If your Streamlit version supports it, you could do:
    # st.rerun()

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("<div class='footer'><p>© 2025 Disaster Prediction & Alert Hub. All rights reserved.</p></div>", unsafe_allow_html=True)
st.markdown("""
<div class="back-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'});">
    <i class="fa fa-arrow-up"></i>
</div>
""", unsafe_allow_html=True)
