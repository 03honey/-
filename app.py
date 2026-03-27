import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) 실시간 AI 유량 예측 시스템")

# 2. 진짜 모델 및 스케일러 불러오기
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('naeseong_model.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"모델 파일을 찾을 수 없습니다: {e}")
        return None, None

model, scaler = load_resources()

# 3. 실시간 데이터 수집 함수
def get_live_data(base_date):
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    start_date = (base_date - timedelta(days=10)).strftime("%Y%m%d")
    end_date = base_date.strftime("%Y%m%d")
    
    try:
        r_url = f"http://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_date}/{end_date}.json"
        f_url = f"http://api.hrfco.go.kr/{api_key}/rfw/list/1D/2004680/{start_date}/{end_date}.json"
        
        r_res = requests.get(r_url).json().get('content', [])
        f_res = requests.get(f_url).json().get('content', [])
        
        if not r_res or not f_res: return None
        
        df_r = pd.DataFrame(r_res)[['ymd', 'rf']]
        df_f = pd.DataFrame(f_res)[['ymd', 'fw']]
        df = pd.merge(df_r, df_f, on='ymd').sort_values('ymd')
        df[['rf', 'fw']] = df[['rf', 'fw']].apply(pd.to_numeric, errors='coerce')
        return df.dropna().tail(7)
    except:
        return None

# 4. 화면 구성
st.sidebar.header("📊 모델 성능 지표")
st.sidebar.write("- **MAPE:** 7.98%")
st.sidebar.write("- **알고리즘:** LSTM")

selected_date = st.date_input("예측 기준일", datetime.now() - timedelta(days=1))
