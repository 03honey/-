import streamlit as st
import pandas as pd
import numpy as np
import os

# ⚠️ 최신 모델 로드를 위해 백엔드 설정
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import joblib
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) 실시간 AI 유량 예측 시스템")

# 2. 모델 및 스케일러 로드 (오류 방어형)
@st.cache_resource
def load_resources():
    try:
        # compile=False: 버전 차이로 인한 설정값 오류 무시
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.sidebar.error(f"❌ 로드 실패: {e}")
        return None, None

model, scaler = load_resources()

# 3. 데이터 수집 함수 (최근 7일치)
def get_data(base_date):
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    start = (base_date - timedelta(days=10)).strftime("%Y%m%d")
    end = base_date.strftime("%Y%m%d")
    try:
        r_url = f"http://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start}/{end}.json"
        f_url = f"http://api.hrfco.go.kr/{api_key}/rfw/list/1D/2004680/{start}/{end}.json"
        r_json = requests.get(r_url).json().get('content', [])
        f_json = requests.get(f_url).json().get('content', [])
        if not r_json or not f_json: return None
        df_r = pd.DataFrame(r_json)[['ymd', 'rf']]
        df_f = pd.DataFrame(f_json)[['ymd', 'fw']]
        df = pd.merge(df_r, df_f, on='ymd').sort_values('ymd')
        df[['rf', 'fw']] = df[['rf', 'fw']].apply(pd.to_numeric, errors='coerce')
        return df.dropna().tail(7)
    except: return None

# 4. UI 구성
st.sidebar.header("📊 시스템 상태")
if model and scaler:
    st.sidebar.success("✅ AI 모델 작동 중")
else:
    st.sidebar.error("❌ 모델 로드 실패")

date_input = st.date_input("예측 기준일", datetime.now() - timedelta(days=1))

if st.button('🚀 예측 시작', use_container_width=True):
    if model and scaler:
        with st.spinner('데이터 분석 중...'):
            df = get_data(date_input)
            if df is not None and len(df) == 7:
                # 전처리 및 예측
                scaled = scaler.transform(df[['rf', 'fw']])
                pred = model.predict(scaled.reshape(1, 7, 2))
                # 역스케일링
                dummy = np.zeros((1, 2))
                dummy[0, 1] = pred[0, 0]
                val = scaler.inverse_transform(dummy)[0, 1]
                
                st.success(f"📅 {(date_input + timedelta(days=3)).strftime('%Y-%m-%d')} 예상 유량: **{val:.3f} $m^3/s$**")
                st.balloons()
                
                # 그래프
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['ymd'], y=df['fw'], name='유량'))
                fig.add_trace(go.Scatter(x=df['ymd'], y=df['rf'], name='강우', yaxis="y2"))
                fig.update_layout(yaxis2=dict(overlaying="y", side="right"), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("데이터가 부족합니다.")
