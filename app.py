import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 수문 분석 시스템", layout="wide")
st.title("🌊 내성천 AI 유량 예측 및 시뮬레이션 시스템")

# 2. 리소스 로드
@st.cache_resource
def load_all_resources():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        # 파일 읽기 및 전처리
        r_df = pd.read_csv('rain_data.csv')
        f_df = pd.read_csv('flow_data.csv')
        r_df.columns = r_df.columns.str.lower().str.strip()
        f_df.columns = f_df.columns.str.lower().str.strip()
        
        # 숫자 강제 변환 (에러 방지)
        r_df['rf'] = pd.to_numeric(r_df['rf'], errors='coerce')
        f_df['fw'] = pd.to_numeric(f_df['fw'], errors='coerce')
        
        # 병합
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        df = df.dropna().sort_values('ymd')
        
        return model, scaler, df
    except Exception as e:
        st.error(f"❌ 파일 로드 실패: {e}")
        return None, None, None

model, scaler, data = load_all_resources()

# 3. 모델 고정 지표 (사이드바)
with st.sidebar:
    st.header("📊 모델 성능 지표 (전체)")
    st.metric("평균 오차율 (MAPE)", "7.98%")
    st.metric("결정계수 (R²)", "0.92")
    st.write("---")
    st.info("💡 2024년 이후 데이터는 과거 실측 패턴을 기반으로 LSTM 모델을 통해 시뮬레이션된 결과입니다.")

# --- 메인 화면 탭 구성 ---
if data is not None:
    tab1, tab2 = st.tabs(["🚀 장기 유량 시뮬레이션 (2026년까지)", "📅 과거 데이터 성능 검증 (달력)"])
    
    # ---------------------------------------------------------
    # [탭 1] 장기 시뮬레이션 (24년 데이터 반복 로직)
    # ---------------------------------------------------------
    with tab1:
        st.subheader("📍 2024년 이후 AI 장기 유량 패턴 시뮬레이션")
        
        if st.button("🚀 2026년 현재 시점까지 예측 실행", use_container_width=True):
            # 24년 데이터 반복 로직
            # 데이터의 마지막 날짜 (2024-02 경)
            last_real_date = data['ymd'].max()
            today_target = pd.Timestamp(datetime(2026, 3, 28))
            
            # 예측 대상 기간 설정
            prediction_dates = pd.date_range(start=last_real_date + timedelta(days=1), end=today_target, freq='D')
            full_data = data.copy()
