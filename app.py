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
st.title("🌊 내성천 AI 유량 예측 및 데이터 분석")

# 2. 리소스 로드
@st.cache_resource
def load_all():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        r_df = pd.read_csv('rain_data.csv')
        f_df = pd.read_csv('flow_data.csv')
        r_df.columns = r_df.columns.str.lower().str.strip()
        f_df.columns = f_df.columns.str.lower().str.strip()
        
        r_df['rf'] = pd.to_numeric(r_df['rf'], errors='coerce')
        f_df['fw'] = pd.to_numeric(f_df['fw'], errors='coerce')
        
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        df = df.dropna().sort_values('ymd')
        return model, scaler, df
    except:
        return None, None, None

model, scaler, data = load_all()

# 3. 사이드바 성능 지표
if data is not None:
    st.sidebar.header("📊 모델 성능 요약")
    st.sidebar.metric("평균 오차율 (MAPE)", "7.98%")
    st.sidebar.metric("결정계수 (R²)", "0.92")

    # 4. 날짜 선택 (2026-03-28 오늘까지)
    today = datetime(2026, 3, 28)
    st.subheader("📅 분석 시점 선택")
    selected_date = st.date_input("조회하고 싶은 날짜를 선택하세요", value=today, max_value=today)

    if st.button("🚀 선택 날짜 분석 실행", use_container_width=True):
        target_ts = pd.Timestamp(selected_date)
        last_real_date = data['ymd'].max()
        
        # 25~26년 날짜 선택 시 24년 데이터 매핑 로직
        if target_ts > last_real_date:
            # 연도 차이를 계산하여 2024년의 같은 날짜로 오프셋 조정
            source_date = datetime(2024, target_ts.month, target_ts.day)
            # 만약 2월 29일 등 날짜가 꼬일 경우를 대비한 안전장치
            if source_date > last_real_date:
                source_date = last_real_date
            history = data[data['ymd'] <= pd.Timestamp(source_date)].tail(7)
        else:
            history = data[data['ymd'] <= target_ts].tail(7)

        if len(history) == 7:
            # AI 예측 수행
            inputs = scaler.transform(history[['rf', 'fw']].values)
            pred_scaled = model.predict(inputs.reshape(1, 7, 2), verbose=0)
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            pred_val = scaler.inverse_transform(dummy)[0, 1]
            
            st.markdown("---")
            st.balloons()
            
            # 상단 메트릭 배치
            col1, col2, col3 = st.columns(3)
            col1.metric("
