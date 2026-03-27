import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="내성천 유량 장기 예측", page_icon="🌊", layout="wide")
st.title("🌊 내성천 AI 장기 미래 예측 시스템 (2026년 현재까지)")

@st.cache_resource
def load_all_resources():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        r_df = pd.read_csv('rain_data.csv')
        f_df = pd.read_csv('flow_data.csv')
        r_df.columns = r_df.columns.str.lower()
        f_df.columns = f_df.columns.str.lower()
        r_df['rf'] = pd.to_numeric(r_df['rf'], errors='coerce')
        f_df['fw'] = pd.to_numeric(f_df['fw'], errors='coerce')
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        return model, scaler, df.dropna().sort_values('ymd')
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None, None, None

model, scaler, data = load_all_resources()

# --- 미래 예측 핵심 로직 ---
def predict_until_today(model, scaler, last_data, target_date):
    current_df = last_data.copy()
    
    # 마지막 날짜부터 목표 날짜(오늘)까지 반복 예측
    while current_df['ymd'].max() < target_date:
        # 최근 7일 추출
        recent_7 = current_df.tail(7)
        inputs_scaled = scaler.transform(recent_7[['rf', 'fw']].values)
        
        # 미래 예측 (3일 뒤)
        pred_scaled = model.predict(inputs_scaled.reshape(1, 7, 2), verbose=0)
        
        # 역스케일링
        dummy = np.zeros((1, 2))
        dummy[0, 1] = pred_scaled[0, 0]
        pred_fw = scaler.inverse_transform(dummy)[0, 1]
        
        # 새로운 날짜 생성 (3일 간격)
        new_date = current_df['ymd'].max() + timedelta(days=3)
        # 미래 강우량은 시나리오상 0으로 가정 (또는 평균치)
        new_row = pd.DataFrame({'ymd': [new_date], 'rf': [0.0], 'fw': [pred_fw]})
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        
    return current_df

# --- 화면 구성 ---
if data is not None:
    st.sidebar.info(f"데이터 최종일: {data['ymd'].max().strftime('%Y-%m-%d')}")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("📅 미래 예측 설정")
        # 오늘 날짜를 기본 목표로 설정
        today_date = datetime.now()
        target_goal = st.date_input("예측 목표 날짜", value=today_date)
        
        run_btn = st.button("🚀 장기 미래 예측 실행", use_container_width=True)

    if run_btn:
        with st.spinner('AI가 2024년부터 오늘까지 시간을 달리고 있습니다...'):
            # 2024년 2월부터 오늘까지 재귀적 예측 수행
            future_data = predict_until_today(model, scaler, data, pd.Timestamp(target_goal))
            
            # 결과 표시
            final_pred = future_data.iloc[-1]
            st.success(f"### ✅ {final_pred['ymd'].strftime('%Y-%m-%d')} 예상 유량: **{final_pred['fw']:.3f} m³/s**")
            
            # 그래프 (전체 흐름 보여주기)
            fig = go.Figure()
            # 기존 데이터 (파란색)
            fig.add_trace(go.Scatter(x=data['ymd'].tail(100), y=data['fw'].tail(100), name='과거 실측(최근 100일)', line=dict(color='blue')))
            # AI가 만든 미래 (빨간색 점선)
            predicted_part = future_data[future_data['ymd'] > data['ymd'].max()]
            fig.add_trace(go.Scatter(x=predicted_part['ymd'], y=predicted_part['fw'], name='AI 장기 예측', line=dict(color='red', dash='dash')))
            
            fig.update_layout(title="2024년 이후 AI의 유량 시뮬레이션 결과", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.balloons()
