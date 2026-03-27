import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측 시스템", layout="wide")
st.title("🌊 내성천 AI 유량 장기 예측 시스템")

# 2. 리소스 로드
@st.cache_resource
def load_all_resources():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        # 파일 읽기 및 숫자 변환
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
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None, None, None

model, scaler, data = load_all_resources()

# 3. 모델 성능 지표 (사이드바)
with st.sidebar:
    st.header("📊 모델 성능 지표")
    st.metric("평균 오차율 (MAPE)", "7.98%")
    st.metric("결정계수 (R²)", "0.92")
    st.write("---")
    st.info("2024년 2월 이후의 데이터는 모델의 예측값을 재입력하는 재귀적 방식으로 산출됩니다.")

# 4. 재귀적 예측 함수
def run_recursive_prediction(model, scaler, start_df, target_date):
    result_df = start_df.copy()
    
    # 마지막 데이터 시점부터 목표 날짜까지 3일 간격으로 예측 반복
    while result_df['ymd'].max() < pd.Timestamp(target_date):
        # 최근 7일 데이터 추출
        recent_7 = result_df.tail(7)[['rf', 'fw']].values
        inputs_scaled = scaler.transform(recent_7)
        
        # 모델 예측
        pred_scaled = model.predict(inputs_scaled.reshape(1, 7, 2), verbose=0)
        
        # 역스케일링
        dummy = np.zeros((1, 2))
        dummy[0, 1] = pred_scaled[0, 0]
        pred_fw = scaler.inverse_transform(dummy)[0, 1]
        
        # 날짜 추가 (미래 강우는 0으로 고정)
        next_date = result_df['ymd'].max() + timedelta(days=3)
        new_row = pd.DataFrame({'ymd': [next_date], 'rf': [0.0], 'fw': [pred_fw]})
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        
    return result_df

# 5. 메인 화면 출력
if data is not None:
    st.write(f"최종 실측 데이터 날짜: {data['ymd'].max().strftime('%Y-%m-%d')}")
    
    # 2026년 오늘 날짜까지 예측
    today_target = datetime(2026, 3, 28)
    
    if st.button("🚀 2026년 현재 시점까지 예측 실행", use_container_width=True):
        full_data = run_recursive_prediction(model, scaler, data, today_target)
        
        today_pred = full_data.iloc[-1]
        st.success(f"### ✅ {today_pred['ymd'].strftime('%Y-%m-%d')} 기준 예상 유량: **{today_pred['fw']:.3f}** m³/s")
        
        # 그래프 생성
        fig = go.Figure()
        # 과거 실측 (파란색)
        fig.add_trace(go.Scatter(x=data['ymd'].tail(100), y=data['fw'].tail(100), name='과거 실측치', line=dict(color='blue')))
        # 미래 예측 (빨간색 점선)
        future_part = full_data[full_data['ymd'] > data['ymd'].max()]
        fig.add_trace(go.Scatter(x=future_part['ymd'], y=future_part['fw'], name='AI 장기 예측치', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title="연속 예측을 통한 유량 시뮬레이션 결과 (2024-2026)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.balloons()
