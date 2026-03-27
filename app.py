import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 수문 분석 시스템", layout="wide")

# 2. 데이터 및 모델 로드
@st.cache_resource
def load_all():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        # 파일 읽기
        r_df = pd.read_csv('rain_data.csv')
        f_df = pd.read_csv('flow_data.csv')
        r_df.columns = r_df.columns.str.lower().str.strip()
        f_df.columns = f_df.columns.str.lower().str.strip()
        
        # 숫자 변환
        r_df['rf'] = pd.to_numeric(r_df['rf'], errors='coerce')
        f_df['fw'] = pd.to_numeric(f_df['fw'], errors='coerce')
        f_df['wl'] = pd.to_numeric(f_df['wl'], errors='coerce')
        
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw', 'wl']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        df = df.dropna().sort_values('ymd')
        return model, scaler, df
    except Exception as e:
        return None, None, None

model, scaler, data = load_all()

# --- 메인 로직 시작 ---
if data is not None:
    # 데이터의 마지막 날짜 (2024년 2월 경)
    last_real_date = data['ymd'].max()
    today = datetime(2026, 3, 28) # 오늘 기준

    # 3. 장기 예측 함수 (24년 데이터를 재료로 오늘까지 생성)
    @st.cache_data
    def get_long_term_prediction(_model, _scaler, _base_data, target_date):
        current_df = _base_data.copy()
        while current_df['ymd'].max() < target_date:
            recent_7 = current_df.tail(7)[['rf', 'fw']].values
            inputs_scaled = _scaler.transform(recent_7)
            pred_scaled = _model.predict(inputs_scaled.reshape(1, 7, 2), verbose=0)
            
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            pred_fw = _scaler.inverse_transform(dummy)[0, 1]
            
            next_date = current_df['ymd'].max() + timedelta(days=3)
            # 수위(wl)는 유량(fw)에 비례한다고 가정 (단순 시각화용)
            new_row = pd.DataFrame({'ymd': [next_date], 'rf': [0.0], 'fw': [pred_fw], 'wl': [pred_fw*0.15]})
            current_df = pd.concat([current_df, new_row], ignore_index=True)
        return current_df

    # 전체 데이터 생성
    full_data = get_long_term_prediction(model, scaler, data, today)
    latest_info = full_data.iloc[-1]
    prev_info = full_data.iloc[-2]

    # --- 대시보드 UI 레이아웃 ---
    st.write(f"🕒 **화룡교 분석 설정** | 최신 실측 기반 AI 확장 완료 ({today.strftime('%Y-%m-%d')})")
    st.title("🌊 내성천(회룡교) 빅데이터 기반 AI 유량 예측 시스템")

    # [섹션 1: 수문 현황]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재 유량 (예측)", f"{latest_info['fw']:.
