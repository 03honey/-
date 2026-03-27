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
st.title("🌊 내성천 AI 유량 예측 시스템")

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
    except Exception as e:
        st.error(f"데이터 로드 에러: {e}")
        return None, None, None

model, scaler, data = load_all()

if data is not None:
    # 사이드바
    st.sidebar.header("📊 모델 성능")
    st.sidebar.metric("평균 오차율 (MAPE)", "7.98%")
    st.sidebar.metric("결정계수 (R²)", "0.92")

    # 오늘 날짜(2026-03-28) 기준
    today = datetime(2026, 3, 28)
    st.subheader("📅 분석 시점 선택")
    selected_date = st.date_input("조회 날짜", value=today, max_value=today)

    if st.button("🚀 AI 분석 실행", use_container_width=True):
        target_ts = pd.Timestamp(selected_date)
        last_real_date = data['ymd'].max()
        
        # 미래(2026년)는 마지막 7일치 사용
        if target_ts > last_real_date:
            history = data.tail(7)
        else:
            history = data[data['ymd'] <= target_ts].tail(7)

        if len(history) == 7:
            # 1. AI 예측
            input_vals = history[['rf', 'fw']].values
            inputs_scaled = scaler.transform(input_vals)
            pred_raw = model.predict(inputs_scaled.reshape(1, 7, 2), verbose=0)
            
            # 2. 역스케일링
            fw_idx = np.argmax(scaler.data_max_)
            dummy = np.zeros((1, 2))
            dummy[0, fw_idx] = pred_raw[0, 0]
            pred_val = scaler.inverse_transform(dummy)[0, fw_idx]
            
            # 3. 주간 강수량 및 수치 보정
            weekly_rain = history['rf'].sum()
            last_fw = history['fw'].iloc[-1]
            if weekly_rain < 1.0 and pred_val > last_fw * 1.1:
                pred_val = last_fw * 0.95 

            st.markdown("---")
            st.balloons()
            
            # 4. 결과 메트릭
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("분석 기준일", selected_date.strftime('%Y-%m-%d'))
            c2.metric("주간 강수량 합계", f"{weekly_rain:.1f} mm")
            c3.metric("3일 뒤 예측 유량", f"{pred_val:.3f} m³/s")
            
            # 실측 대조
            act_day = history['ymd'].iloc[-1] + timedelta(days=3)
            act_row = data[data['ymd'] == act_day]
            if not act_row.empty:
                act_val = act_row['fw'].values[0]
                mape = abs((act_val - pred_val) / act_val) * 100 if act_val != 0 else 0
                c4.metric("실제 관측 유량", f"{act_val:.3f}", delta=f"오차율 {mape:.2f}%")
            else:
                c4.info("실측 데이터 없음")

            # 5. 그래프
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, 8)), y=history['fw'].values, 
                                     mode='lines+markers', name='최근 7일', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=[10], y=[pred_val], mode='markers+text', 
                                     name='예측', text=[f"{pred_val:.2f}"], textposition="top center",
                                     marker=dict(size=12, color='red', symbol='star')))
            fig.update_layout(xaxis=dict(tickmode='
