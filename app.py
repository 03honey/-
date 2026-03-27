import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="내성천 수문 정밀 분석", layout="wide")
st.title("🌊 내성천 AI 유량 예측 (물리적 제약 모드)")

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

if data is not None:
    today = datetime(2026, 3, 28)
    selected_date = st.date_input("조회 날짜", value=today, max_value=today)

    if st.button("🚀 유량 분석 및 폭주 방지 실행", use_container_width=True):
        target_ts = pd.Timestamp(selected_date)
        last_real_date = data['ymd'].max()
        
        if target_ts > last_real_date:
            try: s_date = datetime(2024, target_ts.month, target_ts.day)
            except: s_date = datetime(2024, 2, 28)
            history = data[data['ymd'] <= pd.Timestamp(s_date)].tail(7)
        else:
            history = data[data['ymd'] <= target_ts].tail(7)

        if len(history) == 7:
            # 1. 원본 AI 예측
            input_vals = history[['rf', 'fw']].values
            inputs_scaled = scaler.transform(input_vals)
            pred_raw = model.predict(inputs_scaled.reshape(1, 7, 2), verbose=0)
            
            # 2. 역스케일링 (정밀 인덱스 추출)
            fw_idx = np.argmax(scaler.data_max_)
            dummy = np.zeros((1, 2))
            dummy[0, fw_idx] = pred_raw[0, 0]
            pred_val = scaler.inverse_transform(dummy)[0, fw_idx]
            
            # --- [핵심] 유량 폭주 방지(Smoothing) 로직 ---
            avg_fw = history['fw'].mean()
            last_fw = history['fw'].iloc[-1]
            max_history = history['fw'].max()
            
            # 비가 오지 않는데(최근 7일 강우합 < 5mm) 유량이 1.5배 이상 튀면 보정
            recent_rain = history['rf'].sum()
            if recent_rain < 5.0:
                # 무강우 시에는 유량이 서서히 줄어드는 것이 물리적 정상 (Recession)
                upper_bound = last_fw * 1.2 # 최대 20% 상승까지만 허용
                if pred_val > upper_bound:
                    # 예측값이 너무 높으면 최근 7일 평균과 마지막 값 사이로 강제 조정
                    pred_val = (last_fw * 0.7) + (avg_fw * 0.3) 
            
            # 어떤 경우에도 3일 만에 유량이 2배 이상 튀는 건 비정상으로 간주 (급변 방지)
            if pred_val > last_fw * 2.0 and recent_rain < 20.0:
                pred_val = last_fw * 1.1 
            # ------------------------------------------

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("분석일", selected_date.strftime('%Y-%m-%d'))
            c2.metric("3일 뒤 예측 유량", f"{pred_val:.3f} m³/s")
            
            act_day = history['ymd'].iloc[-1] + timedelta(days=3)
            act_row = data[data['ymd'] == act_day]
            if not act_row.empty:
                act_val = act_row['fw'].values[0]
                mape = abs((act_val - pred_val) / act_val) * 100 if act_val != 0 else 0
                c3.metric("실제 관측 유량", f"{act_val:.3f}", delta=f"오차율: {mape:.2f}%", delta_color="inverse")
            else:
                c3.info("실측 데이터 없음")

            # 그래프
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, 8)), y=history['fw'].values, 
                                     mode='lines+markers', name='최근 7일 실측', line=dict(color='#1f77b4', width=3)))
            fig.add_trace(go.Scatter(x=[10], y=[pred_val], mode='markers+text', 
                                     name='AI 예측(T+3)', text=[f"{pred_val:.2f}"], textposition="top center",
                                     marker=dict(size=12, color='red', symbol='star')))
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=[1,4,7,10], ticktext=['D-6','D-3','기준일','T+3']),
                              yaxis=dict(range=[min(history['fw'].min(), pred_val)*0.8, max(history['fw'].max(), pred_val)*1.2]),
                              template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
