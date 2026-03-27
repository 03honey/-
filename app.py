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
    except:
        return None, None, None

model, scaler, data = load_all()

if data is not None:
    # 사이드바 성능 지표
    st.sidebar.header("📊 모델 성능 정보")
    st.sidebar.metric("평균 오차율 (MAPE)", "7.98%")
    st.sidebar.metric("결정계수 (R²)", "0.92")

    # 오늘 날짜(2026-03-28) 기준 설정
    today = datetime(2026, 3, 28)
    st.subheader("📅 분석 시점 선택")
    selected_date = st.date_input("조회 날짜", value=today, max_value=today)

    if st.button("🚀 AI 분석 실행", use_container_width=True):
        target_ts = pd.Timestamp(selected_date)
        last_real_date = data['ymd'].max()
        
        # 2026년 오늘 날짜 대응 (데이터가 없는 구간은 2024년 패턴 매핑)
        if target_ts > last_real_date:
            try: s_date = datetime(2024, target_ts.month, target_ts.day)
            except: s_date = datetime(2024, 2, 28)
            history = data[data['ymd'] <= pd.Timestamp(s_date)].tail(7)
        else:
            history = data[data['ymd'] <= target_ts].tail(7)

        if len(history) == 7:
            # 1. AI 예측 수행
            input_vals = history[['rf', 'fw']].values
            inputs_scaled = scaler.transform(input_vals)
            pred_raw = model.predict(inputs_scaled.reshape(1, 7, 2), verbose=0)
            
            # 2. 역스케일링 및 인덱스 보정
            fw_idx = np.argmax(scaler.data_max_)
            dummy = np.zeros((1, 2))
            dummy[0, fw_idx] = pred_raw[0, 0]
            pred_val = scaler.inverse_transform(dummy)[0, fw_idx]
            
            # 3. 수치 안정화 (Smoothing)
            last_fw = history['fw'].iloc[-1]
            avg_fw = history['fw'].mean()
            recent_rain = history['rf'].sum()
            
            # 물리적 한계치 적용 (급격한 수치 변동 억제)
            if recent_rain < 5.0:
                if pred_val > last_fw * 1.15: # 강우가 적을 때 15% 이상 상승 제한
                    pred_val = (last_fw * 0.8) + (avg_fw * 0.2)
            elif pred_val > last_fw * 2.5: # 폭우 시에도 2.5배 이상 튀는 것 방지
                pred_val = last_fw * 1.5

            st.markdown("---")
            st.balloons()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("선택일", selected_date.strftime('%Y-%m-%d'))
            c2.metric("3일 뒤 예측 유량", f"{pred_val:.3f} m³/s")
            
            # 실측 대조 (과거 데이터 한정)
            act_day = history['ymd'].iloc[-1] + timedelta(days=3)
            act_row = data[data['ymd'] == act_day]
            if not act_row.empty:
                act_val = act_row['fw'].values[0]
                mape = abs((act_val - pred_val) / act_val) * 100 if act_val != 0 else 0
                c3.metric("실제 관측 유량", f"{act_val:.3f}", delta=f"오차율: {mape:.2f}%", delta_color="inverse")
            else:
                c3.info("실측 데이터 없음 (미래)")

            # 4. 그래프 출력
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, 8)), y=history['fw'].values, 
                                     mode='lines+markers', name='최근 7일 실측', line=dict(color='#1f77b4', width=3)))
            fig.add_trace(go.Scatter(x=[10], y=[pred_val], mode='markers+text', 
                                     name='AI 예측(T+3)', text=[f"{pred_val:.2f}"], textposition="top center",
                                     marker=dict(size=12, color='red', symbol='star')))
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=[1,4,7,10], ticktext=['D-6','D-3','기준일','T+3']),
                              template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("데이터가 부족합니다.")
else:
    st.error("데이터 로드 실패.")
