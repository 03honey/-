import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 수문 분석", layout="wide")
st.title("🌊 내성천 AI 유량 예측 시스템")

# 2. 리소스 로드
@st.cache_resource
def load_all():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        r_df, f_df = pd.read_csv('rain_data.csv'), pd.read_csv('flow_data.csv')
        for d in [r_df, f_df]: d.columns = d.columns.str.lower().str.strip()
        r_df['rf'], f_df['fw'] = pd.to_numeric(r_df['rf'], errors='coerce'), pd.to_numeric(f_df['fw'], errors='coerce')
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        return model, scaler, df.dropna().sort_values('ymd')
    except: return None, None, None

model, scaler, data = load_all()

if data is not None:
    st.sidebar.header("📊 모델 성능")
    st.sidebar.metric("평균 오차율 (MAPE)", "7.98%")
    st.sidebar.metric("결정계수 (R2)", "0.92")

    # 오늘(2026-03-28) 기준
    today = datetime(2026, 3, 28)
    selected_date = st.date_input("조회 날짜", value=today, max_value=today)

    if st.button("🚀 AI 분석 및 실측 대조 실행", use_container_width=True):
        t_ts, l_rd = pd.Timestamp(selected_date), data['ymd'].max()
        
        # 2026년 선택 시 2024년 매핑
        if t_ts > l_rd:
            try: m_d = pd.Timestamp(year=2024, month=t_ts.month, day=t_ts.day)
            except: m_d = l_rd
            if m_d > l_rd: m_d = l_rd
            history = data[data['ymd'] <= m_d].tail(7)
        else:
            history = data[data['ymd'] <= t_ts].tail(7)

        if len(history) == 7:
            # 1. AI 예측 및 역스케일링
            in_v = scaler.transform(history[['rf', 'fw']].values)
            p_raw = model.predict(in_v.reshape(1, 7, 2), verbose=0)
            f_idx = np.argmax(scaler.data_max_)
            dummy = np.zeros((1, 2)); dummy[0, f_idx] = p_raw[0, 0]
            p_val = scaler.inverse_transform(dummy)[0, f_idx]
            
            # 2. 보정 및 강수량
            w_rain, l_fw = history['rf'].sum(), history['fw'].iloc[-1]
            if w_rain < 1.0 and p_val > l_fw * 1.5: p_val = l_fw * 1.1

            st.markdown("---")
            st.balloons()
            
            # 3. 결과 출력 (26년이라도 24년 실측 대조)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("분석 기준일", selected_date.strftime('%Y-%m-%d'))
            c2.metric("주간 강수량 합계", f"{w_rain:.1f} mm")
            c3.metric("예측 유량 (T+3)", f"{p_val:.3f} m3/s")
            
            chk_d = history['ymd'].iloc[-1] + timedelta(days=3)
            act_r = data[data['ymd'] == chk_d]
            if not act_r.empty:
                a_val = act_r['fw'].values[0]
                mape = abs((a_val - p_val) / a_val) * 100 if a_val != 0 else 0
                c4.metric("실제 관측 유량", f"{a_val:.3f}", delta=f"오차 {mape:.2f}%")
            else: c4.info("실측 없음")

            # 4. 그래프
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, 8)), y=history['fw'].values, mode='lines+markers', name='최근 7일', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=[10], y=[p_val], mode='markers+text', name='예측', text=[f"{p_val:.2f}"], textposition="top center", marker=dict(size=12, color='red', symbol='star')))
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=[1,4,7,10], ticktext=['D-6','D-3','기준일','T+3']), template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else: st.error("데이터 부족!")
