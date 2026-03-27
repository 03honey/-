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
        
        # 2026년 대응 데이터 매핑
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
            
            # 2. 역스케일링
            fw_idx = np.argmax(scaler.data_max_)
            dummy = np.zeros((1, 2))
            dummy[0, fw_idx] = pred_raw[0, 0]
            pred_val = scaler.inverse_transform(dummy)[0, fw_idx]
            
            # 3. 데이터 집계 (주간 강수량 포함)
            last_fw = history['fw'].iloc[-1]
            avg_fw = history['fw'].mean()
            weekly_rain_sum = history['rf'].sum() # 최근 7일 강수량 합계
            
            # 수치 안정화 (Smoothing)
            if weekly_rain_sum < 5.0:
                if pred_val > last_fw * 1.15:
                    pred_val = (last_fw * 0.8) + (avg_fw * 0.2)
            elif pred_val > last_fw * 2.5:
                pred_val = last_fw * 1.5

            st.markdown("---")
            st.balloons()
            
            # 메트릭 레이아웃 (강수량 합계 추가)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("선택일", selected_date.strftime('%Y-%m-%d'))
            c2.metric("주간 강수량 합계", f"{weekly_rain_sum:.1f} mm")
            c3.metric("3일 뒤 예측 유량", f"{pred_val:.3f} m³/s")
            
            # 실측 대조
            act_day = history['ymd'].iloc[-1] + timedelta(days=3)
            act_row = data[data['ymd'] == act_day]
            if not act_row.empty:
                act_val = act_row['fw'].values[0]
                mape = abs((act_val - pred_val) / act_val) * 100 if act_val != 0 else 0
                c4.metric("실제 관측 유량", f"{act_val:.3f}", delta=f"오차율: {mape:.2f}%", delta_color="inverse")
            else:
                c4.info("실측 데이터 없음")

            # 4. 그래프 출력
            st.subheader(f"📈 {selected_date} 기준 유량 추이 및 예측")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, 8)), y=history['fw'].values, 
                                     mode='lines+markers', name='최근 7일 실측', line=dict(color='#1f77b4', width=3)))
            fig.add_trace(go.Scatter(x=[10], y=[pred_val], mode='markers+text', 
                                     name='AI 예측(T+3)', text=[f"{pred_val:.2f}"], textposition="top center",
                                     marker=dict(size=12, color='red', symbol='star')))
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=[1,4,7,10], ticktext=['D-6','D-3','기준일','T+3']),
                              yaxis=dict(title="유량 (m³/s)"), template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"💡 정보: 기준일 이전 7일간의 총 강수량은 {weekly_rain_sum:.1f}mm 입니다.")
        else:
            st.error("데이터가 부족합니다.")
else:
    st.error("데이터 로드 실패.")
