import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측 시스템", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) AI 유량 예측 및 성능 검증 시스템")

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
        r_df.columns = r_df.columns.str.lower()
        f_df.columns = f_df.columns.str.lower()
        
        # 숫자 강제 변환 (에러 방지 핵심)
        r_df['rf'] = pd.to_numeric(r_df['rf'], errors='coerce')
        f_df['fw'] = pd.to_numeric(f_df['fw'], errors='coerce')
        
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        df = df.dropna().sort_values('ymd')
        
        return model, scaler, df
    except Exception as e:
        st.error(f"❌ 파일 로드 중 오류 발생: {e}")
        return None, None, None

model, scaler, data = load_all_resources()

# 3. 사이드바: 모델 고정 지표
with st.sidebar:
    st.header("📊 모델 성능 지표 (전체)")
    st.metric("평균 오차율 (MAPE)", "7.98%")
    st.metric("결정계수 (R²)", "0.92")
    st.write("---")
    st.info("💡 과거 데이터를 조회하면 해당 시점의 오차율을 계산합니다.")

# 4. 메인 탭 구성
tab1, tab2 = st.tabs(["🚀 데이터 기반 미래 예측", "📈 과거 분석 (MAPE/R 검증)"])

# [탭 1] 미래 예측
with tab1:
    if data is not None:
        last_date = data['ymd'].max()
        target_date = last_date + timedelta(days=3)
        st.subheader(f"📍 보유 데이터 이후 3일 뒤 예측 ({target_date.strftime('%Y-%m-%d')})")
        
        if st.button("🔮 미래 유량 예측 실행", key="future_btn"):
            recent_7 = data.tail(7)
            input_data = recent_7[['rf', 'fw']].values
            inputs_scaled = scaler.transform(input_data)
            pred_scaled = model.predict(inputs_scaled.reshape(1, 7, 2))
            
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            st.balloons()
            st.success(f"### ✅ {target_date.strftime('%Y-%m-%d')} 예상 유량: **{final_val:.3f}** m³/s")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent_7['ymd'], y=recent_7['fw'], name='현재 유량', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=[target_date], y=[final_val], name='AI 예측(3일 후)', 
                                     mode='markers+text', text=["예측"], textposition="top center",
                                     marker=dict(size=15, color='red', symbol='star')))
            st.plotly_chart(fig, use_container_width=True)

# [탭 2] 과거 검증
with tab2:
    st.subheader("🧐 과거 데이터 성능 검증")
    if data is not None:
        min_d = data['ymd'].min() + timedelta(days=7)
        max_d = data['ymd'].max() - timedelta(days=3)
        selected_date = st.slider("검증할 날짜 선택", min_value=min_d.to_pydatetime(), max_value=max_d.to_pydatetime(), format="YYYY-MM-DD")

        if st.button("🔎 분석 및 오차 산출", key="test_btn"):
            mask = (data['ymd'] <= pd.Timestamp(selected_date))
            history = data.loc[mask].tail(7)
            prediction_day = pd.Timestamp(selected_date) + timedelta(days=3)
            actual_row = data[data['ymd'] == prediction_day]
            
            if not actual_row.empty:
                input_data = history[['rf', 'fw']].values
                inputs_scaled = scaler.transform(input_data)
                pred_scaled = model.predict(inputs_scaled.reshape(1, 7, 2))
                dummy = np.zeros((1, 2))
                dummy[0, 1] = pred_scaled[0, 0]
                pred_val = scaler.inverse_transform(dummy)[0, 1]
                actual_val = actual_row['fw'].values[0]
                
                mape = abs((actual_val - pred_val) / actual_val) * 100 if actual_val != 0 else 0
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("AI 예측값", f"{pred_val:.3f} m³/s")
                col2.metric("실제 관측값", f"{actual_val:.3f} m³/s")
                col3.metric("오차율 (MAPE)", f"{mape:.2f}%")
                
                st.write("**상관계수 (R):** 0.96 (전체 상관도 기준)")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history['ymd'], y=history['fw'], name='과거 7일 실측', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=[prediction_day], y=[pred_val], name='AI 예측', mode='markers+text', text=["예측"], textposition="top center", marker=dict(size=12, color='red', symbol='star')))
                fig.add_trace(go.Scatter(x=[prediction_day], y=[actual_val], name='실제 실측', mode='markers', marker=dict(size=12, color='green', symbol='x')))
                fig.update_layout(title=f"{prediction_day.strftime('%Y-%m-%d')} 검증 (오차율: {mape:.2f}%)", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("비교할 실측 데이터가 부족합니다.")
