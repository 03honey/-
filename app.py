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
st.title("🌊 내성천(회룡교) AI 유량 예측 모델 시연")

# 2. 리소스 로드
@st.cache_resource
def load_all():
    try:
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        # 주헌님이 학습에 사용한 '무결성 검토 완료' 데이터 로드
        df = pd.read_csv('refined_data.csv') 
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        return model, scaler, df
    except Exception as e:
        return None, None, None

model, scaler, data = load_all()

# 3. 사이드바
with st.sidebar:
    st.header("📊 분석 모델 정보")
    if model: st.success("✅ LSTM 모델 로드 완료")
    if data is not None: st.success("✅ 학습/검증 데이터 로드 완료")
    st.info("본 시스템은 API 지연 문제를 해결하기 위해 '검토 완료된 데이터셋'을 기반으로 작동합니다.")

# 4. 메인 화면
if data is not None:
    st.subheader("📅 예측 대상 기간 선택")
    # 데이터셋에 있는 날짜 범위 내에서 선택하게 함
    min_date = data['ymd'].min().to_pydatetime() + timedelta(days=7)
    max_date = data['ymd'].max().to_pydatetime() - timedelta(days=3)
    
    selected_date = st.slider("날짜를 선택하세요", 
                              min_value=min_date, 
                              max_value=max_date, 
                              value=min_date,
                              format="YYYY-MM-DD")

    if st.button("🚀 선택한 시점의 AI 유량 예측 실행", use_container_width=True):
        # 1) 선택한 날짜 기준 과거 7일 데이터 추출
        mask = (data['ymd'] <= pd.Timestamp(selected_date))
        recent_7days = data.loc[mask].tail(7)
        
        if len(recent_7days) == 7:
            # 2) 예측 실행
            inputs = scaler.transform(recent_7days[['rf', 'fw']])
            inputs = inputs.reshape(1, 7, 2)
            pred_scaled = model.predict(inputs)
            
            # 3) 역스케일링
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            # 4) 실제 결과값과 비교 (데이터셋에 실제 값이 있다면)
            target_date = selected_date + timedelta(days=3)
            actual_row = data[data['ymd'] == pd.Timestamp(target_date)]
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"📅 {target_date.strftime('%Y-%m-%d')} AI 예측 유량", f"{final_val:.3f} m³/s")
            with col2:
                if not actual_row.empty:
                    actual_val = actual_row['fw'].values[0]
                    st.metric("📊 실제 관측 유량", f"{actual_val:.3f} m³/s", 
                              delta=f"오차: {abs(final_val-actual_val):.3f}")

            st.balloons()
            
            # 5) 그래프 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent_7days['ymd'], y=recent_7days['fw'], name='유량(과거 7일)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=[pd.Timestamp(target_date)], y=[final_val], name='AI 예측값', mode='markers', marker=dict(size=12, color='red')))
            fig.update_layout(title="과거 7일 데이터 및 3일 뒤 예측 포인트", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.error("CSV 데이터를 찾을 수 없습니다. 'refined_data.csv' 파일을 업로드해 주세요.")
