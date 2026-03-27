import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="내성천 유량 예측 시스템", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) AI 유량 예측 모델 시연")

@st.cache_resource
def load_all_resources():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        # 파일 읽기 및 컬럼명 강제 소문자화 (에러 방지)
        rain_df = pd.read_csv('rain_data.csv')
        flow_df = pd.read_csv('flow_data.csv')
        
        rain_df.columns = rain_df.columns.str.lower()
        flow_df.columns = flow_df.columns.str.lower()
        
        # 병합
        df = pd.merge(rain_df[['ymd', 'rf']], flow_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        df = df.sort_values('ymd').dropna()
        
        return model, scaler, df
    except Exception as e:
        st.error(f"❌ 파일 로드 중 오류 발생: {e}")
        return None, None, None

model, scaler, data = load_all_resources()

if data is not None:
    st.sidebar.success(f"✅ 데이터 로드 완료 ({len(data)}일치)")
    
    # 분석 가능 범위 설정
    min_date = data['ymd'].min().to_pydatetime() + timedelta(days=7)
    max_date = data['ymd'].max().to_pydatetime() - timedelta(days=3)
    
    st.subheader("📅 시뮬레이션 시점 선택")
    st.info("슬라이더를 움직여 분석하고 싶은 과거 날짜를 선택하세요.")
    selected_date = st.select_slider("분석 기준 날짜", 
                                    options=pd.date_range(min_date, max_date),
                                    format_func=lambda x: x.strftime('%Y-%m-%d'))

    if st.button("🚀 AI 유량 예측 실행", use_container_width=True):
        # 1) 입력 데이터 필터링
        mask = (data['ymd'] <= pd.Timestamp(selected_date))
        recent_7days = data.loc[mask].tail(7)
        
        if len(recent_7days) == 7:
            # 2) 전처리 및 예측
            inputs = scaler.transform(recent_7days[['rf', 'fw']])
            inputs = inputs.reshape(1, 7, 2)
            pred_scaled = model.predict(inputs)
            
            # 3) 역스케일링
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            prediction_day = selected_date + timedelta(days=3)
            actual_row = data[data['ymd'] == pd.Timestamp(prediction_day)]
            
            st.markdown("---")
            st.balloons()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"📅 {prediction_day.strftime('%Y-%m-%d')} AI 예측 유량", f"{final_val:.3f} m³/s")
            with col2:
                if not actual_row.empty:
                    actual_val = actual_row['fw'].values[0]
                    st.metric("📊 실제 관측 유량", f"{actual_val:.3f} m³/s", 
                              delta=f"오차: {abs(final_val-actual_val):.3f}")

            # 4) 그래프 시각화
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent_7days['ymd'], y=recent_7days['fw'], name='과거 유량', line=dict(color='#1f77b4', width=3)))
            fig.add_trace(go.Scatter(x=[pd.Timestamp(prediction_day)], y=[final_val], name='AI 예측값', 
                                     mode='markers+text', text=["AI 예측"], textposition="top center",
                                     marker=dict(size=15, color='red', symbol='star')))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("분석을 위한 데이터가 충분하지 않습니다.")
