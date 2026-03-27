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

# 2. 리소스 로드 (모델, 스케일러 및 파일 데이터 병합)
@st.cache_resource
def load_all_resources():
    try:
        # Keras 3 백엔드 설정
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        # 주헌님의 두 개 파일을 읽어와서 하나로 합칩니다.
        rain_df = pd.read_csv('rain_data.csv')
        flow_df = pd.read_csv('flow_data.csv')
        
        # ymd 컬럼을 기준으로 병합
        df = pd.merge(rain_df[['ymd', 'rf']], flow_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        df = df.sort_values('ymd').dropna()
        
        return model, scaler, df
    except Exception as e:
        st.error(f"❌ 파일 로드 중 오류 발생: {e}")
        return None, None, None

model, scaler, data = load_all_resources()

# 3. 사이드바 구성
with st.sidebar:
    st.header("📊 시스템 상태")
    if model and scaler:
        st.success("✅ AI 모델/스케일러 로드 완료")
    if data is not None:
        st.success(f"✅ 데이터 병합 완료 ({len(data)}일치)")
    st.info("이 시스템은 직접 업로드한 강우/유량 데이터를 기반으로 LSTM 분석을 수행합니다.")

# 4. 메인 분석 로직
if data is not None:
    # LSTM 7일 입력 + 3일 뒤 예측을 위한 날짜 범위 설정
    min_date = data['ymd'].min().to_pydatetime() + timedelta(days=7)
    max_date = data['ymd'].max().to_pydatetime() - timedelta(days=3)
    
    st.subheader("📅 시뮬레이션 시점 선택")
    selected_date = st.slider("분석 기준 날짜를 선택하세요", 
                              min_value=min_date, 
                              max_value=max_date, 
                              value=min_date,
                              format="YYYY-MM-DD")

    if st.button("🚀 선택한 날짜 기준 AI 유량 예측 시작", use_container_width=True):
        with st.spinner('AI가 과거 데이터를 분석하여 3일 뒤를 예측 중입니다...'):
            # 1) 입력 데이터 필터링 (선택일 기준 과거 7일)
            mask = (data['ymd'] <= pd.Timestamp(selected_date))
            recent_7days = data.loc[mask].tail(7)
            
            # 2) 전처리 (Scaler 적용)
            inputs = scaler.transform(recent_7days[['rf', 'fw']])
            inputs = inputs.reshape(1, 7, 2)
            
            # 3) AI 예측 실행
            pred_scaled = model.predict(inputs)
            
            # 4) 역스케일링 (m3/s 단위 복원)
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            # 결과 표시
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

            # 5) 시각화 (Plotly)
            st.subheader("📈 입력 데이터 추이 및 예측 포인트")
            fig = go.Figure()
            # 과거 7일 유량
            fig.add_trace(go.Scatter(x=recent_7days['ymd'], y=recent_7days['fw'], name='과거 유량', line=dict(color='#1f77b4', width=3)))
            # 예측 포인트
            fig.add_trace(go.Scatter(x=[pd.Timestamp(prediction_day)], y=[final_val], name='AI 예측값', 
                                     mode='markers+text', text=["AI 예측"], textposition="top center",
                                     marker=dict(size=15, color='red', symbol='star')))
            # 실제 포인트 (있는 경우)
            if not actual_row.empty:
                fig.add_trace(go.Scatter(x=[pd.Timestamp(prediction_day)], y=[actual_val], name='실제 관측값', 
                                         mode='markers', marker=dict(size=12, color='green', symbol='x')))
                
            fig.update_layout(template="plotly_white", hovermode="x unified", height=500)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.error("⚠️ 'rain_data.csv' 또는 'flow_data.csv' 파일을 찾을 수 없습니다. GitHub에 업로드해 주세요.")
