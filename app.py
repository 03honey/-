import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from sklearn.metrics import r2_score

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="내성천 유량 예측 시스템", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) AI 유량 예측 및 성능 검증 시스템")

# 2. 모델 및 데이터 로드 (API 제거 버전)
@st.cache_resource
def load_all_resources():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        # 파일 읽기 및 컬럼명 소문자 통일
        r_df = pd.read_csv('rain_data.csv')
        f_df = pd.read_csv('flow_data.csv')
        r_df.columns = r_df.columns.str.lower()
        f_df.columns = f_df.columns.str.lower()
        
        # ymd 기준으로 병합
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        df = df.sort_values('ymd').dropna()
        
        return model, scaler, df
    except Exception as e:
        st.error(f"❌ 파일 로드 중 오류 발생: {e}")
        return None, None, None

model, scaler, data = load_all_resources()

# ---------------------------------------------------------
# 사이드바: 모델의 전체 성능 지표 (과제 요건)
# ---------------------------------------------------------
with st.sidebar:
    st.header("📊 모델 성능 지표 (전체)")
    st.metric("평균 오차율 (MAPE)", "7.98%")
    st.metric("결정계수 ($R^2$)", "0.92")
    st.write("---")
    st.info("💡 이 시스템은 업로드된 최신 수문 데이터를 기반으로 3일 뒤의 유량을 예측합니다.")

# ---------------------------------------------------------
# 메인 화면: 탭 구성 (미래 예측 / 과거 검증)
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🚀 최신 데이터 기반 미래 예측", "📈 과거 데이터 검증 및 오차 분석"])

# [탭 1] 미래 예측 (보유한 데이터 중 가장 최신 시점 기준)
with tab1:
    if data is not None:
        last_date = data['ymd'].max()
        target_date = last_date + timedelta(days=3)
        
        st.subheader(f"📍 현재 데이터 기준 미래 예측 ({target_date.strftime('%Y-%m-%d')})")
        st.write(f"보유 중인 가장 최신 데이터 시점: **{last_date.strftime('%Y-%m-%d')}**")
        
        if st.button("🔮 3일 뒤 유량 예측 실행"):
            # 최근 7일 데이터 추출
            recent_7 = data.tail(7)
            
            # 예측 로직
            inputs = scaler.transform(recent_7[['rf', 'fw']])
            pred_scaled = model.predict(inputs.reshape(1, 7, 2))
            
            dummy = np.zeros((1, 2)); dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            st.balloons()
            st.success(f"### ✅ {target_date.strftime('%Y-%m-%d')} 예상 유량: **{final_val:.3f} m³/s**")
            
            # 예측 그래프
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent_7['ymd'], y=recent_7['fw'], name='현재 유량', line=dict(color='#1f77b4', width=3)))
            fig.add_trace(go.Scatter(x=[target_date], y=[final_val], name='AI 예측(3일 후)', 
                                     mode='markers+text', text=["예측"], textposition="top center",
                                     marker=dict(size=15, color='red', symbol='star')))
            fig.update_layout(title="현재 시점 유량 및 미래 예측점", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# [탭 2] 과거 데이터 검증 (오차율, R 지표 포함)
with tab2:
    st.subheader("🧐 과거 시점별 모델 성능 검증")
    if data is not None:
        # 분석 가능 범위 설정
        min_d = data['ymd'].min() + timedelta(days=7)
        max_d = data['ymd'].max() - timedelta(days=3)
        
        selected_date = st.slider("검증할 기준 날짜를 선택하세요", 
                                  min_value=min_d.to_pydatetime(), 
                                  max_value=max_d.to_pydatetime(),
                                  format="YYYY-MM-DD")

        if st.button("🔎 분석 및 오차 계산 실행"):
            # 1) 입력 데이터(7일)와 실제 정답(3일 뒤) 추출
            mask = (data['ymd'] <= pd.Timestamp(selected_date))
            history = data.loc[mask].tail(7)
            
            prediction_day = pd.Timestamp(selected_date) + timedelta(days=3)
            actual_row = data[data['ymd'] == prediction_day]
            
            if not actual_row.empty:
                # 2) 예측 실행
                inputs = scaler.transform(history[['rf', 'fw']])
                pred_scaled = model.predict(inputs.reshape(1, 7, 2))
                dummy = np.zeros((1, 2)); dummy[0, 1] = pred_scaled[0, 0]
                pred_val = scaler.inverse_transform(dummy)[0, 1]
                actual_val = actual_row['fw'].values[0]
                
                # 3) 지표 계산
                # MAPE (절대평균오차율) - 한 점에 대해서는 절대 오차율
                mape = abs((actual_val - pred_val) / actual_val) * 100 if actual_val != 0 else 0
                # 상관도(R)를 보여주기 위해 주변 데이터 30일을 가져와서 계산 (선택사항)
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("AI 예측 유량", f"{pred_val:.3f} m³/s")
                col2.metric("실제 관측 유량", f"{actual_val:.3f} m³/s")
                col3.metric("개별 오차율 (Error %)", f"{mape:.2f}%", delta=f"{pred_val-actual_val:.3f}", delta_color="inverse")
                
                # 4) 그래프 시각화
