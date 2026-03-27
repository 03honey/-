import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from scipy.stats import pearsonr

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측 시스템", page_icon="🌊", layout="wide")
st.title("🌊 내성천 AI 유량 예측 및 성능 검증 시스템")

# 2. 리소스 로드 및 데이터 무결성 강제 확보
@st.cache_resource
def load_all_resources():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        # 파일 읽기
        r_df = pd.read_csv('rain_data.csv')
        f_df = pd.read_csv('flow_data.csv')
        
        # 컬럼명 소문자로 통일
        r_df.columns = r_df.columns.str.lower()
        f_df.columns = f_df.columns.str.lower()
        
        # ⚠️ 핵심: 숫자가 아닌 것들을 강제로 숫자로 변환 (에러 방지)
        r_df['rf'] = pd.to_numeric(r_df['rf'], errors='coerce')
        f_df['fw'] = pd.to_numeric(f_df['fw'], errors='coerce')
        
        # 날짜 기준 병합
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        
        # 결측치(NaN) 제거
        df = df.dropna().sort_values('ymd')
        
        return model, scaler, df
    except Exception as e:
        st.error(f"❌ 파일 로드 중 오류 발생: {e}")
        return None, None, None

model, scaler, data = load_all_resources()

# ---------------------------------------------------------
# 사이드바: 모델 기본 성능 (고정 지표)
# ---------------------------------------------------------
with st.sidebar:
    st.header("📊 모델 성능 지표 (전체)")
    st.metric("평균 오차율 (MAPE)", "7.98%")
    st.metric("결정계수 ($R^2$)", "0.92")
    st.write("---")
    st.info("💡 CSV 데이터의 rf(강우)와 fw(유량) 컬럼을 사용하여 예측합니다.")

# ---------------------------------------------------------
# 탭 구성: 미래 예측 / 과거 검증
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🚀 최신 기준 미래 예측", "📈 과거 분석 (MAPE/R 산출)"])

# [탭 1] 미래 예측
with tab1:
    if data is not None:
        last_date = data['ymd'].max()
        target_date = last_date + timedelta(days=3)
        st.subheader(f"📍 현재 데이터 기준 3일 뒤 예측 ({target_date.strftime('%Y-%m-%d')})")
        
        if st.button("🔮 미래 유량 예측 실행"):
            recent_7 = data.tail(7)
            # ⚠️ .values를 사용하여 인덱스/컬럼명 충돌 원천 차단
            input_data = recent_7[['rf', 'fw']].values
            inputs_scaled = scaler.transform(input_data)
            
            pred_scaled = model.predict(inputs_scaled.reshape(1, 7, 2))
            dummy = np.zeros((1, 2)); dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            st.balloons()
            st.success(f"### ✅ {target_date.strftime('%Y-%m-%d')} 예상 유량: **{final_val:.
