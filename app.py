import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="내성천 유량 예측", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) 실시간 AI 유량 예측")

# --- 1. 진짜 모델 및 스케일러 불러오기 ---
@st.cache_resource
def load_all():
    # 파일이 깃허브에 있어야 합니다!
    model = tf.keras.models.load_model('naeseong_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_all()
    st.sidebar.success("✅ AI 모델 로드 완료")
except Exception as e:
    st.sidebar.error(f"❌ 모델 로드 실패: {e}")

# --- 2. 실시간 데이터 가져오기 (날짜별) ---
def get_data(target_date):
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    # 예측 기준일로부터 과거 7일치 데이터가 필요함 (LSTM 입력용)
    start_date = (target_date - timedelta(days=7)).strftime("%Y%m%d")
    end_date = target_date.strftime("%Y%m%d")
    
    r_url = f"http://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_date}/{end_date}.json"
    f_url = f"http://api.hrfco.go.kr/{api_key}/rfw/list/1D/2004680/{start_date}/{end_date}.json"
    
    r_res = requests.get(r_url).json().get('content', [])
    f_res = requests.get(f_url).json().get('content', [])
    
    if not r_res or not f_res: return None
    
    df_r = pd.DataFrame(r_res)[['ymd', 'rf']]
    df_f = pd.DataFrame(f_res)[['ymd', 'fw']]
    df = pd.merge(df_r, df_f, on='ymd').sort_values('ymd')
    df[['rf', 'fw']] = df[['rf', 'fw']].apply(pd.to_numeric)
    return df.tail(7)

# --- 3. UI 및 예측 ---
selected_date = st.date_input("예측 기준일 (오늘 기준)", datetime.now() - timedelta(days=1))

if st.button('🚀 AI 실시간 분석 실행'):
    with st.spinner('실제 AI 모델이 계산 중입니다...'):
        data = get_data(selected_date)
        
        if data is not None and len(data) == 7:
            # 데이터 전처리 (0~1 스케일링)
            inputs = scaler.transform(data[['rf', 'fw']])
            inputs = inputs.reshape(1, 7, 2)
            
            # 진짜 모델 예측
            pred_scaled = model.predict(inputs)
            
            # 역스케일링 (실제 값으로 복원)
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            st.success(f"✅ AI 분석 완료! **{(selected_date + timedelta(days=3))}** 예상 유량: **{final_val:.3f} $m^3/s$**")
            
            # 날짜에 따른 MAPE(오차율) 계산은 실제값이 있어야 가능하므로 
            # 여기서는 모델의 고유 성능 지표인 7.98%를 공식적으로 표기합니다.
            st.metric("모델 검증 오차율 (MAPE)", "7.98%")
        else:
            st.warning("해당 날짜에 충분한 API 데이터가 없습니다.")
