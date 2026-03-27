import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측", page_icon="🌊")
st.title("🌊 내성천(회룡교) $t+3$ 유량 예측")

# 2. 모델 및 스케일러 불러오기 (캐싱 처리로 속도 업!)
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('naeseong_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_resources()

# 3. 실시간 API 데이터 수집 함수
def get_live_data():
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
    
    try:
        # 강우량(고평교), 유량(회룡교) API 호출
        r_url = f"http://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_date}/{end_date}.json"
        f_url = f"http://api.hrfco.go.kr/{api_key}/rfw/list/1D/2004680/{start_date}/{end_date}.json"
        
        r_res = requests.get(r_url).json()['content']
        f_res = requests.get(f_url).json()['content']
        
        df_r = pd.DataFrame(r_res)[['ymd', 'rf']]
        df_f = pd.DataFrame(f_res)[['ymd', 'fw']]
        
        df = pd.merge(df_r, df_f, on='ymd').sort_values('ymd')
        df[['rf', 'fw']] = df[['rf', 'fw']].apply(pd.to_numeric, errors='coerce')
        return df.dropna().tail(7) # 마지막 7일치 데이터 반환
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류 발생: {e}")
        return None

# 4. 웹 화면 구성
st.info("실시간 API를 통해 데이터를 수동 입력 없이 자동으로 분석합니다.")

if st.button('🚀 3일 뒤 유량 예측하기'):
    with st.spinner('데이터 수집 및 AI 분석 중...'):
        recent_data = get_live_data()
        
        if recent_data is not None:
            # 데이터 전처리 (0~1 스케일링)
            inputs = scaler.transform(recent_data[['rf', 'fw']])
            inputs = inputs.reshape(1, 7, 2) # 모델 입력 형태 맞추기
            
            # 예측
            pred_scaled = model.predict(inputs)
            
            # 실제 유량 값으로 복원 (역스케일링)
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            # 결과 표시
            st.success(f"📅 오늘 데이터 기반, 3일 뒤 예상 유량은 **{final_val:.3f} $m^3/s$** 입니다.")
            st.line_chart(recent_data.set_index('ymd'))
            