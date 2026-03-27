import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) 실시간 AI 유량 예측 시스템")

# 2. 리소스 로드 (가장 단순한 방식으로 수정)
@st.cache_resource
def load_all():
    try:
        # 경로 함수 안 쓰고 파일명만 직접 입력
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        # 에러가 나면 화면에 에러 내용을 직접 뿌립니다.
        st.sidebar.error(f"⚠️ 로드 에러: {e}")
        return None, None

model, scaler = load_all()

# 3. 실시간 데이터 수집
def get_live_data(base_date):
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    start_date = (base_date - timedelta(days=10)).strftime("%Y%m%d")
    end_date = base_date.strftime("%Y%m%d")
    
    try:
        r_url = f"http://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_date}/{end_date}.json"
        f_url = f"http://api.hrfco.go.kr/{api_key}/rfw/list/1D/2004680/{start_date}/{end_date}.json"
        
        r_res = requests.get(r_url).json().get('content', [])
        f_res = requests.get(f_url).json().get('content', [])
        
        if not r_res or not f_res: return None
        
        df_r = pd.DataFrame(r_res)[['ymd', 'rf']]
        df_f = pd.DataFrame(f_res)[['ymd', 'fw']]
        df = pd.merge(df_r, df_f, on='ymd').sort_values('ymd')
        df[['rf', 'fw']] = df[['rf', 'fw']].apply(pd.to_numeric, errors='coerce')
        return df.dropna().tail(7)
    except:
        return None

# 4. 화면 구성
st.sidebar.header("📊 분석 시스템 상태")
if model and scaler:
    st.sidebar.success("✅ AI 모델 정상 작동 중")
else:
    st.sidebar.error("❌ 파일 로드 실패")

selected_date = st.date_input("예측 기준일", datetime.now() - timedelta(days=1))

if st.button('🚀 실시간 AI 유량 예측 시작'):
    if model and scaler:
        with st.spinner('AI 분석 중...'):
            recent_data = get_live_data(selected_date)
            
            if recent_data is not None and len(recent_data) == 7:
                # 데이터 전처리
                inputs = scaler.transform(recent_data[['rf', 'fw']])
                inputs = inputs.reshape(1, 7, 2)
                
                # 예측
                pred_scaled = model.predict(inputs)
                
                # 역스케일링
                dummy = np.zeros((1, 2))
                dummy[0, 1] = pred_scaled[0, 0]
                final_val = scaler.inverse_transform(dummy)[0, 1]
                
                target_date = selected_date + timedelta(days=3)
                st.success(f"✅ {target_date.strftime('%Y-%m-%d')} 예상 유량: **{final_val:.3f} $m^3/s$**")
                
                # 그래프
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['fw'], name='유량'))
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['rf'], name='강우', yaxis="y2"))
                fig.update_layout(yaxis2=dict(overlaying="y", side="right"), template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("데이터가 부족합니다. 다른 날짜를 선택해 보세요.")
