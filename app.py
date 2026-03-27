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

# 2. 진짜 모델 및 스케일러 불러오기 (버전 오류 방지 옵션 추가)
@st.cache_resource
def load_resources():
    try:
        # compile=False: 최신 Keras 버전 설정 오류를 무시하고 모델 구조와 가중치만 로드
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ 모델 로드 실패: {e}")
        return None, None

model, scaler = load_resources()

# 3. 실시간 데이터 수집 함수
def get_live_data(base_date):
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    # 과거 7일치 데이터를 가져오기 위해 10일 전부터 조회
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

# 4. 사이드바 구성
st.sidebar.header("📊 모델 분석 정보")
st.sidebar.metric("검증 오차율 (MAPE)", "7.98%")
st.sidebar.info("LSTM 모델을 사용하여 3일 뒤의 유량을 실시간으로 예측합니다.")

# 5. 메인 화면
selected_date = st.date_input("예측 기준일 설정", datetime.now() - timedelta(days=1))

if st.button('🚀 실시간 AI 유량 예측 시작'):
    if model is not None and scaler is not None:
        with st.spinner('실제 AI 모델이 계산 중입니다...'):
            recent_data = get_live_data(selected_date)
            
            if recent_data is not None and len(recent_data) == 7:
                # 1) 데이터 전처리 (스케일링)
                inputs = scaler.transform(recent_data[['rf', 'fw']])
                inputs = inputs.reshape(1, 7, 2)
                
                # 2) 진짜 모델 예측
                pred_scaled = model.predict(inputs)
                
                # 3) 역스케일링 (원래 단위로 복원)
                dummy = np.zeros((1, 2))
                dummy[0, 1] = pred_scaled[0, 0]
                final_val = scaler.inverse_transform(dummy)[0, 1]
                
                # 결과 표시
                target_date = selected_date + timedelta(days=3)
                st.markdown("---")
                st.success(f"✅ 분석 완료! **{target_date.strftime('%Y-%m-%d')}** 예상 유량은 **{final_val:.3f} $m^3/s$** 입니다.")
                
                # 6. 시각화 (그래프)
                st.subheader("📈 입력 데이터 추이 (최근 7일)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['fw'], name='실측 유량 ($m^3/s$)', line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['rf'], name='실측 강우 (mm)', line=dict(color='#d62728'), yaxis="y2"))
                
                fig.update_layout(
                    yaxis=dict(title="유량 ($m^3/s$)"),
                    yaxis2=dict(title="강우 (mm)", overlaying="y", side="right"),
                    legend=dict(x=0.01, y=0.99),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ API에서 과거 7일치 데이터를 불러오지 못했습니다. 날짜를 변경해 보세요.")
    else:
        st.error("❌ 모델이 정상적으로 로드되지 않았습니다. 깃허브 파일을 확인해 주세요.")
