import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) 실시간 AI 유량 예측 시스템")

# 2. 진짜 모델 및 스케일러 불러오기 (경로 및 버전 오류 방지)
@st.cache_resource
def load_resources():
    # 현재 실행되는 파일(app.py)의 위치를 기준으로 절대 경로 설정
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'naeseong_model.h5')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(model_path):
            st.error(f"❌ 서버에서 'naeseong_model.h5' 파일을 찾을 수 없습니다. (경로: {model_path})")
            return None, None
        if not os.path.exists(scaler_path):
            st.error(f"❌ 서버에서 'scaler.pkl' 파일을 찾을 수 없습니다. (경로: {scaler_path})")
            return None, None
            
        # 모델 로드 (compile=False로 버전 차이 무시)
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ 시스템 로드 실패: {e}")
        return None, None

model, scaler = load_resources()

# 3. 실시간 데이터 수집 함수
def get_live_data(base_date):
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    # LSTM 입력에 필요한 과거 7일치를 확보하기 위해 10일 전부터 데이터 요청
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
st.sidebar.metric("모델 검증 정확도", "92.02%") # 100 - 7.98%
st.sidebar.write("**평균 오차율 (MAPE):** 7.98%")
st.sidebar.info("상류(고평교) 강우와 하류(회룡교) 유량의 상관관계를 학습한 LSTM 모델입니다.")

# 5. 메인 화면 구성
selected_date = st.date_input("예측 기준일 설정", datetime.now() - timedelta(days=1))

if st.button('🚀 실시간 AI 유량 예측 시작', use_container_width=True):
    if model is not None and scaler is not None:
        with st.spinner('실제 AI 모델이 수문 데이터를 분석 중입니다...'):
            recent_data = get_live_data(selected_date)
            
            if recent_data is not None and len(recent_data) == 7:
                # 1) 데이터 전처리 (Scaler 적용)
                inputs = scaler.transform(recent_data[['rf', 'fw']])
                inputs = inputs.reshape(1, 7, 2)
                
                # 2) 진짜 모델 예측
                pred_scaled = model.predict(inputs)
                
                # 3) 역스케일링 (실제 유량 단위로 복원)
                # 스케일러가 2개 변수(rf, fw) 기준이므로 더미 데이터 활용
                dummy = np.zeros((1, 2))
                dummy[0, 1] = pred_scaled[0, 0]
                final_val = scaler.inverse_transform(dummy)[0, 1]
                
                # 결과 안내
                target_date = selected_date + timedelta(days=3)
                st.markdown("---")
                st.balloons() # 성공 축하 풍선
                st.success(f"✅ 분석 완료! **{target_date.strftime('%Y-%m-%d')}** 예상 유량은 약 **{final_val:.3f} $m^3/s$** 입니다.")
                
                # 6. 시각화 (Plotly 그래프)
                st.subheader("📈 입력 데이터 분석 (최근 7일 추이)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['fw'], name='실측 유량 ($m^3/s$)', line=dict(color='#1f77b4', width=3)))
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['rf'], name='실측 강우 (mm)', line=dict(color='#d62728', width=3), yaxis="y2"))
                
                fig.update_layout(
                    title=f"{selected_date.strftime('%Y-%m-%d')} 기준 과거 데이터 분석 결과",
                    yaxis=dict(title="유량 ($m^3/s$)"),
                    yaxis2=dict(title="강우 (mm)", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 해당 날짜의 실시간 수문 데이터를 충분히 확보하지 못했습니다. (7일치 데이터 필요)")
    else:
        st.error("❌ 시스템 구성 파일(모델/스케일러) 로드에 실패했습니다. 깃허브 저장소를 확인해 주세요.")
