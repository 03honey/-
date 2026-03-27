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

# 2. 리소스 로드 (Keras 3 전용 세팅)
@st.cache_resource
def load_all():
    try:
        # Keras 3 백엔드를 tensorflow로 고정 (버전 충돌 방지)
        os.environ["KERAS_BACKEND"] = "tensorflow"
        
        # 모델 로드: compile=False로 설정하여 최신 버전의 특수한 설정값 무시
        # 만약 이래도 안되면 최후의 수단으로 custom_objects를 비웁니다.
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        # 사이드바에 구체적인 에러 메시지 표시
        st.sidebar.error(f"⚠️ 모델 로드 에러: {e}")
        return None, None

model, scaler = load_all()

# 3. 실시간 데이터 수집 함수 (강우+유량)
def get_live_data(base_date):
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    # LSTM 입력 윈도우(7일)를 위해 넉넉히 10일 전부터 가져옴
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

# 4. 사이드바 상태 표시
st.sidebar.header("📊 시스템 상태")
if model and scaler:
    st.sidebar.success("✅ AI 모델 로드 완료")
    st.sidebar.write("**예측 모델:** LSTM (Long Short-Term Memory)")
    st.sidebar.write("**최종 오차율 (MAPE):** 7.98%")
else:
    st.sidebar.error("❌ 모델 로드 대기 중/실패")

# 5. 메인 UI 및 예측 로직
selected_date = st.date_input("예측 기준일 (오늘 또는 과거)", datetime.now() - timedelta(days=1))

if st.button('🚀 실시간 AI 데이터 분석 및 3일 뒤 예측', use_container_width=True):
    if model and scaler:
        with st.spinner('실시간 API 데이터 수집 및 AI 분석 중...'):
            recent_data = get_live_data(selected_date)
            
            if recent_data is not None and len(recent_data) == 7:
                # 데이터 전처리 (정규화)
                inputs = scaler.transform(recent_data[['rf', 'fw']])
                inputs = inputs.reshape(1, 7, 2)
                
                # AI 예측 실행
                pred_scaled = model.predict(inputs)
                
                # 역정규화 (실제 유량 단위 m3/s로 복원)
                dummy = np.zeros((1, 2))
                dummy[0, 1] = pred_scaled[0, 0]
                final_val = scaler.inverse_transform(dummy)[0, 1]
                
                # 결과 출력
                target_date = selected_date + timedelta(days=3)
                st.markdown("---")
                st.balloons()
                st.success(f"### ✅ {target_date.strftime('%Y-%m-%d')} 예상 유량: **{final_val:.3f} $m^3/s$**")
                
                # 6. 결과 시각화
                st.subheader("📈 입력 데이터 분석 (최근 7일 추이)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['fw'], name='유량 ($m^3/s$)', line=dict(color='#1f77b4', width=3)))
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['rf'], name='강우 (mm)', line=dict(color='#d62728', width=3), yaxis="y2"))
                
                fig.update_layout(
                    yaxis=dict(title="유량 ($m^3/s$)"),
                    yaxis2=dict(title="강우 (mm)", overlaying="y", side="right"),
                    hovermode="x unified",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 해당 날짜의 실시간 데이터를 충분히 불러오지 못했습니다. 다른 날짜를 선택해 보세요.")
    else:
        st.error("❌ AI 모델이 로드되지 않아 예측 기능을 사용할 수 없습니다.")
