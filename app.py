import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time

# 1. 페이지 설정
st.set_page_config(page_title="내성천 유량 예측 시스템", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) 실시간 AI 유량 예측 시스템")

# 2. 모델 및 스케일러 로드
@st.cache_resource
def load_resources():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_resources()

# 3. 실시간 데이터 수집 함수 (진행 상황 표시 추가)
def fetch_river_data(base_date):
    api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
    # LSTM 입력 7일치를 위해 넉넉히 12일 전부터 조회
    start_date = (base_date - timedelta(days=12)).strftime("%Y%m%d")
    end_date = base_date.strftime("%Y%m%d")
    
    status_text = st.empty()
    progress_bar = st.progress(0)

    try:
        status_text.text("📡 홍수통제소 서버에서 강우 데이터를 가져오는 중...")
        r_url = f"http://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_date}/{end_date}.json"
        r_res = requests.get(r_url, timeout=10).json().get('content', [])
        progress_bar.progress(30)

        status_text.text("📡 홍수통제소 서버에서 유량 데이터를 가져오는 중...")
        f_url = f"http://api.hrfco.go.kr/{api_key}/rfw/list/1D/2004680/{start_date}/{end_date}.json"
        f_res = requests.get(f_url, timeout=10).json().get('content', [])
        progress_bar.progress(60)

        if not r_res or not f_res:
            status_text.empty()
            progress_bar.empty()
            st.error("⚠️ 선택하신 날짜에 관측 데이터가 존재하지 않습니다. (과거 날짜를 선택해 보세요)")
            return None

        status_text.text("⚙️ 수집된 데이터를 분석용으로 변환 중...")
        df_r = pd.DataFrame(r_res)[['ymd', 'rf']]
        df_f = pd.DataFrame(f_res)[['ymd', 'fw']]
        df = pd.merge(df_r, df_f, on='ymd').sort_values('ymd')
        df[['rf', 'fw']] = df[['rf', 'fw']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        progress_bar.progress(100)
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        if len(df) < 7:
            st.warning(f"⚠️ 데이터 부족: 최소 7일치가 필요하지만 {len(df)}일치만 확보되었습니다.")
            return None
            
        return df.tail(7)
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.error(f"❌ 데이터 수집 중 오류 발생: {e}")
        return None

# 4. 사이드바 및 레이아웃
with st.sidebar:
    st.header("📊 시스템 상태")
    if model and scaler:
        st.success("AI 모델 로드 완료")
    else:
        st.error("모델 로드 실패")
    st.info("이 시스템은 LSTM 신경망을 사용하여 내성천 회룡교 지점의 3일 뒤 유량을 예측합니다.")

# 5. 메인 로직
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📅 예측 설정")
    selected_date = st.date_input("기준 날짜 선택", datetime.now() - timedelta(days=2))
    predict_btn = st.button("🚀 유량 예측 실행", use_container_width=True)

if predict_btn:
    if model and scaler:
        recent_data = fetch_river_data(selected_date)
        
        if recent_data is not None:
            # AI 예측
            inputs = scaler.transform(recent_data[['rf', 'fw']])
            inputs = inputs.reshape(1, 7, 2)
            pred_scaled = model.predict(inputs)
            
            # 역스케일링
            dummy = np.zeros((1, 2))
            dummy[0, 1] = pred_scaled[0, 0]
            final_val = scaler.inverse_transform(dummy)[0, 1]
            
            target_date = selected_date + timedelta(days=3)
            
            with col2:
                st.subheader("🎯 예측 결과")
                st.metric(label=f"📅 {target_date.strftime('%Y-%m-%d')} 예상 유량", 
                          value=f"{final_val:.3f} m³/s")
                st.balloons()
                
                # 그래프 시각화
                st.markdown("---")
                st.subheader("📈 입력 데이터 분석 (최근 7일)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['fw'], name='유량(m³/s)', line=dict(color='#1f77b4')))
                fig.add_trace(go.Scatter(x=recent_data['ymd'], y=recent_data['rf'], name='강우(mm)', line=dict(color='#d62728'), yaxis="y2"))
                fig.update_layout(yaxis2=dict(overlaying="y", side="right"), template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("시스템 구성 파일을 로드할 수 없어 예측이 불가능합니다.")
