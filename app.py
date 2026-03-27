import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="내성천 유량 예측 시스템", page_icon="🌊", layout="wide")
st.title("🌊 내성천(회룡교) AI 실시간 유량 예측 및 시뮬레이션")

@st.cache_resource
def load_resources():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        # 파일 데이터 로드 (백업 및 시뮬레이션용)
        r_df = pd.read_csv('rain_data.csv')
        f_df = pd.read_csv('flow_data.csv')
        r_df.columns = r_df.columns.str.lower()
        f_df.columns = f_df.columns.str.lower()
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        return model, scaler, df.sort_values('ymd').dropna()
    except Exception as e:
        st.error(f"파일 로드 실패: {e}")
        return None, None, None

model, scaler, local_data = load_resources()

# --- 탭 구성: 실시간 예측 vs 과거 시뮬레이션 ---
tab1, tab2 = st.tabs(["🚀 오늘 기준 실시간 예측", "📉 과거 데이터 시뮬레이션"])

# ---------------------------------------------------------
# [탭 1] 실시간 예측 (진짜 미래를 보는 곳)
# ---------------------------------------------------------
with tab1:
    st.subheader("📍 오늘로부터 3일 뒤 유량을 예측합니다.")
    if st.button("🛰️ 실시간 데이터 수집 및 미래 예측 시작"):
        api_key = "5B0DD072-D10C-4185-9679-794D6CCD8081"
        today = datetime.now()
        start = (today - timedelta(days=10)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")
        
        with st.spinner('홍수통제소에서 실시간 수문 정보를 땡겨오는 중...'):
            try:
                # API 호출 시도
                r_url = f"http://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start}/{end}.json"
                f_url = f"http://api.hrfco.go.kr/{api_key}/rfw/list/1D/2004680/{start}/{end}.json"
                r_res = requests.get(r_url, timeout=5).json().get('content', [])
                f_res = requests.get(f_url, timeout=5).json().get('content', [])
                
                if r_res and f_res:
                    live_df = pd.merge(pd.DataFrame(r_res)[['ymd', 'rf']], 
                                       pd.DataFrame(f_res)[['ymd', 'fw']], on='ymd')
                    live_df[['rf', 'fw']] = live_df[['rf', 'fw']].apply(pd.to_numeric)
                    recent_7 = live_df.sort_values('ymd').tail(7)
                else:
                    # API 실패 시 로컬 파일의 가장 최신 데이터 사용
                    st.warning("⚠️ 실시간 API 응답이 없어 보유 데이터 중 가장 최신 기록을 사용합니다.")
                    recent_7 = local_data.tail(7)
                
                # 예측 실행
                inputs = scaler.transform(recent_7[['rf', 'fw']])
                pred = model.predict(inputs.reshape(1, 7, 2))
                dummy = np.zeros((1, 2)); dummy[0, 1] = pred[0, 0]
                final_val = scaler.inverse_transform(dummy)[0, 1]
                
                target_date = pd.to_datetime(recent_7['ymd'].iloc[-1]) + timedelta(days=3)
                
                st.balloons()
                st.success(f"### 🔮 {target_date.strftime('%Y-%m-%d')} 예상 유량: **{final_val:.3f} m³/s**")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent_7['ymd'], y=recent_7['fw'], name='현재 유량', line=dict(width=3)))
                fig.add_trace(go.Scatter(x=[target_date], y=[final_val], name='미래 예측', mode='markers+text', 
                                         text=["3일 뒤"], textposition="top center", marker=dict(size=15, color='gold', symbol='star')))
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"예측 실패: {e}")

# ---------------------------------------------------------
# [탭 2] 과거 시뮬레이션 (검증용)
# ---------------------------------------------------------
with tab2:
    st.subheader("🧐 과거 특정 시점의 AI 판단 확인")
    if local_data is not None:
        min_d = local_data['ymd'].min() + timedelta(days=7)
        max_d = local_data['ymd'].max() - timedelta(days=3)
        sel_d = st.slider("날짜 선택", min_value=min_d.to_pydatetime(), max_value=max_d.to_pydatetime())
        
        if st.button("🔎 분석 결과 보기"):
            mask = (local_data['ymd'] <= pd.Timestamp(sel_d))
            history = local_data.loc[mask].tail(7)
            inputs = scaler.transform(history[['rf', 'fw']])
            pred = model.predict(inputs.reshape(1, 7, 2))
            dummy = np.zeros((1, 2)); dummy[0, 1] = pred[0, 0]
            val = scaler.inverse_transform(dummy)[0, 1]
            
            st.write(f"📅 선택 시점 기준 3일 뒤 예측값: **{val:.3f} m³/s**")
            # (이하 그래프 로직 생략 - 탭1과 유사)
