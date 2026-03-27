import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="내성천 수문 분석 시스템", layout="wide")
st.title("🌊 내성천 AI 유량 예측 (값 보정 모드)")

@st.cache_resource
def load_all():
    try:
        os.environ["KERAS_BACKEND"] = "tensorflow"
        model = tf.keras.models.load_model('naeseong_model.h5', compile=False)
        scaler = joblib.load('scaler.pkl')
        
        r_df = pd.read_csv('rain_data.csv')
        f_df = pd.read_csv('flow_data.csv')
        r_df.columns = r_df.columns.str.lower().str.strip()
        f_df.columns = f_df.columns.str.lower().str.strip()
        
        r_df['rf'] = pd.to_numeric(r_df['rf'], errors='coerce')
        f_df['fw'] = pd.to_numeric(f_df['fw'], errors='coerce')
        
        df = pd.merge(r_df[['ymd', 'rf']], f_df[['ymd', 'fw']], on='ymd')
        df['ymd'] = pd.to_datetime(df['ymd'].astype(str))
        df = df.dropna().sort_values('ymd')
        return model, scaler, df
    except Exception as e:
        st.error(f"로드 에러: {e}")
        return None, None, None

model, scaler, data = load_all()

if data is not None:
    # 사이드바에서 컬럼 순서 강제 조정 옵션 (디버깅용)
    st.sidebar.header("🛠️ 예측값 보정 설정")
    swap_order = st.sidebar.checkbox("강우/유량 순서 바꾸기", value=False)
    st.sidebar.info("예측값이 너무 높으면 위 체크박스를 눌러보세요.")

    today = datetime(2026, 3, 28)
    st.subheader("📅 분석 날짜 선택")
    selected_date = st.date_input("날짜 선택", value=today, max_value=today)

    if st.button("🚀 AI 분석 실행"):
        target_ts = pd.Timestamp(selected_date)
        last_real_date = data['ymd'].max()
        
        # 2024년 데이터 매핑 로직
        if target_ts > last_real_date:
            try: source_date = datetime(2024, target_ts.month, target_ts.day)
            except: source_date = datetime(2024, 2, 28)
            history = data[data['ymd'] <= pd.Timestamp(source_date)].tail(7)
        else:
            history = data[data['ymd'] <= target_ts].tail(7)

        if len(history) == 7:
            # 1. 입력 데이터 준비 (순서 조정 기능 포함)
            if swap_order:
                input_vals = history[['fw', 'rf']].values # 유량, 강우 순서
            else:
                input_vals = history[['rf', 'fw']].values # 강우, 유량 순서 (기본)
            
            inputs_scaled = scaler.transform(input_vals)
            pred_raw = model.predict(inputs_scaled.reshape(1, 7, 2), verbose=0)
            
            # 2. 역스케일링 (Dummy array 생성)
            # 만약 값이 너무 높다면 dummy[0, 0]에 넣어야 할 수도 있습니다.
            dummy = np.zeros((1, 2))
            if swap_order:
                dummy[0, 0] = pred_raw[0, 0] # 유량이 첫 번째 컬럼인 경우
                pred_val = scaler.inverse_transform(dummy)[0, 0]
            else:
                dummy[0, 1] = pred_raw[0, 0] # 유량이 두 번째 컬럼인 경우 (기본)
                pred_val = scaler.inverse_transform(dummy)[0, 1]
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("분석 기준일", selected_date.strftime('%Y-%m-%d'))
            col2.metric("3일 뒤 예측 유량", f"{pred_val:.3f} m³/s")
            
            # 오차 확인
            actual_day = history['ymd'].iloc[-1] + timedelta(days=3)
            actual_row = data[data['ymd'] == actual_day]
            if not actual_row.empty:
                actual_val = actual_row['fw'].values[0]
                mape = abs((actual_val - pred_val) / actual_val) * 100 if actual_val != 0 else 0
                col3.metric("검증 오차율", f"{mape:.2f}%", delta=f"{pred_val-actual_val:.3f}")
            
            # 디버깅 정보 (매우 중요)
            with st.expander("🛠️ 모델 내부 데이터 확인 (디버깅용)"):
                st.write(f"AI가 뱉은 순수 점수 (Scaled): {pred_raw[0, 0]:.4f}")
                st.write(f"사용된 입력 데이터 순서: {'유량, 강우' if swap_order else '강우, 유량'}")
                st.write(f"스케일러 최소값: {scaler.data_min_}")
                st.write(f"스케일러 최대값: {scaler.data_max_}")

            # 그래프
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, 8)), y=history['fw'].values, mode='lines+markers', name='7일 유량 패턴'))
            fig.add_trace(go.Scatter(x=[10], y=[pred_val], mode='markers', name='예측점', marker=dict(size=15, color='red', symbol='star')))
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
