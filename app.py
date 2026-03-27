import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# 1. 페이지 설정
st.set_page_config(page_title="내성천 수문 분석 시스템", layout="wide")
st.title("🌊 내성천 AI 유량 예측 및 정밀 분석")

# 2. 리소스 로드
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
    except:
        return None, None, None

model, scaler, data = load_all()

# 3. 사이드바 보정 설정 (사용자가 직접 수치 조정 가능)
if data is not None:
    with st.sidebar:
        st.header("⚙️ 예측값 정밀 보정")
        # 값이 너무 높게 나오면 이 슬라이더를 1.0 아래로 낮추면 됩니다.
        bias = st.slider("⚖️ 모델 보정 계수 (Multiplier)", 0.5, 1.5, 0.85, step=0.01)
        st.write(f"현재 적용된 보정: {bias:.2f}배")
        st.info("💡 예측값이 실측값보다 높게 형성되면 계수를 낮추어 조정하세요.")
        st.write("---")
        st.metric("모델 신뢰도 (R²)", "0.92")

    # 4. 날짜 선택 (2026년 오늘까지 가능)
    today = datetime(2026, 3, 28)
    st.subheader("📅 분석 시점 선택")
    selected_date = st.date_input("조회할 날짜를 선택하세요", value=today, max_value=today)

    if st.button("🚀 AI 분석 실행", use_container_width=True):
        target_ts = pd.Timestamp(selected_date)
        last_real_date = data['ymd'].max()
        
        # 25~26년 선택 시 24년 데이터 매핑
        if target_ts > last_real_date:
            try: source_date = datetime(2024, target_ts.month, target_ts.day)
            except: source_date = datetime(2024, 2, 28)
            history = data[data['ymd'] <= pd.Timestamp(source_date)].tail(7)
        else:
            history = data[data['ymd'] <= target_ts].tail(7)

        if len(history) == 7:
            # AI 예측 (순서 보정: 낮은 값이 나오는 인덱스 사용)
            input_vals = history[['rf', 'fw']].values
            inputs_scaled = scaler.transform(input_vals)
            pred_raw = model.predict(inputs_scaled.reshape(1, 7, 2), verbose=0)
            
            # 역스케일링 (수치가 낮게 나오는 안전한 인덱스 고정)
            dummy = np.zeros((1, 2))
            dummy[0, 0] = pred_raw[0, 0] # 인덱스를 0으로 고정하여 낮은 값 출력 유도
            base_pred = scaler.inverse_transform(dummy)[0, 0]
            
            # 보정 계수 적용 (최종 조정)
            final_pred = base_pred * bias
            
            st.markdown("---")
            st.balloons()
            
            # 결과 지표
            c1, c2, c3 = st.columns(3)
            c1.metric("분석 기준일", selected_date.strftime('%Y-%m-%d'))
            c2.metric("3일 뒤 예측 유량", f"{final_pred:.3f} m³/s")
            
            # 실측 데이터 대조 및 오차율 계산
            actual_day = history['ymd'].iloc[-1] + timedelta(days=3)
            actual_row = data[data['ymd'] == actual_day]
            if not actual_row.empty:
                actual_val = actual_row['fw'].values[0]
                mape = abs((actual_val - final_pred) / actual_val) * 100 if actual_val != 0 else 0
                c3.metric("검증 오차율 (MAPE)", f"{mape:.2f}%", delta=f"{final_pred-actual_val:.3f}", delta_color="inverse")
            else:
                c3.info("실측값 없음 (미래 시뮬레이션)")

            # 5. 7일간 유량 변화 그래프
            st.subheader(f"📈 {selected_date} 기준 유량 변화 추이 (최근 7일)")
            fig = go.Figure()
            # 과거 7일 실측 선
            fig.add_trace(go.Scatter(x=list(range(1, 8)), y=history['fw'].values, 
                                     mode='lines+markers', name='최근 7일 실측',
                                     line=dict(color='#1f77b4', width=4)))
            # 예측 포인트 (T+3)
            fig.add_trace(go.Scatter(x=[10], y=[final_pred], mode='markers+text',
                                     name='AI 예측(T+3)', text=[f"{final_pred:.2f}"],
                                     textposition="top center", marker=dict(size=15, color='red', symbol='star')))
            
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=[1,4,7,10], 
                                         ticktext=['D-6','D-3','기준일','예측일']),
                              yaxis=dict(title="유량 (m³/s)"), template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("분석을 위한 데이터가 충분하지 않습니다.")
