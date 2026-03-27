# app.py 내의 history 추출 부분(약 55행~65행)을 이 로직으로 교체하세요.

        # 2026년 대응 데이터 매핑 (데이터 부족 시 마지막 유효 데이터 강제 참조)
        if target_ts > last_real_date:
            try: 
                # 윤년 등을 고려하여 안전하게 매핑
                s_date = pd.Timestamp(year=2024, month=target_ts.month, day=target_ts.day)
                if s_date > last_real_date:
                    s_date = last_real_date
            except: 
                s_date = last_real_date
            
            # 데이터가 끊긴 지점이라면 마지막 7일을 가져옴
            history = data[data['ymd'] <= s_date].tail(7)
        else:
            history = data[data['ymd'] <= target_ts].tail(7)

        # 데이터가 7개가 안 되면 무조건 전체 데이터의 마지막 7일 사용 (안전장치)
        if len(history) < 7:
            history = data.tail(7)
