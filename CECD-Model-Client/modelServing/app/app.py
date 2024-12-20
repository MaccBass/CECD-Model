import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel

import joblib
import pickle
import pandas as pd
import numpy as np

# 모델 및 인코더 불러오기
# app.py와 같은 경로에 있어야 함.
# joblib 파일 링크: https://drive.google.com/file/d/1tN0Pq114ElPtcobSPWAq3wxxmioEydMv/view?usp=sharing
with open("random_forest.joblib", "rb") as fr:
    mlCore = joblib.load(fr);
with open("le_gu.pkl", "rb") as fr:
    le_gu = pickle.load(fr);
with open("le_dong.pkl", "rb") as fr:
    le_dong = pickle.load(fr);

# 현재 월부터 과거 12개월까지 포함한 DataFrame 반환하는 함수
from datetime import datetime, timedelta
def make_x_df(gu, dong, exclusiveArea, floor, buildYear):
    current_date = datetime.now()
    deal_dates = []
    years = []
    month_sins = []
    month_coss = []
    
    for i in range(12):
        month = current_date.month - i
        year = current_date.year
        if month <= 0:
            month += 12
            year -= 1
        deal_dates.append(f"{year}-{month:02d}-15")
            
        years.append(year)
        month_sins.append(np.sin(2 * np.pi * month / 12))
        month_coss.append(np.cos(2 * np.pi * month / 12))

    interest_rates = make_ir(deal_dates)
    
    x_df = pd.DataFrame({
        "interestRate": interest_rates,
        "gu": [gu] * 12,
        "dong": [dong] * 12,
        "exclusiveArea": [exclusiveArea] * 12,
        "floor": [floor] * 12,
        "buildYear": [buildYear] * 12,
        "year": years,
        "month_sin": month_sins,
        "month_cos": month_coss,
        "day_sin": [0] * 12,
        "day_cos": [-1] * 12
    })
    return x_df
    
# 날짜 기준으로 금리값 넣어주는 함수
# 최근 10.11에 변동 외에는 없음. 금리 변동시마다 그때그때 갱신 필요
def make_ir(deal_dates):
    ir_values = []
    reference_date = datetime(2024, 10, 11)
    
    for date_str in deal_dates:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        if date >= reference_date:
            ir_values.append(3.25)
        else:
            ir_values.append(3.5)
    
    return ir_values

# 서버 실행
app = FastAPI(title="API test")

'''
# 받아올 데이터 feature 목록
class InDataset(BaseModel):
    gu: str
    dong: str
    exclusiveArea: float
    floor: int
    buildYear: int
'''

# GET - DB-모델간 통신
@app.get("/predict_anomality", status_code=200)
async def predict_anomality(
    dealDate: str=Query(...),
    interestRate: float=Query(...),
    gu: str=Query(...),
    dong: str=Query(...),
    exclusiveArea: float=Query(...),
    floor: int=Query(...),
    buildYear: int=Query(...),
    dealAmount: int=Query(...),
):
    x_df = pd.DataFrame([{'dealDate': dealDate, 'interestRate': interestRate, 'gu': gu, 'dong': dong, 
    'exclusiveArea': exclusiveArea, 'floor': floor, 'buildYear': buildYear}])

    x_df['dealDate'] = pd.to_datetime(x_df['dealDate'])
    x_df['year'] = x_df['dealDate'].dt.year
    x_df['month'] = x_df['dealDate'].dt.month
    x_df['day'] = x_df['dealDate'].dt.day

    x_df['month_sin'] = np.sin(2*np.pi*x_df['month'] / 12)
    x_df['month_cos'] = np.cos(2*np.pi*x_df['month'] / 12)
    x_df['day_sin'] = np.sin(2*np.pi*x_df['day'] / 31)
    x_df['day_cos'] = np.cos(2*np.pi*x_df['day'] / 31)
    x_df = x_df.drop(['dealDate', 'month', 'day'], axis=1)

    x_df['gu'] = le_gu.transform(x_df['gu'])
    x_df['dong'] = le_dong.transform(x_df['dong'])
    
    y_predict = int(mlCore.predict(x_df)[0])
    y_truth = dealAmount
    difference = abs(y_truth - y_predict) / y_truth
    if difference > 0.2:
        reliable = False
    else:
        reliable = True

    return {"prediction": y_predict, "difference": difference, "reliable": reliable}
    

# GET - 클라-모델간 통신
@app.get("/predict", status_code=200)
async def predict_tf(gu: str=Query(...), dong: str=Query(...), exclusiveArea=Query(...),
                    floor: int=Query(...), buildYear: int=Query(...)):
    x_df = make_x_df(gu, dong, exclusiveArea, floor, buildYear)
    # 구, 동 labeling
    x_df['gu'] = le_gu.transform(x_df['gu'])
    x_df['dong'] = le_dong.transform(x_df['dong'])
    
    y = mlCore.predict(x_df)
    current_date = datetime.now()
    deal_dates = []
    for i in range(12):
        month = current_date.month - i
        year = current_date.year
        if month <= 0:
            month += 12
            year -= 1
        deal_dates.append(f"{year}{month:02d}")
        
    predictions = {deal_dates[i]: int(y[i]) for i in range(12)}
    
    return predictions
