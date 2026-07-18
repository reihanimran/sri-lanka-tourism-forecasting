"""
regenerate_results.py
Rebuild every results CSV on the CORRECTED data (comma-parsing fixed in
src/data_loader.py). Produces the files the dashboard and thesis consume.
"""
import sys, os, warnings
sys.path.append('src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import xgboost as xgb
from data_loader import get_monthly_total, get_monthly_total_synthetic, train_test_split
from models import make_ml_features, ML_FEATURES

DATA='data/Tourism_MOM_Dataset.csv'; SPLIT='2025-01-01'; OUT='results'

def m_mae(a,p): return float(np.mean(np.abs(np.asarray(a,float)-np.asarray(p,float))))
def m_rmse(a,p): return float(np.sqrt(np.mean((np.asarray(a,float)-np.asarray(p,float))**2)))
def m_mape(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float); m=a!=0
    return float(np.mean(np.abs((a[m]-p[m])/a[m]))*100)
def m_wape(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float)
    return float(np.sum(np.abs(a-p))/np.sum(np.abs(a))*100)
def m_smape(a,p):
    a,p=np.asarray(a,float),np.asarray(p,float); d=(np.abs(a)+np.abs(p))
    return float(np.mean(2*np.abs(a-p)/np.where(d==0,1,d))*100)

def sarima(train): return SARIMAX(train,order=(1,1,1),seasonal_order=(1,1,1,12),
    enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)

def xgb_tf(y,params):
    fd=make_ml_features(y).dropna(); tr,te=fd[fd.index<SPLIT],fd[fd.index>=SPLIT]
    m=xgb.XGBRegressor(random_state=42,verbosity=0,**params); m.fit(tr[ML_FEATURES],tr['y'])
    return pd.Series(m.predict(te[ML_FEATURES]),index=te.index)

def hybrid(train,sar,sfc,idx,params,feats):
    resid=train-sar.fittedvalues; rd=pd.DataFrame({'r':resid})
    for L in [1,2,12]: rd[f'lag{L}']=rd['r'].shift(L)
    rd['month']=rd.index.month; rd=rd.dropna()
    xr=xgb.XGBRegressor(random_state=42,verbosity=0,**params); xr.fit(rd[feats],rd['r'])
    allr,out=resid.copy(),[]
    for i,d in enumerate(idx):
        f={'lag1':allr.iloc[-1],'lag2':allr.iloc[-2],'lag12':allr.iloc[-12],'month':d.month}
        rp=xr.predict(pd.DataFrame([f])[feats])[0]; out.append(sfc.iloc[i]+rp)
        allr=pd.concat([allr,pd.Series([rp],index=[d])])
    return pd.Series(np.clip(out,0,None),index=idx)

def six_models(y,test):
    tr,_=train_test_split(y,SPLIT); n=len(test); idx=test.index; rows=[]
    sn=pd.Series([y.iloc[y.index.get_loc(d)-12] for d in idx],index=idx)
    rows.append(('Seasonal Naive',sn))
    sar=sarima(tr); sfc=sar.forecast(n); rows.append(('SARIMA',sfc))
    au=pm.auto_arima(tr,seasonal=True,m=12,stepwise=True,suppress_warnings=True,error_action='ignore')
    rows.append(('Auto-SARIMA',pd.Series(au.predict(n),index=idx)))
    ex=pd.DataFrame({'month':y.index.month,'quarter':y.index.quarter,'t':np.arange(len(y))},index=y.index)
    smx=SARIMAX(tr,exog=ex[ex.index<SPLIT],order=(1,1,1),seasonal_order=(1,1,1,12),
        enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    rows.append(('SARIMAX',smx.forecast(n,exog=ex[ex.index>=SPLIT])))
    rows.append(('XGBoost',xgb_tf(y,dict(n_estimators=200,max_depth=4,learning_rate=0.05))))
    rows.append(('SARIMA-XGBoost Hybrid',hybrid(tr,sar,sfc,idx,dict(n_estimators=50,max_depth=2,learning_rate=0.05),['lag1','lag12','month'])))
    out=[{'Model':nm,'MAE':round(m_mae(test,p),2),'RMSE':round(m_rmse(test,p),2),'MAPE':round(m_mape(test,p),2)} for nm,p in rows]
    return pd.DataFrame(out).sort_values('MAPE').reset_index(drop=True)

y0=get_monthly_total(DATA); ys=get_monthly_total_synthetic(DATA)
test=y0[y0.index>=SPLIT]

# 1. model_results.csv (default, zero-based)
mr=six_models(y0,test); mr.to_csv(f'{OUT}/model_results.csv',index=False)
print("model_results.csv (corrected, zero, default):"); print(mr.to_string(index=False))

# 2. final_results_synthetic.csv
fs=six_models(ys,test)[['Model','MAPE']]; fs.to_csv(f'{OUT}/final_results_synthetic.csv',index=False)

# 3. robust_metrics.csv (leading models, zero)
tr0,_=train_test_split(y0,SPLIT); s0=sarima(tr0)
lead={'XGBoost':xgb_tf(y0,dict(n_estimators=200,max_depth=4,learning_rate=0.05)),'SARIMA':s0.forecast(len(test))}
rob=pd.DataFrame([{'Model':k,'MAE':round(m_mae(test,v),2),'RMSE':round(m_rmse(test,v),2),
    'MAPE':round(m_mape(test,v),2),'WAPE':round(m_wape(test,v),2),'sMAPE':round(m_smape(test,v),2)} for k,v in lead.items()])
rob.to_csv(f'{OUT}/robust_metrics.csv',index=False)

# 4. walk-forward (zero-based, min 48 train, step 3, horizon 3)
# Config matches notebook 06 exactly (XGB 100/3/0.05, 16 windows) so the
# notebook and this script produce identical CSVs.
y=y0; rows=[]; start=48
ends=list(range(start,len(y)-3,3))
for w,e in enumerate(ends,1):
    tr=y.iloc[:e]; te=y.iloc[e:e+3]
    if len(te)<3: break
    sar=sarima(tr); sfc=sar.forecast(3)
    fd=make_ml_features(y).dropna(); xtr=fd[fd.index<te.index[0]]; xte=fd.loc[fd.index.isin(te.index)]
    xm=xgb.XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.05,random_state=42,verbosity=0)
    xm.fit(xtr[ML_FEATURES],xtr['y']); xgp=pd.Series(xm.predict(xte[ML_FEATURES]),index=xte.index)
    hy=hybrid(tr,sar,sfc,te.index,dict(n_estimators=50,max_depth=2,learning_rate=0.05),['lag1','lag12','month'])
    rows.append({'Window':w,'Train End':tr.index[-1].strftime('%Y-%m'),'Test Start':te.index[0].strftime('%Y-%m'),
        'Train Size':len(tr),'SARIMA MAPE':round(m_mape(te,sfc),2),
        'XGBoost MAPE':round(m_mape(te,xgp.reindex(te.index)),2),'Hybrid MAPE':round(m_mape(te,hy),2)})
wf=pd.DataFrame(rows); wf.to_csv(f'{OUT}/walkforward_results.csv',index=False)
NAMES={'SARIMA':'SARIMA','XGBoost':'XGBoost','Hybrid':'SARIMA-XGBoost Hybrid'}
summ=pd.DataFrame([{'Model':NAMES[mdl],'Mean MAPE':round(wf[f'{mdl} MAPE'].mean(),2),
    'Median MAPE':round(wf[f'{mdl} MAPE'].median(),2),
    'Best Window Count':int((wf[['SARIMA MAPE','XGBoost MAPE','Hybrid MAPE']].idxmin(axis=1)==f'{mdl} MAPE').sum())}
    for mdl in ['SARIMA','XGBoost','Hybrid']])
summ.to_csv(f'{OUT}/walkforward_summary.csv',index=False)
print("\nwalkforward_summary.csv:"); print(summ.to_string(index=False))

# 4b. Diebold-Mariano test (pinned, reproducible): XGBoost vs SARIMA on the holdout
from scipy import stats
xgb_hold=xgb_tf(y0,dict(n_estimators=200,max_depth=4,learning_rate=0.05))
sar_hold=sarima(train_test_split(y0,SPLIT)[0]).forecast(len(test))
e1=test.values-xgb_hold.reindex(test.index).values; e2=test.values-sar_hold.values
d=e1**2-e2**2; n=len(d)
dm_stat=float(np.mean(d)/np.sqrt(np.var(d,ddof=1)/n))
dm_p=float(2*(1-stats.t.cdf(abs(dm_stat),df=n-1)))
pd.DataFrame([{'Comparison':'XGBoost vs SARIMA','Loss':'squared error','n':n,
    'DM_stat':round(dm_stat,4),'p_value':round(dm_p,4),
    'Significant_5pct':dm_p<0.05}]).to_csv(f'{OUT}/dm_test.csv',index=False)
print(f"\nDM test (XGBoost vs SARIMA): stat={dm_stat:.4f}  p={dm_p:.4f}")

# 4c. Robustness: leakage-free fixed-origin (multi-step) versions of XGBoost
# and Seasonal Naive. Primary table evaluates XGBoost one-step-ahead (features
# from observed lags); here lags beyond the Dec-2024 origin come from the
# model's own predictions (recursive), and Seasonal Naive uses the last value
# observed BEFORE the origin for every horizon.
tr0=train_test_split(y0,SPLIT)[0]
xm=xgb.XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,random_state=42,verbosity=0)
ftr=make_ml_features(tr0).dropna(); xm.fit(ftr[ML_FEATURES],ftr['y'])
hist=tr0.copy(); recs=[]
for dts in test.index:
    f={'lag1':hist.iloc[-1],'lag2':hist.iloc[-2],'lag3':hist.iloc[-3],
       'lag6':hist.iloc[-6],'lag12':hist.iloc[-12],
       'rm3':hist.iloc[-3:].mean(),'rm6':hist.iloc[-6:].mean(),
       'month':dts.month,'quarter':dts.quarter}
    p=float(xm.predict(pd.DataFrame([f])[ML_FEATURES])[0])
    recs.append(p); hist=pd.concat([hist,pd.Series([p],index=[dts])])
xgb_ms=pd.Series(np.clip(recs,0,None),index=test.index)
last_cycle=tr0.iloc[-12:]  # Jan-Dec 2024, the last full year observed at the origin
sn_fo=pd.Series([last_cycle[last_cycle.index.month==dts.month].iloc[0] for dts in test.index],
                index=test.index)
ms=pd.DataFrame([
    {'Model':'XGBoost (recursive multi-step)','MAE':round(m_mae(test,xgb_ms),2),
     'RMSE':round(m_rmse(test,xgb_ms),2),'MAPE':round(m_mape(test,xgb_ms),2)},
    {'Model':'Seasonal Naive (fixed origin)','MAE':round(m_mae(test,sn_fo),2),
     'RMSE':round(m_rmse(test,sn_fo),2),'MAPE':round(m_mape(test,sn_fo),2)}])
ms.to_csv(f'{OUT}/multistep_robustness.csv',index=False)
print("\nmultistep_robustness.csv:"); print(ms.to_string(index=False))

# 5. forecast (24-month SARIMA on full corrected series, with 95% intervals)
# SARIMA is used (not XGBoost): tree ensembles cannot extrapolate beyond the training
# range, which collapses the XGBoost forward path (diagnosed in Notebook 10). SARIMA
# extrapolates via its parametric seasonal model and gives native prediction intervals.
full=y0.copy()
fut=pd.date_range(full.index[-1]+pd.offsets.MonthEnd(1),periods=24,freq='ME')
sar_full=sarima(full); gf=sar_full.get_forecast(24); ci=gf.conf_int(alpha=0.05)
mean=np.clip(np.round(gf.predicted_mean.values,0),0,None)
lo=np.clip(np.round(ci.iloc[:,0].values,0),0,None)
hi=np.clip(np.round(ci.iloc[:,1].values,0),0,None)
fcs=pd.DataFrame({'Date':fut.strftime('%Y-%m-%d'),'Forecast':mean,'Lower_95':lo,'Upper_95':hi})
fcs.to_csv(f'{OUT}/forecasts.csv',index=False)
fcs[['Date','Forecast']].to_csv(f'{OUT}/future_forecast.csv',index=False)
fcs.to_csv(f'{OUT}/forecast_with_intervals.csv',index=False)

print("\nForecast Apr 2026 (SARIMA):",f"{mean[0]:,.0f}  (recursive XGBoost path undershoots at 94,986 — see Notebook 10)")
print("ALL CSVs regenerated in results/")
