import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters


df_train = us_apl_train
df_train = df_train.drop(df_train.columns[0],axis=1)

print(df_train.shape)

df_test = us_apl_test
df_test = df_test.drop(df_test.columns[0],axis=1)

df_train.head()

# need organizing 
df_train['fiscal_month'] = df_train['fiscal_year_historical'].map(str) + '-' + df_train['fiscal_month_historical'].map(str).apply(lambda x: x.zfill(2))
df_test['fiscal_month'] = df_test['fiscal_year_historical'].map(str) + '-' + df_test['fiscal_month_historical'].map(str).apply(lambda x: x.zfill(2))
df_train.head()

X_train = df_train.sort_values(by='fiscal_month')
#X_train['fiscal_month'] = X_train['fiscal_month'].astype('datetime64[ns]')
X_train = X_train.set_index('fiscal_month')
X_train['fiscal_month'] = X_train['fiscal_month'].astype('datetime64[ns]')
X_train = X_train['sales_amount']

X_train.shape
X_train = X_train.groupby('fiscal_month').agg({'sales_amount':"sum"})
X_train.head()

def plot_rolling(df):
  plt.plot(df, color = 'blue', label = 'Original')
  plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
  plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
  plt.legend(loc = 'best', prop={'size': 6})
  plt.title('Rolling Mean & Rolling Standard Deviation')
  plt.show()

def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best', prop={'size': 6})
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['sales_amount'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
        
# exp decay
rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
df_log_exp_decay = df_log - rolling_mean_exp_decay
df_log_exp_decay.dropna(inplace=True)
get_stationarity(df_log_exp_decay)

df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)
get_stationarity(df_log_shift)

decomposition = seasonal_decompose(df_log) 
model = ARIMA(df_log, order=(2,1,2))
results = model.fit(disp=-1)

def plot_decomp(df_log_shift,results):
  plt.plot(df_log_shift)
  plt.plot(results.fittedvalues, color='red')


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(emea_series,order=(5,1,0))

model_fit = model.fit(disp=0)

model_fit.forecast(steps=15)[0].tolist()

#another verison 
X = us_series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(12):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
