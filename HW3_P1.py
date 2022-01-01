import yfinance as yf
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Methon reference from : Book by Marcos López de Prado 'Advances in Financial Machine Learning'

# Setting Triple Barrier 

def apply_pt_sl_on_t1(close, events, pt_sl):

    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    
    events_ = events
    out = events_[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] #* events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs

    if pt_sl[1] > 0:
        sl = -pt_sl[1] #* events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    count = 0
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1)  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl].index.min()  # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt].index.min()  # earliest profit taking
        
    return out

def add_vertical_barrier(t_events, close, num_days=1):
    
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])  # NaNs at end
    return t1

def get_events(close, t_events, pt_sl, target, min_ret, 
              vertical_barrier_times=False, side=None, label = None):
   
    # 1) Get target
    target = target.loc[target.index.intersection(t_events)]
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    # 3) Form events object, apply stop loss on vertical barrier
    if side is None:
        side_ = pd.Series(1., index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[1]]
    else:
        side_ = side.loc[target.index]
        pt_sl_ = pt_sl[:2]

    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_},
                        axis=1)
    events = events.dropna(subset=['trgt'])

    # Apply Triple Barrier
    
    df0 = apply_pt_sl_on_t1(close=close,
                            events=events,
                            pt_sl=pt_sl_)

    if label is None:
       label = [1, -1, 0] # Setting labels

    events['t1'] = df0.dropna(how='all').min(axis=1)  # ignores NaN
    
    for loc, t1 in events['t1'].fillna(close.index[-1]).iteritems():
        px = close[t1]
        p0 = events.loc[loc, 'trgt']
        events.loc[loc, 'label'] = label[2]
        if (px/p0 - 1) > pt_sl_[0]: events.loc[loc, 'label'] = label[0] 
        if (px/p0 - 1) < -pt_sl_[1]: events.loc[loc, 'label'] = label[1]
        
    
    if side is None:
        events = events.drop('side', axis=1)

    return events

if __name__ == '__main__':
    
    # Weighted index
    
    order = ['Open', 'Close', 'High', 'Low', 'Volume']
    order_l = ['open', 'close', 'high', 'low', 'volume']
    

    # Problem 1A
    # Collect the TAIEX from 2012/12/01 to 2021/12/02 (Day Bar)
    
    data = yf.download('%5ETWII', start='2012-12-01', end='2021-12-02')
    data.drop('Adj Close', axis='columns', inplace=True)
    data = data[order]
    data.columns = order_l
   
    
    # Problem 1B
    # Set labels and (apply stop loss / profit taking) pt_sl
    
    labels = [1,2,0]
    PTSL = [0.04, 0.02] 
    min_ret = 0.0005

    close = data['close']

    # Adding vertical barriers in 20 days
    
    vertical_barriers = add_vertical_barrier(t_events=close.index , close=close, num_days=20) 
    
    # Implement triple barrier method and label

    triple_barrier_events = get_events(close=close,
                                  t_events=close.index,
                                  pt_sl=PTSL,
                                  target=close,
                                  min_ret=min_ret,
                                  vertical_barrier_times=vertical_barriers,
                                  label = ['1', '2', '0'])
    
    data = data.assign(label = triple_barrier_events['label'])


    # Problem 1C
    # (i) Bias of moving average for 5-days, 10-days, 20-days, 60-day
    
    SMA5 = ta.SMA(close, timeperiod = 5)
    SMA10 = ta.SMA(close, timeperiod = 10)
    SMA20 = ta.SMA(close, timeperiod = 20)
    SMA60 = ta.SMA(close, timeperiod = 60)
    data['bias5'] = (close - SMA5)/SMA5
    data['bias10'] = (close - SMA10)/SMA10
    data['bias20'] = (close - SMA20)/SMA20
    data['bias60'] = (close - SMA60)/SMA60

    # (ii) RSI : 14
    data['RSI14'] = ta.RSI(close, timeperiod = 14)

    # (iii) MACD, MACD signal, MACD histogram
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = \
        ta.MACDFIX(close, signalperiod=9)
    
    # (iv) Save problem 1,2,3 results to csv
    data.to_csv('./TAIEX.csv')
    


    