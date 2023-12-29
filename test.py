import alphalens as al

import pandas as pd
from pandas_ta.core import rsi

def read_df(path):
    '''
    读取单个数据文档
    '''
    binance_price_df = pd.read_csv(path)
    binance_price_df.columns = ['open_time','open','high','low','close','volume','close_time','quote_volume','count',
                                'taker_buy_volume','taker_buy_quote_volume','ignore']
    return binance_price_df


def set_datetime_index(data_df, colname='open_time'):
    data_df['DateTime'] = pd.to_datetime(data_df[colname], unit='ms')
    data_df = data_df.set_index('DateTime')
    return data_df

df1 = read_df('../data/15m_data/BTC2020/BTCUSDT-15m-2020-01.zip')
df1 = set_datetime_index(df1)

price_1 = pd.DataFrame(rsi(df1['close']))
price_1.columns = ['BTC']

df2 = read_df('../data/15m_data/ETH2020/ETHUSDT-15m-2020-01.zip')
df2 = set_datetime_index(df1)

price_2 = pd.DataFrame(rsi(df1['close']))
price_2.columns = ['ETH']

a = price_1.copy()
a['price'] = a['BTC']
a['BTC'] = 'BTC'
a.rename({'BTC':'coin'},axis=1, inplace=True)

b = price_2.copy()
b['price'] = b['ETH']
b['ETH'] = 'ETH'
b.rename({'ETH':'coin'},axis=1, inplace=True)

df = pd.concat([a,b])
df.reset_index(inplace=True)
factor_df = df.set_index(keys=['DateTime', 'coin'])
factor_df.sort_index(inplace=True)

btc_df = df1[['close']]
btc_df.columns = ['BTC']

eth_df = df2[['close']]
eth_df.columns = ['ETH']

price_df = pd.concat([btc_df, eth_df], axis=1)
price_df.sort_index(inplace=True)

factor_df.index.levels[0].freq = '15T'
price_df.index.freq = '15T'


al.utils.get_clean_factor_and_forward_returns(factor_df, price_df)