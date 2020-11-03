# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:08:56 2019

@author: auW10
"""

import pandas as pd
import talib


class TaLib():
    """ Uma classe para calcular os indicadores """

    def __init__(self, candles):

        self.data  = candles
        # print("candles len: ", len(candles))

    def lista_de_indicadores(self):
        seq_len = 0
        
        if(len(self.data) == 7):
            date = self.data['date']
            time = self.data['time']
            oopen = self.data['open']
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']      
            
            # Carregar apenas os últimos 'seq_len' dados
            frame = {'DATE':date[seq_len:],'TIME':time[seq_len:],
                     'OPEN':oopen[seq_len:], 'HIGH':high[seq_len:],
                     'LOW':low[seq_len:], 'CLOSE':close[seq_len:],
                     'VOL':volume[seq_len:]
                     }
            
        if(len(self.data) == 5):
            oopen = self.data['open']
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']      
            
            # Carregar apenas os últimos 'seq_len' dados
            frame = {'OPEN':oopen[seq_len:], 'HIGH':high[seq_len:],
                     'LOW':low[seq_len:], 'CLOSE':close[seq_len:],
                     'VOL':volume[seq_len:]
                     }
        
        # Cria um DataFrame usando o frame com nome da coluna seguido dos dados
        dataframe = pd.DataFrame(frame)
        
        #############################
        # Overlap Studies Functions #
        #############################
        
        ##### EMA - Exponential Moving Average - 20 ------------------------------------------------
        ema5c = talib.EMA(self.data['close'], timeperiod=5)
        ema10c = talib.EMA(self.data['close'], timeperiod=10)
        ema20c = talib.EMA(self.data['close'], timeperiod=20)
        
        ema5o = talib.EMA(self.data['oopen'], timeperiod=5)
        ema10o = talib.EMA(self.data['oopen'], timeperiod=10)
        ema20o = talib.EMA(self.data['oopen'], timeperiod=20)

        ema5h = talib.EMA(self.data['high'], timeperiod=5)
        ema10h = talib.EMA(self.data['high'], timeperiod=10)
        ema20h = talib.EMA(self.data['high'], timeperiod=20)

        ema5l = talib.EMA(self.data['low'], timeperiod=5)
        ema10l = talib.EMA(self.data['low'], timeperiod=10)
        ema20l = talib.EMA(self.data['low'], timeperiod=20)
        

        dataframe['EMA_5_CLOSE'] = pd.Series(ema5c)
        dataframe['EMA_10_CLOSE'] = pd.Series(ema10c)
        dataframe['EMA_20_CLOSE'] = pd.Series(ema20c)

        dataframe['EMA_5_OPEN'] = pd.Series(ema5o)
        dataframe['EMA_10_OPEN'] = pd.Series(ema10o)
        dataframe['EMA_20_OPEN'] = pd.Series(ema20o)

        dataframe['EMA_5_HIGH'] = pd.Series(ema5h)
        dataframe['EMA_10_HIGH'] = pd.Series(ema10h)
        dataframe['EMA_20_HIGH'] = pd.Series(ema20h)

        dataframe['EMA_5_LOW'] = pd.Series(ema5l)
        dataframe['EMA_10_LOW'] = pd.Series(ema10l)
        dataframe['EMA_20_LOW'] = pd.Series(ema20l)
        
        ##### KAMA - Kaufman Adaptive Moving Average - 5/10/20 -------------------------------------
        kama5 = talib.KAMA(close, timeperiod=5)
        kama10 = talib.KAMA(close, timeperiod=10)
        kama20 = talib.KAMA(close, timeperiod=20)
        dataframe['KAMA_5'] = pd.Series(kama5[seq_len:])
        dataframe['KAMA_10'] = pd.Series(kama10[seq_len:])
        dataframe['KAMA_20'] = pd.Series(kama20[seq_len:])
        
        ##### MIDPOINT - MidPoint over period - 3/12 -----------------------------------------------
        midpoint3 = talib.MIDPOINT(close, timeperiod=3)
        midpoint12 = talib.MIDPOINT(close, timeperiod=12)
        dataframe['MIDPOINT_3'] = pd.Series(midpoint3[seq_len:])
        dataframe['MIDPOINT_12'] = pd.Series(midpoint12[seq_len:])

        ##### MIDPRICE - Midpoint Price over period - 3/12 -----------------------------------------
        midprice3 = talib.MIDPRICE(high, low, timeperiod=3)
        midprice12 = talib.MIDPRICE(high, low, timeperiod=12)
        dataframe['MIDPRICE_3'] = pd.Series(midprice3[seq_len:])
        dataframe['MIDPRICE_12'] = pd.Series(midprice12[seq_len:])
        
        
        ##### TRIMA - Triangular Moving Average - 5/10/20 ------------------------------------------
        trima5 = talib.TRIMA(close, timeperiod=5)
        trima10 = talib.TRIMA(close, timeperiod=10)
        trima20 = talib.TRIMA(close, timeperiod=20)
        dataframe['TRIMA_5'] = pd.Series(trima5[seq_len:])
        dataframe['TRIMA_10'] = pd.Series(trima10[seq_len:])
        dataframe['TRIMA_20'] = pd.Series(trima20[seq_len:])
        
        
        ################################
        # Momentum Indicator Functions #
        ################################
        
        ##### ADX - Average Directional Movement Index 3/6/10 --------------------------------------
        adx3 = talib.ADX(high, low, close, timeperiod=3)
        adx6 = talib.ADX(high, low, close, timeperiod=6)
        adx10 = talib.ADX(high, low, close, timeperiod=10)
        dataframe['ADX_3'] = pd.Series(adx3[seq_len:])
        dataframe['ADX_6'] = pd.Series(adx6[seq_len:])
        dataframe['ADX_10'] = pd.Series(adx10[seq_len:])
        
        ##### BOP - Balance Of Power
        bop = talib.BOP(oopen, high, low, close)
        dataframe['BOP'] = pd.Series(bop[seq_len:])
        
        ##### MOM - Momentum 3/6/10/14 -------------------------------------------------------------
        mom3 = talib.MOM(close, timeperiod=3)
        mom6 = talib.MOM(close, timeperiod=6)
        mom10 = talib.MOM(close, timeperiod=10)
        mom14 = talib.MOM(close, timeperiod=14)
        dataframe['MOM_3'] = pd.Series(mom3[seq_len:])
        dataframe['MOM_6'] = pd.Series(mom6[seq_len:])
        dataframe['MOM_10'] = pd.Series(mom10[seq_len:])
        dataframe['MOM_14'] = pd.Series(mom14[seq_len:])
        
        ##### ROC - Rate of change : ((price/prevPrice)-1)*100 3/5/7/10 ----------------------------
        roc3 = talib.ROC(close, timeperiod=3)
        roc5 = talib.ROC(close, timeperiod=5)
        roc7 = talib.ROC(close, timeperiod=7)
        roc10 = talib.ROC(close, timeperiod=10)
        dataframe['ROC_3'] = pd.Series(roc3[seq_len:])
        dataframe['ROC_5'] = pd.Series(roc5[seq_len:])
        dataframe['ROC_7'] = pd.Series(roc7[seq_len:])
        dataframe['ROC_10'] = pd.Series(roc10[seq_len:])
                
        ##### RSI - Relative Strength Index 3/6/10/14 ----------------------------------------------
        rsi3 = talib.RSI(close, timeperiod=3)
        rsi6 = talib.RSI(close, timeperiod=6)
        rsi10 = talib.RSI(close, timeperiod=10)
        rsi14 = talib.RSI(close, timeperiod=14)
        dataframe['RSI_3'] = pd.Series(rsi3[seq_len:])
        dataframe['RSI_6'] = pd.Series(rsi6[seq_len:])
        dataframe['RSI_10'] = pd.Series(rsi10[seq_len:])
        dataframe['RSI_14'] = pd.Series(rsi14[seq_len:])

        ##### TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA 3/5/7 -----------------------
        trix3 = talib.TRIX(close, timeperiod=3)
        trix5 = talib.TRIX(close, timeperiod=5)
        trix7 = talib.TRIX(close, timeperiod=7)
        dataframe['TRIX_3'] = pd.Series(trix3[seq_len:])
        dataframe['TRIX_5'] = pd.Series(trix5[seq_len:])
        dataframe['TRIX_7'] = pd.Series(trix7[seq_len:])
        
        ##### WILLR - Williams' %R 3/6/12/20 -------------------------------------------------------
        willr3 = talib.WILLR(high, low, close, timeperiod=3)
        willr6 = talib.WILLR(high, low, close, timeperiod=6)
        willr12 = talib.WILLR(high, low, close, timeperiod=12)
        willr20 = talib.WILLR(high, low, close, timeperiod=20)
        dataframe['WILLR_3'] = pd.Series(willr3[seq_len:])
        dataframe['WILLR_6'] = pd.Series(willr6[seq_len:])
        dataframe['WILLR_12'] = pd.Series(willr12[seq_len:])
        dataframe['WILLR_20'] = pd.Series(willr20[seq_len:])
        
        ##############################
        # Volume Indicator Functions #
        ##############################
        
        ##### AD - Chaikin A/D Line ----------------------------------------------------------------
        ad = talib.AD(high, low, close, volume)
        dataframe['AD'] = pd.Series(ad[seq_len:])
        
        ##### ADOSC - Chaikin A/D Oscillator -------------------------------------------------------
        adosc = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=12)
        dataframe['ADOSC'] = pd.Series(adosc[seq_len:])
        
        ##### OBV - On Balance Volume --------------------------------------------------------------
        obv = talib.OBV(close, volume)
        dataframe['OBV'] = pd.Series(obv[seq_len:])
        
        ##################################
        # Volatility Indicator Functions #
        ##################################
        
        ##### ATR - Average True Range 3/6/12/20 ---------------------------------------------------
        atr3 = talib.ATR(high, low, close, timeperiod=3)
        atr6 = talib.ATR(high, low, close, timeperiod=6)
        atr12 = talib.ATR(high, low, close, timeperiod=12)
        atr20 = talib.ATR(high, low, close, timeperiod=20)
        dataframe['ATR_3'] = pd.Series(atr3[seq_len:])
        dataframe['ATR_6'] = pd.Series(atr6[seq_len:])
        dataframe['ATR_12'] = pd.Series(atr12[seq_len:])
        dataframe['ATR_20'] = pd.Series(atr20[seq_len:])
        
        ##### TRANGE - True Range ------------------------------------------------------------------
        trange = talib.TRANGE(high, low, close)
        dataframe['TRANGE'] = pd.Series(trange[seq_len:])
        
        
        return(dataframe)