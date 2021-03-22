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
        
        ##### BBANDS - Bollinger Bands - 5 / 10 / 20 -----------------------------------------------
        upper5,_, lower5 = talib.BBANDS(self.data['close'], 5)
        upper10,_, lower10 = talib.BBANDS(self.data['close'], 10)
        upper20,_, lower20 = talib.BBANDS(self.data['close'], 20)
        
        # Carregar apenas os últimos 'seq_len' dados
        # frame = {'BBANDS_Upper_5':upper5[seq_len:], 'BBANDS_Lower_5':lower5[seq_len:],
        #          'BBANDS_Upper_10':upper10[seq_len:], 'BBANDS_Lower_10':lower10[seq_len:],
        #          'BBANDS_Upper_20':upper20[seq_len:], 'BBANDS_Lower_20':lower20[seq_len:]
        #          }
        
        dataframe['BBANDS_Upper_5'] = pd.Series(upper5[seq_len:])
        dataframe['BBANDS_Lower_5'] = pd.Series(lower5[seq_len:])
        dataframe['BBANDS_Upper_10'] = pd.Series(upper10[seq_len:])
        dataframe['BBANDS_Lower_10'] = pd.Series(lower10[seq_len:])
        dataframe['BBANDS_Upper_20'] = pd.Series(upper20[seq_len:])
        dataframe['BBANDS_Lower_20'] = pd.Series(lower20[seq_len:])
        
        # Cria um DataFrame usando o frame com nome da coluna seguido dos dados
        #dataframe = pd.DataFrame(frame)

        ##### DEMA - Double Exponential Moving Average - 10 ----------------------------------------
        dema = talib.DEMA(self.data['close'], timeperiod=10)
        # Adiciona um Series à direita de um DataFrame
        dataframe['DEMA_20'] = pd.Series(dema[seq_len:])
        
        ##### EMA - Exponential Moving Average - 20 ------------------------------------------------
        ema5c = talib.EMA(self.data['close'], timeperiod=5)
        ema10c = talib.EMA(self.data['close'], timeperiod=10)
        ema20c = talib.EMA(self.data['close'], timeperiod=20)
        
        ema5o = talib.EMA(self.data['open'], timeperiod=5)
        ema10o = talib.EMA(self.data['open'], timeperiod=10)
        ema20o = talib.EMA(self.data['open'], timeperiod=20)

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
       
        ##### SAR - Parabolic SAR - 0.02/0.06/0.10/0.14/0.18 ---------------------------------------
        sar2 = talib.SAR(high, low, acceleration=0.02)
        sar6 = talib.SAR(high, low, acceleration=0.06)
        sar10 = talib.SAR(high, low, acceleration=0.10)
        sar14 = talib.SAR(high, low, acceleration=0.14)
        sar18 = talib.SAR(high, low, acceleration=0.18)
        dataframe['SAR_2'] = pd.Series(sar2[seq_len:])
        dataframe['SAR_6'] = pd.Series(sar6[seq_len:])
        dataframe['SAR_10'] = pd.Series(sar10[seq_len:])
        dataframe['SAR_14'] = pd.Series(sar14[seq_len:])
        dataframe['SAR_18'] = pd.Series(sar18[seq_len:])
        
        ##### TEMA - Triple Exponential Moving Average - 5 -----------------------------------------
        tema5 = talib.TEMA(close, timeperiod=5)
        dataframe['TEMA_5'] = pd.Series(tema5[seq_len:])
        
        ##### TRIMA - Triangular Moving Average - 5/10/20 ------------------------------------------
        trima5 = talib.TRIMA(close, timeperiod=5)
        trima10 = talib.TRIMA(close, timeperiod=10)
        trima20 = talib.TRIMA(close, timeperiod=20)
        dataframe['TRIMA_5'] = pd.Series(trima5[seq_len:])
        dataframe['TRIMA_10'] = pd.Series(trima10[seq_len:])
        dataframe['TRIMA_20'] = pd.Series(trima20[seq_len:])
        
        ##### WMA - Weighted Moving Average - 5/10/20 ----------------------------------------------
        wma5 = talib.WMA(close, timeperiod=5)
        wma10 = talib.WMA(close, timeperiod=10)
        wma20 = talib.WMA(close, timeperiod=20)
        dataframe['WMA_5'] = pd.Series(wma5[seq_len:])
        dataframe['WMA_10'] = pd.Series(wma10[seq_len:])
        dataframe['WMA_20'] = pd.Series(wma20[seq_len:])
        
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
        
        ##### ADXR - Average Directional Movement Index Rating 3/6 ---------------------------------
        adxr3 = talib.ADXR(high, low, close, timeperiod=3)
        adxr6 = talib.ADXR(high, low, close, timeperiod=6)
        dataframe['ADXR_3'] = pd.Series(adxr3[seq_len:])
        dataframe['ADXR_6'] = pd.Series(adxr6[seq_len:])

        ##### APO - Absolute Price Oscillator 3-12/5-12 --------------------------------------------
        apo3_12 = talib.APO(close, fastperiod=3, slowperiod=12, matype=0)
        apo5_12 = talib.APO(close, fastperiod=5, slowperiod=12, matype=0)
        dataframe['APO_3_12'] = pd.Series(apo3_12[seq_len:])
        dataframe['APO_5_12'] = pd.Series(apo5_12[seq_len:])
        
        ##### AROONOSC - Aroon Oscillator 3/6/12 ---------------------------------------------------
        aroonosc3 = talib.AROONOSC(high, low, timeperiod=3)
        aroonosc6 = talib.AROONOSC(high, low, timeperiod=6)
        aroonosc12 = talib.AROONOSC(high, low, timeperiod=12)
        dataframe['AROONOSC_3'] = pd.Series(aroonosc3[seq_len:])
        dataframe['AROONOSC_6'] = pd.Series(aroonosc6[seq_len:])
        dataframe['AROONOSC_12'] = pd.Series(aroonosc12[seq_len:])
        
        ##### BOP - Balance Of Power
        bop = talib.BOP(oopen, high, low, close)
        dataframe['BOP'] = pd.Series(bop[seq_len:])
        
        ##### CCI - Commodity Channel Index 3/6/12/20 ----------------------------------------------
        cci3 = talib.CCI(high, low, close, timeperiod=3)
        cci6 = talib.CCI(high, low, close, timeperiod=6)
        cci12 = talib.CCI(high, low, close, timeperiod=12)
        cci20 = talib.CCI(high, low, close, timeperiod=20)
        dataframe['CCI_3'] = pd.Series(cci3[seq_len:])
        dataframe['CCI_6'] = pd.Series(cci6[seq_len:])
        dataframe['CCI_12'] = pd.Series(cci12[seq_len:])
        dataframe['CCI_20'] = pd.Series(cci20[seq_len:])
        
        ##### CMO - Chande Momentum Oscillator 3/6/12/20 -------------------------------------------
        cmo3 = talib.CMO(close, timeperiod=3)
        cmo6 = talib.CMO(close, timeperiod=6)
        cmo12 = talib.CMO(close, timeperiod=12)
        cmo20 = talib.CMO(close, timeperiod=20)
        dataframe['CMO_3'] = pd.Series(cmo3[seq_len:])
        dataframe['CMO_6'] = pd.Series(cmo6[seq_len:])
        dataframe['CMO_12'] = pd.Series(cmo12[seq_len:])
        dataframe['CMO_20'] = pd.Series(cmo20[seq_len:])
        
        ##### DX - Directional Movement Index 3/6/12/20 --------------------------------------------
        dx3 = talib.DX(high, low, close, timeperiod=3)
        dx6 = talib.DX(high, low, close, timeperiod=6)
        dx12 = talib.DX(high, low, close, timeperiod=12)
        dx20 = talib.DX(high, low, close, timeperiod=20)
        dataframe['DX_3'] = pd.Series(dx3[seq_len:])
        dataframe['DX_6'] = pd.Series(dx6[seq_len:])
        dataframe['DX_12'] = pd.Series(dx12[seq_len:])
        dataframe['DX_20'] = pd.Series(dx20[seq_len:])
        
        ##### MFI - Money Flow Index 3/6/12 --------------------------------------------------------
        mfi3 = talib.MFI(high, low, close, volume, timeperiod=3)
        mfi6 = talib.MFI(high, low, close, volume, timeperiod=6)
        mfi12 = talib.MFI(high, low, close, volume, timeperiod=12)    
        dataframe['MFI_3'] = pd.Series(mfi3[seq_len:])
        dataframe['MFI_6'] = pd.Series(mfi6[seq_len:])
        dataframe['MFI_12'] = pd.Series(mfi12[seq_len:])

        ##### MINUS_DI - Minus Directional Indicator 3/6/12/20 -------------------------------------
        minusdi3 = talib.MINUS_DI(high, low, close, timeperiod=3)
        minusdi6 = talib.MINUS_DI(high, low, close, timeperiod=6)
        minusdi12 = talib.MINUS_DI(high, low, close, timeperiod=12)
        minusdi20 = talib.MINUS_DI(high, low, close, timeperiod=20)
        dataframe['MINUS_DI_3'] = pd.Series(minusdi3[seq_len:])
        dataframe['MINUS_DI_6'] = pd.Series(minusdi6[seq_len:])
        dataframe['MINUS_DI_12'] = pd.Series(minusdi12[seq_len:])
        dataframe['MINUS_DI_20'] = pd.Series(minusdi20[seq_len:])
        
        ##### MINUS_DM - Minus Directional Movement 3/6/12/20 --------------------------------------
        minusdm3 = talib.MINUS_DM(high, low, timeperiod=3)
        minusdm6 = talib.MINUS_DM(high, low, timeperiod=6)
        minusdm12 = talib.MINUS_DM(high, low, timeperiod=12)
        minusdm20 = talib.MINUS_DM(high, low, timeperiod=20)
        dataframe['MINUS_DM_3'] = pd.Series(minusdm3[seq_len:])
        dataframe['MINUS_DM_6'] = pd.Series(minusdm6[seq_len:])
        dataframe['MINUS_DM_12'] = pd.Series(minusdm12[seq_len:])
        dataframe['MINUS_DM_20'] = pd.Series(minusdm20[seq_len:])
        
        ##### MOM - Momentum 3/6/10/14 -------------------------------------------------------------
        mom3 = talib.MOM(close, timeperiod=3)
        mom6 = talib.MOM(close, timeperiod=6)
        mom10 = talib.MOM(close, timeperiod=10)
        mom14 = talib.MOM(close, timeperiod=14)
        dataframe['MOM_3'] = pd.Series(mom3[seq_len:])
        dataframe['MOM_6'] = pd.Series(mom6[seq_len:])
        dataframe['MOM_10'] = pd.Series(mom10[seq_len:])
        dataframe['MOM_14'] = pd.Series(mom14[seq_len:])
        
        ##### PLUS_DI - Plus Directional Indicator 3/6/12/20 ---------------------------------------
        plusdi3 = talib.PLUS_DI(high, low, close, timeperiod=3)
        plusdi6 = talib.PLUS_DI(high, low, close, timeperiod=6)
        plusdi12 = talib.PLUS_DI(high, low, close, timeperiod=12)
        plusdi20 = talib.PLUS_DI(high, low, close, timeperiod=20)
        dataframe['PLUS_DI_3'] = pd.Series(plusdi3[seq_len:])
        dataframe['PLUS_DI_6'] = pd.Series(plusdi6[seq_len:])
        dataframe['PLUS_DI_12'] = pd.Series(plusdi12[seq_len:])
        dataframe['PLUS_DI_20'] = pd.Series(plusdi20[seq_len:])
        
        ##### PLUS_DM - Plus Directional Movement 3/6/12/20 ----------------------------------------
        plusdm3 = talib.PLUS_DM(high, low, timeperiod=3)
        plusdm6 = talib.PLUS_DM(high, low, timeperiod=6)
        plusdm12 = talib.PLUS_DM(high, low, timeperiod=12)
        plusdm20 = talib.PLUS_DM(high, low, timeperiod=20)
        dataframe['PLUS_DM_3'] = pd.Series(plusdm3[seq_len:])
        dataframe['PLUS_DM_6'] = pd.Series(plusdm6[seq_len:])
        dataframe['PLUS_DM_12'] = pd.Series(plusdm12[seq_len:])
        dataframe['PLUS_DM_20'] = pd.Series(plusdm20[seq_len:])    
        
        ##### PPO - Percentage Price Oscillator 3-12/12-20 -----------------------------------------
        ppo3_12 = talib.PPO(close, fastperiod=3, slowperiod=12, matype=0)
        ppo12_20 = talib.PPO(close, fastperiod=12, slowperiod=20, matype=0)
        dataframe['PPO_3_12'] = pd.Series(ppo3_12[seq_len:])
        dataframe['PPO_12_20'] = pd.Series(ppo12_20[seq_len:])
        
        ##### ROC - Rate of change : ((price/prevPrice)-1)*100 3/5/7/10 ----------------------------
        roc3 = talib.ROC(close, timeperiod=3)
        roc5 = talib.ROC(close, timeperiod=5)
        roc7 = talib.ROC(close, timeperiod=7)
        roc10 = talib.ROC(close, timeperiod=10)
        dataframe['ROC_3'] = pd.Series(roc3[seq_len:])
        dataframe['ROC_5'] = pd.Series(roc5[seq_len:])
        dataframe['ROC_7'] = pd.Series(roc7[seq_len:])
        dataframe['ROC_10'] = pd.Series(roc10[seq_len:])
        
        ##### ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice 3/5/7/10 ---------------
        rocp3 = talib.ROCP(close, timeperiod=3)
        rocp5 = talib.ROCP(close, timeperiod=5)
        rocp7 = talib.ROCP(close, timeperiod=7)
        rocp10 = talib.ROCP(close, timeperiod=10)
        dataframe['ROCP_3'] = pd.Series(rocp3[seq_len:])
        dataframe['ROCP_5'] = pd.Series(rocp5[seq_len:])
        dataframe['ROCP_7'] = pd.Series(rocp7[seq_len:])
        dataframe['ROCP_10'] = pd.Series(rocp10[seq_len:])        
        
        ##### ROCR - Rate of change ratio: (price/prevPrice) 3/5/7/10 ------------------------------
        rocr3 = talib.ROCR(close, timeperiod=3)
        rocr5 = talib.ROCR(close, timeperiod=5)
        rocr7 = talib.ROCR(close, timeperiod=7)
        rocr10 = talib.ROCR(close, timeperiod=10)
        dataframe['ROCR_3'] = pd.Series(rocr3[seq_len:])
        dataframe['ROCR_5'] = pd.Series(rocr5[seq_len:])
        dataframe['ROCR_7'] = pd.Series(rocr7[seq_len:])
        dataframe['ROCR_10'] = pd.Series(rocr10[seq_len:])
                
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
        
        ##### ULTOSC - Ultimate Oscillator 3-6-12/5-10-20 ------------------------------------------
        ultosc3612 = talib.ULTOSC(high, low, close, timeperiod1=3, timeperiod2=6, timeperiod3=12)
        ultosc51020 = talib.ULTOSC(high, low, close, timeperiod1=5, timeperiod2=10, timeperiod3=20)
        dataframe['ULTOSC_3_6_12'] = pd.Series(ultosc3612[seq_len:])
        dataframe['ULTOSC_5_10_20'] = pd.Series(ultosc51020[seq_len:])
        
        ##### WILLR - Williams' %R 3/6/12/20 -------------------------------------------------------
        willr3 = talib.WILLR(high, low, close, timeperiod=3)
        willr6 = talib.WILLR(high, low, close, timeperiod=6)
        willr12 = talib.WILLR(high, low, close, timeperiod=12)
        willr20 = talib.WILLR(high, low, close, timeperiod=20)
        dataframe['WILLR_3'] = pd.Series(willr3[seq_len:])
        dataframe['WILLR_6'] = pd.Series(willr6[seq_len:])
        dataframe['WILLR_12'] = pd.Series(willr12[seq_len:])
        dataframe['WILLR_20'] = pd.Series(willr20[seq_len:])

        '''    
        ##### MACD - Moving Average Convergence/Divergence
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        ##### MACDEXT - MACD with controllable MA type
        macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        
        ##### MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        macd, macdsignal, macdhist = MACDFIX(close, signalperiod=9)
        
        ##### ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100 3/5/7/10 -------------
        rocr1003 = talib.ROCR100(close, timeperiod=3)
        rocr1005 = talib.ROCR100(close, timeperiod=5)
        rocr1007 = talib.ROCR100(close, timeperiod=7)
        rocr10010 = talib.ROCR100(close, timeperiod=10)
        dataframe['ROCR100_3'] = pd.Series(rocr1003[seq_len:])
        dataframe['ROCR100_5'] = pd.Series(rocr1005[seq_len:])
        dataframe['ROCR100_7'] = pd.Series(rocr1007[seq_len:])
        dataframe['ROCR100_10'] = pd.Series(rocr10010[seq_len:])

        ##### STOCH - Stochastic
        slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        
        ##### STOCHF - Stochastic Fast
        fastk, fastd = STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
        
        ##### STOCHRSI - Stochastic Relative Strength Index
        fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

        '''
        
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