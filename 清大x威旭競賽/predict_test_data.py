import os
import pickle
import argparse
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import math

BATCH_SIZE = 128

class ViciInference:
    def __init__(self, model_path):
        """
        Loading the pre-trained model.
        :param model_path: Path to the model file (pickle format).
        """
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Load the trained model
        :param model_path: Path to the model file (pickle format).
        :return: Loaded model.
        """
        # You can modify the way to load your trained model.
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            raise ValueError(f'Failed to load model: {e}')

    def _create_features(self, window, window_trade):

        window = window.filter(pl.col("Weight") == 1)

        # 計算 mid_price 與 spread
        mid_price = (window["BidPrice1"] + window["AskPrice1"]) / 2
        spread = window["AskPrice1"] - window["BidPrice1"]

        # 掛單量加總
        bid_vol_total = window.select(pl.sum_horizontal(["BidLots1", "BidLots2", "BidLots3"])).to_series()
        ask_vol_total = window.select(pl.sum_horizontal(["AskLots1", "AskLots2", "AskLots3"])).to_series()

        # 深度不均衡序列
        depth_imbalance_series = ((window["BidLots1"] - window["AskLots1"]) / 
                                (window["BidLots1"] + window["AskLots1"] + 1e-6))

        depth_imbalance1 = depth_imbalance_series.mean()

        # VWAP Bid / Ask
        vwap_bid = ((window["BidPrice1"] * window["BidLots1"] +
                    window["BidPrice2"] * window["BidLots2"] +
                    window["BidPrice3"] * window["BidLots3"]) /
                    (window["BidLots1"] + window["BidLots2"] + window["BidLots3"] + 1e-6)).mean()

        vwap_ask = ((window["AskPrice1"] * window["AskLots1"] +
                    window["AskPrice2"] * window["AskLots2"] +
                    window["AskPrice3"] * window["AskLots3"]) /
                    (window["AskLots1"] + window["AskLots2"] + window["AskLots3"] + 1e-6)).mean()

        mid_np = mid_price.to_numpy()
        log_return = np.diff(np.log(mid_np + 1e-6))
        log_return_std = np.std(log_return) if len(log_return) > 1 else 0.0
        rolling_std = pd.Series(mid_np).rolling(window=5, min_periods=1).std()
        rolling_std_val = rolling_std.mean()
        rolling_std_val = float(rolling_std_val) if not pd.isna(rolling_std_val) else 0.0
        price_impact_bid = window["BidPrice1"].std()

        if isinstance(vwap_bid, float) and isinstance(vwap_ask, float):
            volatility_skew = 0.0  # 因為是常數之差，無需計算 variance
        elif hasattr(vwap_bid, '__len__'):
            volatility_skew = float(np.var(vwap_bid - vwap_ask))

        feature_dict = {
            # 價格與波動性
            'mid_price_mean': mid_np.mean(),
            'spread_mean': float(spread.mean()),
            'spread_std': float(spread.std()) if not pd.isna(spread.std()) else 0.0,
            'mid_price_range': mid_np.max() - mid_np.min(),
            'log_return_std': log_return_std,
            'rolling_std_mid_price': rolling_std_val,

            # 掛單與流動性
            'bid_vol_total': bid_vol_total.sum(),
            'ask_vol_total': ask_vol_total.sum(),
            'depth_imbalance1': float(depth_imbalance1),
            'liquidity_score': ((window["BidLots1"] + window["AskLots1"]) / (spread + 1e-6)).mean(),

            # VWAP 與其他特徵
            'vwap_bid': float(vwap_bid),
            'vwap_ask': float(vwap_ask),
            'mid_vs_vwap_avg': mid_np.mean() - ((vwap_bid + vwap_ask) / 2),
            'order_pressure': float(depth_imbalance_series.std()) if depth_imbalance_series.std() is not None else 0.0,
            'cancel_rate_bid': (window["BidLots1"].diff().fill_null(0) < 0).sum() / (len(window) + 1e-6),
            'price_impact_bid': float(price_impact_bid) if price_impact_bid is not None else 0.0,
            'volatility_skew': volatility_skew
        }

        # 成交資訊
        if window_trade.shape[0] > 0:
            fill_price = window_trade["FillPrice"]
            fill_lots = window_trade["FillLots"]
            inout = window_trade["InOut"].to_numpy()
            buy = window_trade.filter(pl.col("InOut") == 1)
            sell = window_trade.filter(pl.col("InOut") == -1)

            feature_dict.update({
                'fillprice_mean': float(fill_price.mean()),
                'fillprice_std': float(fill_price.std()) if not pd.isna(fill_price.std()) else 0.0,
                'buy_ratio': buy.shape[0] / (window_trade.shape[0] + 1e-6),
                'vol_buy_minus_sell': buy["FillLots"].sum() - sell["FillLots"].sum(),
                'inout_imbalance_std': float(np.std(inout)),
                'fill_density': window_trade.shape[0] / (len(window) + 1e-6),
                'avg_fill_speed': fill_lots.sum() / (len(window) + 1e-6),
                'tick_imbalance': buy.shape[0] / (window_trade.shape[0] + 1e-6),
                'trade_fill_rate': fill_lots.sum() / (bid_vol_total.sum() + ask_vol_total.sum() + 1e-6)
            })
        else:
            feature_dict.update({
                'fillprice_mean': 0.0,
                'fillprice_std': 0.0,
                'buy_ratio': 0.0,
                'vol_buy_minus_sell': 0.0,
                'inout_imbalance_std': 0.0,
                'fill_density': 0.0,
                'avg_fill_speed': 0.0,
                'tick_imbalance': 0.0,
                'trade_fill_rate': 0.0
            })

        return pl.DataFrame([feature_dict])



    def _slice_window_for_features(self, test_orderbook, test_trade):
        """
        Perform inference on test data using a moving window.

        :param test_orderbook: Polars DataFrame containing the test orderbook.
        :param test_trade: Polars DataFrame containing the test trade.
        :return: List of predictions for the last row in each window.
        """
        if not isinstance(test_orderbook, pl.DataFrame):
            raise ValueError("test_orderbook must be a polars DataFrame")

        all_features = []
        
        for date_id in test_orderbook['DateID'].unique():

            # dataset divided by DateID
            date_orderbook = test_orderbook.filter(pl.col('DateID') == date_id)
            date_trade = test_trade.filter(pl.col('DateID') == date_id)
            
            groups_cnt = date_orderbook['GroupID'].n_unique()
            for cnt in tqdm(range(groups_cnt)):
                window = date_orderbook.slice(cnt * BATCH_SIZE, BATCH_SIZE)
                group_id = date_orderbook[cnt * BATCH_SIZE, 'GroupID']
                window_trade = date_trade.filter(pl.col('GroupID') == group_id)
                if len(window) < BATCH_SIZE:
                    continue
                
                # Create your features with window and window_trade here
                features = self._create_features(window, window_trade)
                all_features.append(features)
                
        all_features = pl.concat(all_features, how='vertical_relaxed')
        return all_features

    def predict(self, test_orderbook, test_trade):
        all_features = self._slice_window_for_features(test_orderbook, test_trade)
        # Predict for the features
        predictions = self.model.predict(all_features.to_pandas())
        
        pred_out = test_orderbook.filter(pl.col('Weight') == 1)[['RowID']]
        pred_out = pred_out.with_columns(pl.Series('Target', predictions.tolist()))
            
        return pred_out
        
        
def vici_predict(test_orderbook_path, test_trade_path, model_path, file_name): 
    
    test_orderbook = pl.read_csv(test_orderbook_path)
    test_trade = pl.read_csv(test_trade_path)
    
    # Initialize and predict
    model_inference = ViciInference(model_path)

    # Predict
    predictions = model_inference.predict(test_orderbook, test_trade)
    predictions.to_pandas().to_csv(file_name, index=False)
    return predictions

if __name__ == '__main__':
    
    model_path = './your_model'  # Modify the file name to your model
    file_name = './pred.csv'  # output file name
    
    parser = argparse.ArgumentParser(description="Paths for Python scripts.")
    parser.add_argument("-ob", "--orderbooks", required=True, help="Path to read orderbooks")
    parser.add_argument("-tr", "--trades", required=True, help="Path to read trades")
    
    parser.add_argument("-m", "--model", default=model_path, help="Path to trained model")
    parser.add_argument("-o", "--output", default=file_name, help="Path to output file")
    args = parser.parse_args()
    
    predictions = vici_predict(args.orderbooks, args.trades, args.model, args.output)