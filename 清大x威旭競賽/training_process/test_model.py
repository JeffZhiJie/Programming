import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import GroupKFold
import numpy as np

# === 讀取資料 ===
orderbook = pd.read_csv("train_orderbook_v1.csv")
trade = pd.read_csv("train_trade_v1.csv")

# === 特徵萃取函數 ===
def extract_features(order_df, trade_df):
    mid_price = (order_df['BidPrice1'] + order_df['AskPrice1']) / 2
    spread = order_df['AskPrice1'] - order_df['BidPrice1']

    bid_vol_total_series = order_df[['BidLots1', 'BidLots2', 'BidLots3']].sum(axis=1)
    ask_vol_total_series = order_df[['AskLots1', 'AskLots2', 'AskLots3']].sum(axis=1)

    depth_imbalance_series = ((order_df['BidLots1'] - order_df['AskLots1']) /
                              (order_df['BidLots1'] + order_df['AskLots1'] + 1e-6))
    
    depth_imbalance1 = depth_imbalance_series.mean()

    vwap_bid = ((order_df['BidPrice1'] * order_df['BidLots1'] +
                 order_df['BidPrice2'] * order_df['BidLots2'] +
                 order_df['BidPrice3'] * order_df['BidLots3']) /
                (order_df['BidLots1'] + order_df['BidLots2'] + order_df['BidLots3'] + 1e-6)).mean()

    vwap_ask = ((order_df['AskPrice1'] * order_df['AskLots1'] +
                 order_df['AskPrice2'] * order_df['AskLots2'] +
                 order_df['AskPrice3'] * order_df['AskLots3']) /
                (order_df['AskLots1'] + order_df['AskLots2'] + order_df['AskLots3'] + 1e-6)).mean()

    log_return = np.diff(np.log(mid_price + 1e-6))
    rolling_std = pd.Series(mid_price).rolling(window=5, min_periods=1).std()

    feature_dict = {
        'mid_price_mean': mid_price.mean(),
        'spread_mean': spread.mean(),
        'spread_std': spread.std(),
        'mid_price_range': mid_price.max() - mid_price.min(),
        'log_return_std': np.std(log_return),
        'rolling_std_mid_price': rolling_std.mean(),
        'bid_vol_total': bid_vol_total_series.sum(),
        'ask_vol_total': ask_vol_total_series.sum(),
        'depth_imbalance1': depth_imbalance1,
        'liquidity_score': ((order_df['BidLots1'] + order_df['AskLots1']) / (spread + 1e-6)).mean(),
        'vwap_bid': vwap_bid,
        'vwap_ask': vwap_ask,
        'mid_vs_vwap_avg': mid_price.mean() - ((vwap_bid + vwap_ask) / 2),
        'order_pressure': depth_imbalance_series.std(),
        'cancel_rate_bid': (order_df['BidLots1'].diff() < 0).sum() / (len(order_df) + 1e-6),
        'price_impact_bid': order_df['BidPrice1'].std(),
        'volatility_skew': np.var(vwap_bid - vwap_ask)
    }

    if len(trade_df) > 0:
        inout = trade_df['InOut'].to_numpy()
        fill_price = trade_df['FillPrice']
        fill_lots = trade_df['FillLots']
        buy = trade_df[trade_df['InOut'] == 1]
        sell = trade_df[trade_df['InOut'] == -1]

        feature_dict.update({
            'fillprice_mean': fill_price.mean(),
            'fillprice_std': fill_price.std(),
            'buy_ratio': len(buy) / (len(trade_df) + 1e-6),
            'vol_buy_minus_sell': buy['FillLots'].sum() - sell['FillLots'].sum(),
            'inout_imbalance_std': np.std(inout),
            'fill_density': len(trade_df) / (len(order_df) + 1e-6),
            'avg_fill_speed': fill_lots.sum() / (len(order_df) + 1e-6),
            'tick_imbalance': len(buy) / (len(trade_df) + 1e-6),
            'trade_fill_rate': fill_lots.sum() / (bid_vol_total_series + ask_vol_total_series).sum()
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

    return feature_dict


# === 萃取所有 Group 特徵 ===
train_data = []
group_ids = orderbook[orderbook['Weight'] == 1]['GroupID'].unique()
for gid in group_ids:
    ob = orderbook[orderbook['GroupID'] == gid]
    tr = trade[trade['GroupID'] == gid]
    features = extract_features(ob, tr)
    features['Target'] = ob.iloc[-1]['Target']
    features['GroupID'] = gid
    train_data.append(features)

df = pd.DataFrame(train_data)
X = df.drop(['Target', 'GroupID'], axis=1)
y = df['Target']
groups = df['GroupID']
feature_cols = list(X.columns)

# === LightGBM 參數 ===
params = {
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'seed': 42,
    'verbosity': -1
}

# === 用 GroupKFold 找出最佳迭代數 ===
best_iter = None
best_score = float('inf')
gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, y, groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val)

    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=100,
                      callbacks=[
                          lgb.early_stopping(50),
                          lgb.log_evaluation(50)
                          ])

    if model.best_score['valid_0']['l2'] < best_score:
        best_iter = model.best_iteration
        best_score = model.best_score['valid_0']['l2']

print(f"✅ 最佳迭代輪數: {best_iter}, 驗證集 MSE: {best_score:.6f}")

# === 用全部資料重新訓練模型 ===
final_train = lgb.Dataset(X, y)
final_model = lgb.train(params, final_train, num_boost_round=best_iter)

# === 儲存模型（不包含 feature_cols）===
joblib.dump(final_model, "model.pkl")
print("✅ 模型已儲存為 model.pkl")