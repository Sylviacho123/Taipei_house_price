
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# 初始模型評估表格
dct = {'模型': [], '細節':[], 'RMSE (test)':[],
       'R2 (train)':[], 'adj. R2 (train)':[],
       'R2 (test)':[], 'adj. R2 (test)':[]}
df_eval = pd.DataFrame(dct)

# 讀取資料
df = pd.read_csv('Taipei_house.csv')

# one-hot encoding 行政區
df = pd.get_dummies(df, columns=['行政區'])

# 處理車位類別：有為1，無為0
df['車位類別'] = [0 if x=='無' else 1 for x in df['車位類別']]

# 建立惡意抬價欄位
df['是否惡意抬價'] = 0
district_cols = [col for col in df.columns if col.startswith('行政區_')]
for col in district_cols:
    idx = df[col] == 1
    avg = df.loc[idx, '總價'].mean()
    std = df.loc[idx, '總價'].std()
    df.loc[idx & (df['總價'] > avg + 2 * std), '是否惡意抬價'] = 1

# Adjusted R2 函數
def adj_R2(r2, n, k):
    return r2 - (k-1)/(n-k) * (1 - r2)

# 模型評估
from sklearn.metrics import mean_squared_error
def measurement(model, X_train, X_test):
    y_pred = model.predict(X_test)

    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 0)
    r2_train = round(model.score(X_train, y_train), 4)
    adj_r2_train = round(adj_R2(r2_train, X_train.shape[0], X_train.shape[1]), 4)
    r2_test = round(model.score(X_test, y_test), 4)
    adj_r2_test = round(adj_R2(r2_test, X_test.shape[0], X_test.shape[1]), 4)
    return [rmse, r2_train, adj_r2_train, r2_test, adj_r2_test]

# 建立特徵與目標欄位
features = df.drop(['總價', '交易日期', '經度', '緯度', '是否惡意抬價'], axis=1).columns
target = '總價'

# 訓練/測試切分
from sklearn.model_selection import train_test_split
seed = 42
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=seed)

# 模型列表
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

lst_model, lst_info = [], []

# 多元線性迴歸
lst_model.append(linear_model.LinearRegression())
lst_info.append(['多元迴歸','15 features'])

# Ridge
lst_model.append(linear_model.Ridge(alpha=10))
lst_info.append(['Ridge','15 features'])

# 多項式特徵
poly_fea = PolynomialFeatures(degree=2)
X_train_poly = poly_fea.fit_transform(X_train)
X_test_poly = poly_fea.fit_transform(X_test)

# 多項式迴歸
lst_model.append(linear_model.LinearRegression())
lst_info.append(['多項式迴歸','deg=2'])

# 多項式 + L1正規化
lst_model.append(linear_model.Lasso(alpha=10))
lst_info.append(['多項式迴歸+L1正規化','deg=2'])

# 評估每個模型
idx = df_eval.shape[0]
for i in range(len(lst_model)):
    if '多項式' in lst_info[i][0]:
        X_train_temp, X_test_temp = X_train_poly, X_test_poly
    else:
        X_train_temp, X_test_temp = X_train, X_test

    model = lst_model[i].fit(X_train_temp, y_train)
    row = lst_info[i] + measurement(model, X_train_temp, X_test_temp)
    df_eval.loc[idx+i] = row

print('對訓練集的最大 Adjusted R-squared: %.4f' % max(df_eval['adj. R2 (train)']))
print('對測試集的最小 RMSE:%d' % min(df_eval['RMSE (test)']))
print('兩個模型對測試集的最大 Adjusted R-squared: %.4f' %
      max(df_eval.loc[:1, 'adj. R2 (test)']))

# ================================================
# ❗ 預測房價 + 預測是否惡意抬價
# ================================================
X = df[features]
y = df[target]
X_poly = poly_fea.fit_transform(X)

# 建立新樣本（請根據實際特徵數調整）
new = np.array([36, 99, 32, 4, 4, 0, 3, 2, 1, 0, 0, 0, 0, 0, 1]).reshape(1, -1)
df_new = pd.DataFrame(new, columns=features)
df_poly_fea = poly_fea.fit_transform(df_new)

# 選擇最佳模型
lst = df_eval['adj. R2 (test)'].tolist()
idx = lst.index(max(lst))

if idx <= 1:
    model = lst_model[idx].fit(X, y)
    price = model.predict(df_new)[0]
else:
    model = lst_model[idx].fit(X_poly, y)
    price = model.predict(df_poly_fea)[0]

print(f'\n💰 預測房價：{int(price)} 萬元')

# =====================================================
# ⛔ 建立分類模型來預測是否惡意抬價
# =====================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_class = df[features]
y_class = df['是否惡意抬價']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=seed)

clf = RandomForestClassifier(n_estimators=100, random_state=seed)
clf.fit(Xc_train, yc_train)

# 預測新樣本是否有惡意抬價
is_overpriced = clf.predict(df_new)[0]
label = '有惡意抬價' if is_overpriced else '價格正常'
print(f'🕵️ 判斷結果：{label}')
