# Code showing Linear Regression on amazon stock data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor



df = pd.read_csv(r"AMZN.csv")

df["Volatility"] =  (df["High"] - df["Low"]) / df["Close"]
df["PCT_Change"] = (df["Close"] - df["Open"] ) / df["Open"]



df = df[["Open", "Volume", "Volatility", "PCT_Change"]]
df.fillna(-999, inplace = True)

forecast_length = 60

df["Label"] = df["Open"].shift(-forecast_length)
df.dropna(inplace = True)

df_train = df.iloc[:4000]

X = np.array(df_train.drop(["Label"], axis = 1))
y = np.array(df_train["Label"])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = SGDRegressor()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(f"The accuracy of the model is {accuracy}")


data_X = df[["Open", "Volume", "Volatility", "PCT_Change"]].values
data_y = df["Label"].values
prices = df["Open"].values[4000:]

money = [10000]
worth = [money[0]]
bought = []
buy_signals = np.zeros(len(prices), dtype=bool)
sell_signals = np.zeros(len(prices), dtype=bool)


for i in range(len(prices)):
    
    start_idx = i+3500
    window_size = 500

    X_window = preprocessing.scale(data_X[start_idx:start_idx+window_size])
    y_window = data_y[start_idx:start_idx+window_size]
    
    
    clf.partial_fit(X_window, y_window)
    forecast = clf.predict(X_window[-1].reshape(1, -1))[0]
    
    
    
    if forecast > prices[i] * 1.02 and money[-1] > prices[i]:   # Buying a stock
        shares_bought = int(0.5 * money[-1] // prices[i])
        for j in range(shares_bought):
            bought.append(prices[i])
        money.append(money[-1] - prices[i] * shares_bought)
        buy_signals[i] = True
        
    elif (forecast < prices[i] * 0.98 and len(bought) > 0) or money[-1] < 100:  # Selling a stock
        money.append(money[-1] + prices[i] * len(bought))
        bought = []
        sell_signals[i] = True
        
    else: 
        money.append(money[-1])
    worth.append(money[-1] + len(bought)*prices[i])

money.append(money[-1] + prices[-1] * len(bought) )
        
plt.plot(worth, label = "Portfolio value")
plt.plot(money, label = "Cash")
plt.legend()
plt.xlabel("Time")

plt.show()


plt.plot(prices, label="Stock Price")
plt.scatter(np.where(buy_signals), prices[buy_signals], marker = "^", color = "g", label = "Buy")
plt.scatter(np.where(sell_signals), prices[sell_signals], marker = "v", color = "r", label = "Sell")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
        
        









