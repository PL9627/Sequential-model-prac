import warnings
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers

warnings.filterwarnings('ignore')

df = pd.read_csv('abalone.csv')
""" print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T) """

""" print(df.isnull().sum()) """

x = df['Sex'].value_counts()
labels = x.index
values = x.values
plt.pie(
    values,
    labels=labels,
    autopct='%1.1f%%'
)

features = df.loc[:, 'Length':'Shell weight'].columns
plt.subplots(figsize=(20, 10))
for i, feat in enumerate(features):
    plt.subplot(2, 4, i+1)
    sb.scatterplot(
        data=df,
        x=feat,
        y='Rings',
        hue='Sex'
    )

plt.subplots(figsize=(20, 10))
for i, feat in enumerate(features):
    plt.subplot(2, 4, i+1)
    sb.violinplot(
        data=df,
        x=feat,
        hue='Sex'
    )
plt.subplot(2, 4, 8)
sb.violinplot(
    data=df,
    x='Rings',
    hue='Sex'
)

features = df.drop('Rings', axis=1)
target = df['Rings']
X_train, X_val, Y_train, Y_val = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=22
)
""" print(X_train.shape, X_val.shape) """

model = keras.Sequential(
    [
        layers.Dense(256, activation='relu', input_shape=[8]),
        layers.BatchNormalization(),
        layers.Dense(256,activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(1, activation='relu')
    ]
)

model.compile(
    loss='mae',
    optimizer='adam',
    metrics=['mape']
)
model.summary()

history = model.fit(
    X_train,
    Y_train,
    epochs=50,
    verbose=1,
    batch_size=64,
    validation_data=(X_val, Y_val)
)

hist_df = pd.DataFrame(history.history)
""" print(hist_df.head()) """

hist_df['loss'].plot()
hist_df['val_loss'].plot()
plt.title('Loss vs Validation Loss')
plt.legend()

hist_df['mape'].plot()
hist_df['val_mape'].plot()
plt.title('MAPE vs Validation MAPE')
plt.legend()
""" plt.show() """