import os
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from numpy import sqrt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import json

df = pd.read_csv('D:/PyCharmProjects/forecastCostMedicalInsurance/data.csv', sep=',', header=0)
VALID_SIZE = 0.3
RAND_SEED = 8

def create_linear():
    features = ['age']
    target = ['charges']
    df_x = df[features]
    df_y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        df_x,
        df_y,
        test_size=VALID_SIZE,
        random_state=RAND_SEED,
        shuffle=True
    )
    lr = linear_model.LinearRegression()
    lin_model_charges = lr.fit(
        x_train[['age']],
        y_train[['charges']]
    )
    y_pred = lin_model_charges.predict(x_test[['age']])
    r2 = metrics.r2_score(y_test[['charges']], y_pred)
    rmse = sqrt(metrics.mean_squared_error(y_test[['charges']], y_pred))
    print(f"R2:{r2}")
    print(f"RMSE: {rmse}")
    if (r2 > 0.75):
        info = {'type': 'linear',
                'normalize': False,
                'features': 'age',
                'target': 'charges',
                'r2': r2,
                'rmse': rmse}
        with open('D:/PyCharmProjects/forecastCostMedicalInsurance/models/linear/info.json', 'w', encoding='utf-8') as filestream:
            json.dump(info, filestream, ensure_ascii=False)
        with open('D:/PyCharmProjects/forecastCostMedicalInsurance/models/linear/model.pickle', mode='wb') as filestream:
            pickle.dump(lin_model_charges, filestream)
        df_x.to_csv('D:/PyCharmProjects/forecastCostMedicalInsurance/models/linear/dfX.csv', index=False)
        df_y.to_csv('D:/PyCharmProjects/forecastCostMedicalInsurance/models/linear/dfY.csv', index=False)


def create_neural_network():
    epochForTrain = 100
    features = ['age', 'children', 'bmi', 'sex_male']
    target = ['charges']
    dfX = df[features]
    dfY = df[target]

    scalerNormX = MinMaxScaler()
    scalerNormX.fit(dfX)
    dfXNorm = pd.DataFrame(
        data=scalerNormX.transform(dfX),
        columns=dfX.columns,
        index=dfX.index
    )

    scalerNormY = MinMaxScaler()
    scalerNormY.fit(dfY)
    dfYNorm = pd.DataFrame(
        data=scalerNormY.transform(dfY),
        columns=dfY.columns,
        index=dfY.index
    )

    xNorm_train, xNorm_test, yNorm_train, yNorm_test = train_test_split(
        dfXNorm[features],
        dfYNorm[target],
        test_size=VALID_SIZE,
        random_state=RAND_SEED,
        shuffle=True
    )

    with tf.device('/CPU:0'):
        totalLossTrain = []
        totalLossTest = []

        input_size = 4
        output_size = 1

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_size,)))
        model.add(tf.keras.layers.Dense(units=4, activation='relu'))
        model.add(tf.keras.layers.Dense(units=4, activation='relu'))
        model.add(tf.keras.layers.Dense(units=output_size, activation='linear'))

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            metrics=[tf.keras.metrics.MeanSquaredError()]
        )

        print(model.summary())

        history = model.fit(
            xNorm_train[features],
            yNorm_train,
            validation_data=(xNorm_test[features], yNorm_test),
            epochs=epochForTrain,
            verbose=2
        )
        totalLossTrain.extend(history.history['loss'])
        if 'val_loss' in history.history.keys():
            totalLossTest.extend(history.history['val_loss'])

        yNorm_pred = model.predict(xNorm_test[features])

    # Правильный расчет метрик
    mse = metrics.mean_squared_error(yNorm_test, yNorm_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(yNorm_test, yNorm_pred)

    print(f"R2: {r2}")
    print(f"RMSE: {rmse}")

    if r2 > 0.75:
        info = {
            'type': 'neural network',
            'normalize': True,
            'features': features,
            'target': 'charges',
            'r2': float(r2),  # преобразуем numpy тип в python float
            'rmse': float(rmse),
            'scheme': '4->4->4->1'
        }
        os.makedirs('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network', exist_ok=True)
        with open('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network/info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False)
        with open('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network/normalizer.pickle', 'wb') as f:
            pickle.dump({'x': scalerNormX, 'y': scalerNormY}, f)
        model.save('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network/model.keras')
        dfX.to_csv('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network/dfX.csv', index=False)
        dfY.to_csv('D:/PyCharmProjects/forecastCostMedicalInsurance/models/neural_network/dfY.csv', index=False)

def main():
    create_linear()
    create_neural_network()

if __name__ == '__main__':
    main()
