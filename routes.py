from flask import request, jsonify, render_template
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def fetch_stock_data(stock_symbol, start_date='2010-01-01', end_date='2020-01-01'):
    stock = yf.Ticker(stock_symbol)
    df = stock.history(start=start_date, end=end_date)
    df = df[['Close']]
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    stock_symbols = data['stocks']
    days_to_predict = data['days']
    scaler = MinMaxScaler(feature_range=(0, 1))

    company_data = []
    for symbol in stock_symbols:
        df = fetch_stock_data(symbol)
        df_scaled = scaler.fit_transform(df.values)
        company_data.append(df_scaled)

    # Create and train the LSTM model for each stock
    results = []
    for data in company_data:
        training_size = int(len(data) * 0.65)
        train_data = data[:training_size]
        test_data = data[training_size:]
        
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        model = create_lstm_model()
        model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=10, batch_size=64, verbose=1)
        
        # Predict the future
        x_input = test_data[len(test_data) - time_step:].reshape(1, -1)
        temp_input = list(x_input[0])
        
        lst_output = []
        i = 0
        while i < days_to_predict:
            if len(temp_input) > time_step:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, time_step, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i += 1
            else:
                x_input = x_input.reshape((1, time_step, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i += 1
        
        lst_output = scaler.inverse_transform(lst_output)
        df_pred = np.concatenate((data, lst_output), axis=0)

        # Generate plots
        plt.figure(figsize=(10,6))
        plt.plot(scaler.inverse_transform(data), label='Original Data')
        plt.plot(np.arange(len(data), len(data) + days_to_predict), lst_output, label='Predicted Data')
        plt.legend(loc='upper left')
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        results.append({'symbol': symbol, 'plot_url': plot_url})

    return jsonify(results)
