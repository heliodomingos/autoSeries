'''from fbprophet import Prophet


def predict_prophet(train, validation):
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=365)
    future.tail()
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig1 = m.plot(forecast)
    '''
