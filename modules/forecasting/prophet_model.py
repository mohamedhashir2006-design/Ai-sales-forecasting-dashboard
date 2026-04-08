from prophet import Prophet

def train_prophet(df):
    model = Prophet()
    model.fit(df)
    return model

def forecast_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast