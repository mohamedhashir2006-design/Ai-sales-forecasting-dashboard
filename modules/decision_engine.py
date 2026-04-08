def get_sales_trend(forecast):
    recent = forecast.tail(7)
    previous = forecast.tail(14).head(7)

    trend = recent["yhat"].mean() - previous["yhat"].mean()

    return "down" if trend < 0 else "up"