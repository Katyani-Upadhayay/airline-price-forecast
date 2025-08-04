import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("Airline.csv")
print(df)
print(df.head())

### Ensure the date column is successfully converted.
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
print(df.info())

### Date_of_Journey | Source | Destination | Price | Route with one row per route per day.
df_ts = df.groupby(['Date_of_Journey', 'Source', 'Destination'])['Price'].mean().reset_index()
df_ts['Route'] = df_ts['Source'] + " → " + df_ts['Destination']
print(df_ts.head())
print(df_ts['Route'].nunique())

###line chart with 5 different routes, showing how average ticket prices changed over time.
top_routes = df_ts['Route'].value_counts().head(5).index
print(top_routes)
df_top_routes = df_ts[df_ts['Route'].isin(top_routes)]
print(df_top_routes)

plt.figure(figsize=(14, 6))
for route in top_routes:
    route_data = df_top_routes[df_top_routes['Route'] == route]
    plt.plot(route_data['Date_of_Journey'], route_data['Price'], label=route)

plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title('Price Trends for Top 5 Routes')
plt.legend()
plt.show()

###one clean time series dataset (Date vs. Avg_Price) and a plotted trend for it.

selected_route = top_routes[0]
print("Selected Route:", selected_route)

df_route = df_top_routes[df_top_routes['Route'] == selected_route].copy()
df_route = df_route[['Date_of_Journey', 'Price']]
df_route.columns = ['Date', 'Avg_Price']
df_route = df_route.sort_values('Date')
print(df_route.head())

plt.figure(figsize=(12, 5))
plt.plot(df_route['Date'], df_route['Avg_Price'])
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title(f'Price Trend for {selected_route}')
plt.show()

###train data covers the earlier timeline and test data is the recent period.

split_point = int(len(df_route) * 0.8)
train = df_route.iloc[:split_point]
test = df_route.iloc[split_point:]

print("Train size:", len(train))
print("Test size:", len(test))

plt.figure(figsize=(12, 5))
plt.plot(train['Date'], train['Avg_Price'], label='Train')
plt.plot(test['Date'], test['Avg_Price'], label='Test')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title(f'Train/Test Split for {selected_route}')
plt.legend()
plt.show()


### a forecast chart showing:
# Historical prices
# Predicted future prices
# Uncertainty intervals

df_prophet = train.rename(columns={'Date': 'ds', 'Avg_Price': 'y'})
df_test = test.rename(columns={'Date': 'ds', 'Avg_Price': 'y'})

model = Prophet()
model.fit(df_prophet)

future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

model.plot(forecast)
plt.title(f'Price Forecast for {selected_route}')
plt.show()


###how close Prophet’s predictions are to the real ticket prices and get error values (MAE, RMSE)

forecast_test = forecast.iloc[-len(test):][['ds', 'yhat']]
forecast_test.columns = ['ds', 'Predicted']
eval_df = df_test.merge(forecast_test, on='ds')
print(eval_df.head())

mae = mean_absolute_error(eval_df['y'], eval_df['Predicted'])
rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['Predicted']))

print("MAE:", mae)
print("RMSE:", rmse)

plt.figure(figsize=(12, 5))
plt.plot(eval_df['ds'], eval_df['y'], label='Actual')
plt.plot(eval_df['ds'], eval_df['Predicted'], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title(f'Actual vs Predicted Prices for {selected_route}')
plt.legend()
plt.show()

df_extra = df[df['Source'] + " → " + df['Destination'] == selected_route].copy()

# Convert stops to numeric
df_extra['Total_Stops'] = df_extra['Total_Stops'].replace({'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4})
df_extra['Date_of_Journey'] = pd.to_datetime(df_extra['Date_of_Journey'], format='%d/%m/%Y')

# Aggregate with extra features
df_route_extra = df_extra.groupby('Date_of_Journey').agg({
    'Price': 'mean',
    'Total_Stops': 'mean'
}).reset_index()
df_route_extra.columns = ['ds', 'y', 'Stops']
df_route_extra.head()

model_adv = Prophet()
model_adv.add_regressor('Stops')

model_adv.fit(df_route_extra[['ds', 'y', 'Stops']])
future_adv = df_route_extra[['ds', 'Stops']]
forecast_adv = model_adv.predict(future_adv)

plt.figure(figsize=(12, 5))
plt.plot(df_route_extra['ds'], df_route_extra['y'], label='Actual')
plt.plot(forecast_adv['ds'], forecast_adv['yhat'], label='Predicted')
plt.title('Improved Prophet Model with Stops as Regressor')
plt.legend()
plt.show()

###a chart showing historical prices and predicted future ticket prices for the next 30 days

# Extend 30 days beyond the last date
future_dates = pd.date_range(start=df_route_extra['ds'].max(), periods=30, freq='D')

# For simplicity, assume average stops = mean of past stops
avg_stops = df_route_extra['Stops'].mean()

future_df = pd.DataFrame({
    'ds': future_dates,
    'Stops': avg_stops
})

forecast_future = model_adv.predict(future_df)
print(forecast_future)

plt.figure(figsize=(12, 5))
plt.plot(df_route_extra['ds'], df_route_extra['y'], label='Historical')
plt.plot(forecast_future['ds'], forecast_future['yhat'], label='Future Forecast')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.title(f'Future 30-Day Price Forecast for {selected_route}')
plt.legend()
plt.show()




