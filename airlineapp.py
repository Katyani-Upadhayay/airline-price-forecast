import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load dataset
df = pd.read_csv("Airline.csv")

# Convert date
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')

# Create Route column
df['Route'] = df['Source'] + " → " + df['Destination']

# Get unique routes
routes = df['Route'].unique()

# Streamlit app title
st.title("✈️ Airline Ticket Price Forecasting")

# Route selection
selected_route = st.selectbox("Select a Route:", routes)

# Filter data for selected route
df_route = df[df['Route'] == selected_route].copy()

# Aggregate daily average price
df_route = df_route.groupby('Date_of_Journey')['Price'].mean().reset_index()
df_route.columns = ['ds', 'y']

# Initialize Prophet model
model = Prophet()
model.fit(df_route)

# Create future dataframe (next 30 days)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_route['ds'], df_route['y'], label='Historical')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='Confidence Interval')
ax.set_title(f"30-Day Ticket Price Forecast for {selected_route}")
ax.set_xlabel("Date")
ax.set_ylabel("Average Price")
ax.legend()

# Display in Streamlit
st.pyplot(fig)

# Add slider for forecast period
forecast_days = st.slider("Select forecast period (days):", 7, 90, 30)

# Update future dataframe
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

st.subheader("Forecast Data")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days))

csv = forecast[['ds', 'yhat']].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Forecast as CSV",
    data=csv,
    file_name='forecast.csv',
    mime='text/csv',
)


