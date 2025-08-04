Price Forecasting
Overview:
This project predicts future airline ticket prices for different routes using Facebook Prophet for time series forecasting.
Users can select a route and view historical price trends, future predictions, confidence intervals, and download the forecasted data.

Features:
Forecast ticket prices for any route
Interactive Streamlit web app
Adjustable prediction horizon (7–90 days)
Confidence intervals for predictions
Download forecast results as CSV
Fully deployed on Streamlit Cloud

Tech Stack:
Python (Pandas, Matplotlib, Prophet)
Streamlit for web app
GitHub + Streamlit Cloud for deployment

Dataset:
The dataset contains flight records with columns:
Airline, Source, Destination, Date_of_Journey, Total_Stops, Price

Model Workflow:
Data preprocessing & cleaning
Aggregating daily average ticket prices
Training Prophet model for each route
Forecasting future prices
Deploying Streamlit app

