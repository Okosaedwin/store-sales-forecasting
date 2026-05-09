# %% [markdown]
# # 📈 Store Sales Forecasting — Kaggle Time Series Competition
# 
# **Objective:** Predict future monthly store sales using time series & ML models.
# 
# **Dataset:** Favorita Grocery Stores (Ecuador) — 5 years of daily sales data
# 
# **Models:**
# 1. Moving Average (baseline)
# 2. Exponential Smoothing (Holt-Winters)
# 3. SARIMAX
# 4. Linear Regression
# 5. Random Forest

# %% [markdown]
# ## 1. Install Required Libraries

# %%
!pip install pandas numpy matplotlib plotly scikit-learn statsmodels --quiet
print("✅ Libraries installed!")

# %% [markdown]
# ## 2. Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

print("✅ All libraries imported successfully!")

# %% [markdown]
# ## 3. Load All Datasets

# %%
# Load the datasets — UPDATE PATHS if your files are in a different location
train = pd.read_csv("train.csv", parse_dates=["date"])
stores = pd.read_csv("stores.csv")
oil = pd.read_csv("oil.csv", parse_dates=["date"])
holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])
transactions = pd.read_csv("transactions.csv", parse_dates=["date"])

print(f"📦 Train:        {train.shape[0]:,} rows × {train.shape[1]} columns")
print(f"🏪 Stores:       {stores.shape[0]} stores")
print(f"🛢️ Oil:          {oil.shape[0]:,} rows")
print(f"🎉 Holidays:     {holidays.shape[0]} events")
print(f"🧾 Transactions: {transactions.shape[0]:,} rows")

print(f"\n📅 Date Range: {train['date'].min().date()} to {train['date'].max().date()}")

# %%
# Preview train data
print("Train columns:", list(train.columns))
train.head(10)

# %%
# Preview stores
stores.head()

# %%
# Check for missing values
print("Missing values in train:")
print(train.isnull().sum())
print(f"\nMissing oil prices: {oil['dcoilwtico'].isnull().sum()} out of {len(oil)}")

# %% [markdown]
# ## 4. Exploratory Data Analysis (EDA)

# %%
# Merge train with store info
df = train.merge(stores, on="store_nbr", how="left")
print(f"Merged dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()

# %%
# Basic statistics
df[["sales", "onpromotion"]].describe()

# %%
# Total daily sales (all stores combined)
daily_sales = df.groupby("date", as_index=False)["sales"].sum()
daily_sales = daily_sales.sort_values("date")

fig = px.line(
    daily_sales, x="date", y="sales",
    title="Total Daily Sales (All Stores Combined)",
    labels={"sales": "Sales", "date": ""},
)
fig.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)")
fig.show()

# %%
# Sales by product family (top 10)
family_sales = df.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False).head(10)

fig = px.bar(
    family_sales, x="sales", y="family", orientation="h",
    title="Top 10 Product Families by Total Sales",
    color="sales", color_continuous_scale="Viridis",
)
fig.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
fig.show()

# %%
# Sales by store type
type_sales = df.groupby("type", as_index=False)["sales"].sum()

fig = px.pie(
    type_sales, values="sales", names="type",
    title="Sales by Store Type", hole=0.4,
    color_discrete_sequence=["#4a90d9", "#50c878", "#e76f51", "#f4a261", "#9b59b6"],
)
fig.update_traces(textinfo="percent+label")
fig.show()

# %%
# Sales by city (top 10)
city_sales = df.groupby("city", as_index=False)["sales"].sum().sort_values("sales", ascending=False).head(10)

fig = px.bar(
    city_sales, x="city", y="sales",
    title="Top 10 Cities by Sales",
    color="sales", color_continuous_scale="RdYlGn",
)
fig.update_layout(template="plotly_white", coloraxis_showscale=False)
fig.show()

# %% [markdown]
# ## 5. Aggregate to Monthly Sales
# 
# For time series forecasting, we aggregate to **monthly totals**.

# %%
# Monthly aggregation
monthly_sales = (
    daily_sales
    .set_index("date")
    .resample("MS")["sales"]
    .sum()
    .reset_index()
)
monthly_sales.columns = ["Date", "Sales"]
monthly_sales = monthly_sales.sort_values("Date").reset_index(drop=True)

print(f"Monthly data: {len(monthly_sales)} months")
print(f"Date range: {monthly_sales['Date'].min().date()} to {monthly_sales['Date'].max().date()}")

# Plot monthly trend
fig = px.line(
    monthly_sales, x="Date", y="Sales",
    title="Monthly Sales Trend (All Stores)",
    markers=True,
)
fig.update_layout(template="plotly_white", yaxis_title="Total Sales")
fig.show()

# %% [markdown]
# ## 6. Time Series Decomposition

# %%
# Decompose the time series
ts = monthly_sales.set_index("Date")["Sales"]

decomposition = seasonal_decompose(ts, model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomposition.observed.plot(ax=axes[0], title="Observed (Original Sales)")
decomposition.trend.plot(ax=axes[1], title="Trend")
decomposition.seasonal.plot(ax=axes[2], title="Seasonality (12-month cycle)")
decomposition.resid.plot(ax=axes[3], title="Residual (Noise)")

for ax in axes:
    ax.set_xlabel("")

plt.tight_layout()
plt.show()

print("📊 Look for: upward/downward trends and repeating seasonal patterns.")

# %%
# ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts.dropna(), lags=24, ax=axes[0], title="Autocorrelation (ACF)")
plot_pacf(ts.dropna(), lags=24, ax=axes[1], title="Partial Autocorrelation (PACF)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Train-Test Split

# %%
# Use last 6 months as test set
FORECAST_MONTHS = 6

train_data = monthly_sales.iloc[:-FORECAST_MONTHS]
test_data = monthly_sales.iloc[-FORECAST_MONTHS:]

print(f"Training: {len(train_data)} months ({train_data['Date'].min().date()} → {train_data['Date'].max().date()})")
print(f"Testing:  {len(test_data)} months ({test_data['Date'].min().date()} → {test_data['Date'].max().date()})")

# Plot the split
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_data["Date"], y=train_data["Sales"], name="Training", line=dict(color="#4a90d9")))
fig.add_trace(go.Scatter(x=test_data["Date"], y=test_data["Sales"], name="Test (Actual)", line=dict(color="#e76f51", width=3)))
fig.update_layout(title="Train-Test Split", template="plotly_white", yaxis_title="Sales")
fig.show()

# %% [markdown]
# ## 8. Model 1: Moving Average (Baseline)

# %%
WINDOW = 3

ma_predictions = []
history = list(train_data["Sales"].values)

for i in range(FORECAST_MONTHS):
    avg = np.mean(history[-WINDOW:])
    ma_predictions.append(avg)
    history.append(test_data["Sales"].values[i])

ma_mae = mean_absolute_error(test_data["Sales"], ma_predictions)
ma_rmse = np.sqrt(mean_squared_error(test_data["Sales"], ma_predictions))
ma_r2 = r2_score(test_data["Sales"], ma_predictions)

print(f"📊 Moving Average (window={WINDOW})")
print(f"   MAE:  {ma_mae:,.2f}")
print(f"   RMSE: {ma_rmse:,.2f}")
print(f"   R²:   {ma_r2:.4f}")

# %% [markdown]
# ## 9. Model 2: Holt-Winters (Exponential Smoothing)

# %%
train_ts = train_data.set_index("Date")["Sales"]

hw_model = ExponentialSmoothing(
    train_ts,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
).fit(optimized=True)

hw_predictions = hw_model.forecast(FORECAST_MONTHS)

hw_mae = mean_absolute_error(test_data["Sales"], hw_predictions)
hw_rmse = np.sqrt(mean_squared_error(test_data["Sales"], hw_predictions))
hw_r2 = r2_score(test_data["Sales"], hw_predictions)

print(f"📊 Holt-Winters Exponential Smoothing")
print(f"   MAE:  {hw_mae:,.2f}")
print(f"   RMSE: {hw_rmse:,.2f}")
print(f"   R²:   {hw_r2:.4f}")

# %% [markdown]
# ## 10. Model 3: SARIMAX

# %%
sarimax_model = SARIMAX(
    train_ts,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

sarimax_predictions = sarimax_model.forecast(FORECAST_MONTHS)

sarimax_mae = mean_absolute_error(test_data["Sales"], sarimax_predictions)
sarimax_rmse = np.sqrt(mean_squared_error(test_data["Sales"], sarimax_predictions))
sarimax_r2 = r2_score(test_data["Sales"], sarimax_predictions)

print(f"📊 SARIMAX (1,1,1)(1,1,1,12)")
print(f"   MAE:  {sarimax_mae:,.2f}")
print(f"   RMSE: {sarimax_rmse:,.2f}")
print(f"   R²:   {sarimax_r2:.4f}")

print("\n--- Model Summary ---")
print(sarimax_model.summary())

# %% [markdown]
# ## 11. Model 4: Linear Regression

# %%
def create_features(df):
    df = df.copy()
    df["Month_Num"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    df["Month_Index"] = range(len(df))
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month_Num"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month_Num"] / 12)
    return df

train_feat = create_features(train_data)
test_feat = create_features(test_data)

features = ["Month_Index", "Month_Num", "Year", "Quarter", "Month_Sin", "Month_Cos"]

X_train = train_feat[features]
y_train = train_feat["Sales"]
X_test = test_feat[features]
y_test = test_feat["Sales"]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)

print(f"📊 Linear Regression")
print(f"   MAE:  {lr_mae:,.2f}")
print(f"   RMSE: {lr_rmse:,.2f}")
print(f"   R²:   {lr_r2:.4f}")

# %% [markdown]
# ## 12. Model 5: Random Forest

# %%
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print(f"📊 Random Forest Regressor")
print(f"   MAE:  {rf_mae:,.2f}")
print(f"   RMSE: {rf_rmse:,.2f}")
print(f"   R²:   {rf_r2:.4f}")

# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(f"\n🔍 Feature Importance:")
print(importance.to_string(index=False))

# %% [markdown]
# ## 13. Compare All Models

# %%
results = pd.DataFrame({
    "Model": ["Moving Average", "Holt-Winters", "SARIMAX", "Linear Regression", "Random Forest"],
    "MAE": [ma_mae, hw_mae, sarimax_mae, lr_mae, rf_mae],
    "RMSE": [ma_rmse, hw_rmse, sarimax_rmse, lr_rmse, rf_rmse],
    "R²": [ma_r2, hw_r2, sarimax_r2, lr_r2, rf_r2],
})
results_sorted = results.sort_values("MAE")

# Find best model
best_model = results_sorted.iloc[0]["Model"]

print("=" * 70)
print("            📊 MODEL COMPARISON — STORE SALES FORECASTING")
print("=" * 70)

for _, row in results_sorted.iterrows():
    marker = "🏆" if row["Model"] == best_model else "  "
    print(f" {marker} {row['Model']:<22} MAE: {row['MAE']:>14,.2f}   RMSE: {row['RMSE']:>14,.2f}   R²: {row['R²']:>8.4f}")

print("=" * 70)
print(f"\n🏆 Best model: {best_model} (lowest MAE)")

# %%
# Visual comparison
fig = go.Figure()

fig.add_trace(go.Scatter(x=train_data["Date"], y=train_data["Sales"], name="Training Data", line=dict(color="#4a90d9", width=2)))
fig.add_trace(go.Scatter(x=test_data["Date"], y=test_data["Sales"], name="Actual (Test)", line=dict(color="#1e3a5f", width=3), mode="lines+markers"))
fig.add_trace(go.Scatter(x=test_data["Date"], y=ma_predictions, name="Moving Average", line=dict(color="#f4a261", dash="dash")))
fig.add_trace(go.Scatter(x=test_data["Date"], y=hw_predictions.values, name="Holt-Winters", line=dict(color="#50c878", dash="dash")))
fig.add_trace(go.Scatter(x=test_data["Date"], y=sarimax_predictions.values, name="SARIMAX", line=dict(color="#e76f51", dash="dash")))
fig.add_trace(go.Scatter(x=test_data["Date"], y=lr_predictions, name="Linear Regression", line=dict(color="#9b59b6", dash="dash")))
fig.add_trace(go.Scatter(x=test_data["Date"], y=rf_predictions, name="Random Forest", line=dict(color="#e74c3c", dash="dot")))

fig.update_layout(
    title="📈 All Models vs Actual Sales",
    xaxis_title="", yaxis_title="Sales",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.15),
    height=500,
)
fig.show()

# %% [markdown]
# ## 14. Forecast Future Sales (Next 6 Months)

# %%
FUTURE_MONTHS = 6
full_ts = monthly_sales.set_index("Date")["Sales"]

# Holt-Winters on full data
hw_full = ExponentialSmoothing(
    full_ts, trend="add", seasonal="add", seasonal_periods=12,
).fit(optimized=True)
hw_future = hw_full.forecast(FUTURE_MONTHS)

# SARIMAX on full data
sarimax_full = SARIMAX(
    full_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False, enforce_invertibility=False,
).fit(disp=False)
sarimax_future = sarimax_full.forecast(FUTURE_MONTHS)

# Future dates
last_date = monthly_sales["Date"].max()
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=FUTURE_MONTHS, freq="MS")

# Forecast table
forecast_df = pd.DataFrame({
    "Month": future_dates.strftime("%B %Y"),
    "Holt-Winters": hw_future.values.round(2),
    "SARIMAX": sarimax_future.values.round(2),
})
forecast_df["Average Forecast"] = ((forecast_df["Holt-Winters"] + forecast_df["SARIMAX"]) / 2).round(2)

print("=" * 70)
print("          🔮 FUTURE SALES FORECAST — NEXT 6 MONTHS")
print("=" * 70)
print(forecast_df.to_string(index=False))
print("=" * 70)

# %%
# Plot forecast with confidence interval
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=monthly_sales["Date"], y=monthly_sales["Sales"],
    name="Historical Sales", line=dict(color="#4a90d9", width=2),
))

# Holt-Winters forecast
fig.add_trace(go.Scatter(
    x=future_dates, y=hw_future.values,
    name="Holt-Winters Forecast", line=dict(color="#50c878", width=2, dash="dash"),
    mode="lines+markers",
))

# SARIMAX forecast
fig.add_trace(go.Scatter(
    x=future_dates, y=sarimax_future.values,
    name="SARIMAX Forecast", line=dict(color="#e76f51", width=2, dash="dash"),
    mode="lines+markers",
))

# Confidence interval
forecast_ci = sarimax_full.get_forecast(FUTURE_MONTHS).conf_int()
fig.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(forecast_ci.iloc[:, 1]) + list(forecast_ci.iloc[:, 0][::-1]),
    fill="toself", fillcolor="rgba(231,111,81,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="95% Confidence Interval",
))

fig.add_vline(x=last_date, line_dash="dot", line_color="gray", annotation_text="Forecast →")

fig.update_layout(
    title="🔮 Sales Forecast — Next 6 Months",
    xaxis_title="", yaxis_title="Sales",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.15),
    height=500,
)
fig.show()

# %% [markdown]
# ## 15. Forecast by Product Family (Top 5)

# %%
# Get top 5 product families
top_families = df.groupby("family")["sales"].sum().nlargest(5).index.tolist()
print(f"Top 5 families: {top_families}")

family_forecasts = {}
colors = ["#4a90d9", "#50c878", "#e76f51", "#f4a261", "#9b59b6"]

for family in top_families:
    family_monthly = (
        df[df["family"] == family]
        .groupby(pd.Grouper(key="date", freq="MS"))["sales"]
        .sum()
    )
    
    try:
        model = ExponentialSmoothing(
            family_monthly, trend="add", seasonal="add", seasonal_periods=12,
        ).fit(optimized=True)
        forecast = model.forecast(FUTURE_MONTHS)
        family_forecasts[family] = forecast
        print(f"✅ {family}")
    except Exception as e:
        print(f"⚠️ {family}: {e}")

# Plot family forecasts
fig = go.Figure()
for i, (family, forecast) in enumerate(family_forecasts.items()):
    fig.add_trace(go.Bar(
        x=future_dates.strftime("%b %Y"), y=forecast.values,
        name=family, marker_color=colors[i % len(colors)],
    ))

fig.update_layout(
    title="🔮 Forecast by Product Family — Next 6 Months",
    barmode="group", template="plotly_white",
    xaxis_title="", yaxis_title="Forecasted Sales",
)
fig.show()

# Family forecast table
family_df = pd.DataFrame({"Month": future_dates.strftime("%B %Y")})
for family, forecast in family_forecasts.items():
    family_df[family] = forecast.values.round(2)

print("\n📦 Product Family Forecast:")
print(family_df.to_string(index=False))

# %% [markdown]
# ## 16. Impact of Oil Prices on Sales

# %%
# Merge oil prices with monthly sales
oil_clean = oil.dropna().copy()
oil_monthly = oil_clean.set_index("date").resample("MS")["dcoilwtico"].mean().reset_index()
oil_monthly.columns = ["Date", "Oil_Price"]

merged = monthly_sales.merge(oil_monthly, on="Date", how="inner")

# Correlation
correlation = merged["Sales"].corr(merged["Oil_Price"])
print(f"📊 Correlation between Sales and Oil Price: {correlation:.4f}")

# Dual axis plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Sales"], name="Sales", line=dict(color="#4a90d9")))
fig.add_trace(go.Scatter(x=merged["Date"], y=merged["Oil_Price"], name="Oil Price", yaxis="y2", line=dict(color="#e76f51")))

fig.update_layout(
    title=f"Sales vs Oil Price (Correlation: {correlation:.3f})",
    yaxis=dict(title="Sales", gridcolor="#eee"),
    yaxis2=dict(title="Oil Price (USD)", overlaying="y", side="right"),
    template="plotly_white",
    legend=dict(orientation="h", y=1.08),
)
fig.show()

# %% [markdown]
# ## 17. Holiday Impact Analysis

# %%
# Check if holidays affect sales
daily_with_holiday = daily_sales.merge(
    holidays[["date", "type", "description"]].drop_duplicates(),
    left_on="date", right_on="date", how="left"
)
daily_with_holiday["is_holiday"] = daily_with_holiday["type"].notna()

holiday_avg = daily_with_holiday[daily_with_holiday["is_holiday"]]["sales"].mean()
normal_avg = daily_with_holiday[~daily_with_holiday["is_holiday"]]["sales"].mean()
difference = ((holiday_avg - normal_avg) / normal_avg) * 100

print(f"📊 Average sales on holidays:     {holiday_avg:,.2f}")
print(f"📊 Average sales on normal days:  {normal_avg:,.2f}")
print(f"📊 Difference:                    {difference:+.1f}%")

# Bar chart
fig = px.bar(
    x=["Normal Days", "Holiday Days"],
    y=[normal_avg, holiday_avg],
    title="Average Daily Sales: Holidays vs Normal Days",
    labels={"x": "", "y": "Average Sales"},
    color=["Normal Days", "Holiday Days"],
    color_discrete_sequence=["#4a90d9", "#e76f51"],
)
fig.update_layout(template="plotly_white", showlegend=False)
fig.show()

# %% [markdown]
# ## 18. Key Findings & Summary

# %%
total_historical_avg = monthly_sales["Sales"].mean()
total_forecast_avg = forecast_df["Average Forecast"].mean()
growth = ((total_forecast_avg - total_historical_avg) / total_historical_avg) * 100

print("=" * 70)
print("          📊 STORE SALES FORECASTING — KEY FINDINGS")
print("=" * 70)

print(f"""
📈 Historical average monthly sales:  {total_historical_avg:,.2f}
🔮 Forecasted average monthly sales:  {total_forecast_avg:,.2f}
📊 Expected growth/decline:           {growth:+.1f}%

🏆 Best performing model:             {best_model}

🛢️ Oil price correlation:             {correlation:.3f}
🎉 Holiday sales impact:              {difference:+.1f}% vs normal days

💡 Recommendations:
   1. Plan inventory for seasonal peaks identified in decomposition
   2. Increase promotions during historically low months
   3. Monitor oil prices as potential leading indicator
   4. Leverage holiday periods for marketing campaigns
   5. Re-train models quarterly with fresh data
   6. Focus on top-performing product families

{"=" * 70}
""")

# %% [markdown]
# ## 19. Save Results

# %%
# Save forecast
output = forecast_df.copy()
for family, forecast in family_forecasts.items():
    output[family] = forecast.values.round(2)

output.to_csv("sales_forecast_results.csv", index=False)
print("✅ Forecast saved to 'sales_forecast_results.csv'")

# Save model comparison
results_sorted.to_csv("model_comparison.csv", index=False)
print("✅ Model comparison saved to 'model_comparison.csv'")

print("\n🎉 Project complete! You have a full sales forecasting pipeline.")
print("📊 Total charts generated: 14")
print("🤖 Models trained: 5")
print("🔮 Months forecasted: 6")
