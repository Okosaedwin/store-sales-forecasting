# 📈 Store Sales Forecasting

> Time series forecasting on 3 million rows of retail data — comparing 5 statistical and machine learning models.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project forecasts monthly retail sales for **Favorita Grocery Stores** in Ecuador using 5 years of historical data (2013–2017). It compares **5 different forecasting models** — both classical statistical methods and modern machine learning — to determine which approach works best for this real-world dataset.

**Key Question:** Can we accurately predict the next 6 months of sales using past patterns?

## 📊 Dataset

- **Source:** [Kaggle Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Size:** 3,000,888 rows × 6 columns
- **Period:** January 2013 – August 2017 (56 months)
- **Stores:** 54 stores across Ecuador
- **Product Families:** 33 categories

### Files Used

| File | Description |
|------|-------------|
| `train.csv` | Daily sales by store and product family *(excluded — too large for GitHub)* |
| `stores.csv` | Store metadata (city, state, type, cluster) |
| `oil.csv` | Daily oil prices (Ecuador's economy is oil-dependent) |
| `holidays_events.csv` | National and local holidays |
| `transactions.csv` | Daily store transactions *(excluded — too large)* |

> **Note:** `train.csv` and `transactions.csv` are not committed to this repo due to GitHub size limits. Download them from the Kaggle link above.

## 🤖 Models Compared

| # | Model | Type | Description |
|---|-------|------|-------------|
| 1 | Moving Average | Statistical | Baseline — average of last 3 months |
| 2 | Holt-Winters | Statistical | Exponential smoothing with trend & seasonality |
| 3 | **SARIMAX** 🏆 | Statistical | Seasonal ARIMA with external regressors |
| 4 | Linear Regression | ML | Linear model on time-based features |
| 5 | Random Forest | ML | Ensemble model on time-based features |

## 🏆 Results

### Model Performance (Test Set: Last 6 Months)

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| 🏆 **SARIMAX** | **2,980,597** | **5,804,754** | **-0.225** |
| Moving Average | 3,027,587 | 5,819,817 | -0.232 |
| Holt-Winters | 3,095,654 | 5,953,445 | -0.289 |
| Random Forest | 11,675,769 | 12,746,814 | -4.908 |
| Linear Regression | 13,339,315 | 14,519,864 | -6.666 |

### 🔍 Key Business Insights

- **🛢️ Oil Price Correlation: -0.75**
  Strong negative correlation between oil prices and retail sales — Ecuador's oil-dependent economy means lower oil prices reduce consumer spending.

- **🎉 Holiday Sales Boost: +11.8%**
  Sales on holidays are 11.8% higher than on normal days, validating holiday marketing investment.

- **📅 December = Peak Sales**
  Forecasted as the highest-sales month due to seasonal/holiday effect.

### 💡 The Biggest Lesson

**ML isn't always better than statistical methods.** SARIMAX and Holt-Winters significantly outperformed Random Forest and Linear Regression on this dataset. Choose the right tool for the data, not the trendiest one.

## 📁 Repository Structure

```
store-sales-forecasting/
├── store_sales_forecasting.ipynb    # Main Jupyter notebook
├── sales_forecasting.py             # Python script version
├── stores.csv                       # Store metadata
├── oil.csv                          # Oil price history
├── holidays_events.csv              # Holidays dataset
├── sales_forecast_results.csv       # Output: 6-month forecast
├── model_comparison.csv             # Output: model performance metrics
├── .gitignore
└── README.md
```

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas & NumPy** — data manipulation
- **Statsmodels** — SARIMAX, Holt-Winters, time series decomposition
- **Scikit-learn** — Linear Regression, Random Forest, evaluation metrics
- **Plotly & Matplotlib** — interactive and static visualizations
- **Jupyter Notebook** — interactive development

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/Okosaedwin/store-sales-forecasting.git
cd store-sales-forecasting
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib plotly scikit-learn statsmodels jupyter
```

### 3. Download missing CSVs

Get `train.csv` and `transactions.csv` from the [Kaggle dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and place them in the repo root.

### 4. Open the notebook

```bash
jupyter notebook store_sales_forecasting.ipynb
```

Run cells top to bottom to reproduce all results.

## 📈 Pipeline Steps

1. **Load 5 datasets** (train, stores, oil, holidays, transactions)
2. **Merge & explore** the data
3. **Aggregate** daily sales to monthly totals
4. **Decompose** time series into trend + seasonality + residual
5. **Train-test split** (50 months train, 6 months test)
6. **Train 5 models** and evaluate each
7. **Compare** all models using MAE, RMSE, R²
8. **Forecast** the next 6 months with confidence intervals
9. **Forecast by category** for top 5 product families
10. **Analyze external factors** (oil prices, holidays)

## 🚀 Roadmap

- [ ] Add Prophet model (Facebook's forecasting library)
- [ ] Try LSTM neural network for comparison
- [ ] Include external regressors in SARIMAX (oil price, holidays)
- [ ] Per-store forecasting (54 individual models)
- [ ] Hyperparameter tuning with grid search
- [ ] Deploy as a Streamlit web app

## 👤 Author

**Edwin Okosa**
[LinkedIn](https://www.linkedin.com/in/edwin-okosa/) · [GitHub](https://github.com/Okosaedwin)

## 📄 License

This project is licensed under the MIT License — feel free to use it for learning or your own work.

---

⭐ If you found this project helpful, give it a star!
