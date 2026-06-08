# UK House Price Dynamics Analysis and Growth Forecasting

## Overview

This project investigates the dynamics of UK house price growth using the UK House Price Index (HPI) and macroeconomic indicators.

The analysis combines exploratory data analysis, feature engineering, and machine learning to model annual house price growth across UK local authorities from 2004–2025.

The project focuses on understanding:

* Regional differences in house price growth
* The persistence of housing market trends
* The relationship between transaction volumes, inflation, and future price growth
* The predictive value of historical housing market indicators

---

## Dataset

### UK House Price Index (HPI)

Source: HM Land Registry

The dataset contains monthly observations for over 400 UK local authorities, including:

* Average house prices
* House price indices
* Monthly and annual growth rates
* Transaction volumes

### Additional Data

#### Regional Mapping

Local authorities were mapped to broader UK regions to enable regional analysis and modelling.

#### Inflation (CPIH)

Consumer Prices Index including owner occupiers' housing costs (CPIH) data from the Office for National Statistics (ONS) was merged with the HPI dataset.

---

## Exploratory Data Analysis

The EDA investigates:

* Missing data patterns and structural missingness
* Long-term regional house price trends
* Growth dynamics before and after the 2008 financial crisis
* Regional growth volatility
* Convergence and divergence of regional housing markets
* Inflation-adjusted market context

Key findings include:

* Northern Ireland exhibited substantially higher growth volatility than other UK regions between 2006 and 2014.
* Regional growth patterns became more similar after 2015.
* Growth slowed across many southern regions after 2022, while several Midlands and northern regions continued to show moderate growth.
* Historical growth rates demonstrate strong persistence over time.

---

## Feature Engineering

The modelling dataset was constructed using lagged and rolling-window features.

### Growth Features

* 1-month growth
* Lagged 12-month growth (1, 3, and 12 months)
* 3-month rolling average of annual growth
* 6-month rolling volatility of annual growth

### Market Activity Features

* Sales volume
* Lagged sales volume
* Six-month sales volume growth

### Trend Features

* Normalised house price index
* 12-month rolling average of the index

### Macroeconomic Features

* CPIH inflation index

### Regional Features

* UK region encoded as categorical variables

---

## Modelling

### Target Variable

The target is annual house price growth (`12m%Change`).

Predicting annual growth rather than price levels helps reduce seasonality and focuses the model on medium-term market dynamics.

### Models

The following models were implemented:

1. Mean baseline
2. Linear Regression
3. Random Forest Regressor

### Validation

A time-based train-test split was used, with the most recent 20% of observations reserved for testing.

This preserves temporal ordering and avoids look-ahead bias.

---

## Results

| Model             | MAE  | RMSE |
| ----------------- | ---- | ---- |
| Mean Baseline     | 4.22 | 5.25 |
| Linear Regression | 0.99 | 1.37 |
| Random Forest     | 1.04 | 1.45 |

### Feature Importance

The Random Forest model identified lagged annual growth as the strongest predictor of future annual growth.

This suggests that UK house price growth exhibits strong momentum and persistence, with historical growth rates containing substantial predictive information.

Regional and macroeconomic variables contributed additional predictive power but were secondary to temporal dynamics.

---

## Future Improvements

Potential extensions include:

* Walk-forward backtesting
* Additional macroeconomic variables (interest rates, unemployment, earnings growth)
* Gradient boosting models (XGBoost, LightGBM)
* Time-series specific forecasting approaches (SARIMAX, Prophet)
* Prediction intervals and uncertainty estimates
* Region-specific forecasting models

---

## Project Structure

```text
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── pipelines/
├── main.py
├── README.md
└── requirements.txt
```
