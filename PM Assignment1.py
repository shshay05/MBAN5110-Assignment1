#!/usr/bin/env python
# coding: utf-8

# In[4]:


# =====================================================
# MBAN 5110  ASSIGNMENT 1  
# QUESTION 1 | STEP 1: BIWEEKLY DEMAND FORECASTING - DATA PREPARATION
# =====================================================

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


df_transactions = pd.read_csv("sales_transactions.csv")
df_transactions.head()


# In[5]:


# Convert dates
df_transactions["order_date"] = pd.to_datetime(
    df_transactions["order_date"]
)

df_transactions["delivery_date"] = pd.to_datetime(
    df_transactions["delivery_date"]
)

# Aggregate total demand across all bike models into 14-day periods
start = df_transactions["delivery_date"].min()
end = df_transactions["delivery_date"].max()

demand = (
    df_transactions.set_index("delivery_date")
      .resample("14D", origin=start)["quantity"]
      .sum()
      .to_frame("demand")
)

# Exclude incomplete final period
demand = demand[
    demand.index + pd.Timedelta(days=13) <= end
].copy()

# Time index for OLS forecasting
demand["t"] = np.arange(len(demand))

# Data validation
print("Transactions:", len(df_transactions))
print("Original total bikes:", df_transactions["quantity"].sum())
print("Complete biweekly periods:", len(demand))
print("Total demand in complete periods:", demand["demand"].sum())
print("Average biweekly demand:", round(demand["demand"].mean(), 2))

display(demand.head())


# In[6]:


# QUESTION 1 | STEP 2: FOUR SEASONAL OLS MODELS

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

season = 365.25 / 14
split = len(demand) // 2

models, forecasts, results = {}, {}, []

for h in range(1, 5):
    data = demand.copy()

    data["target"] = data["demand"].shift(-h)
    data["sin"] = np.sin(2 * np.pi * (data["t"] + h) / season)
    data["cos"] = np.cos(2 * np.pi * (data["t"] + h) / season)

    features = ["demand", "t", "sin", "cos"]

    train = data.iloc[:split-h].dropna()
    test = data.iloc[split:-4].copy()

    model = sm.OLS(
        train["target"],
        sm.add_constant(train[features], has_constant="add")
    ).fit()

    prediction = np.maximum(
        model.predict(
            sm.add_constant(test[features], has_constant="add")
        ),
        0
    )

    models[h] = model

    forecasts[h] = pd.DataFrame({
        "Actual": test["target"],
        "Forecast": prediction,
        "Naive": test["demand"]
    })

    results.append({
        "Days Ahead": 14 * h,
        "MAE": mean_absolute_error(test["target"], prediction),
        "RMSE": np.sqrt(
            mean_squared_error(test["target"], prediction)
        ),
        "Test R2": r2_score(test["target"], prediction),
        "Naive MAE": mean_absolute_error(
            test["target"], test["demand"]
        )
    })

performance = pd.DataFrame(results)

print("QUESTION 1 | SEASONAL OLS FORECAST PERFORMANCE")
display(performance.round(3))


# In[7]:


# QUESTION 1 | STEP 3: FORECASTING DASHBOARD

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Bicycle Demand Forecasting | Seasonal OLS",
             fontsize=17, fontweight="bold")

# 1. Historical biweekly demand
ax[0, 0].plot(demand.index, demand["demand"],
              color="#2563EB", linewidth=1.7)
ax[0, 0].axhline(demand["demand"].mean(),
                 color="#DC2626", linestyle="--",
                 label="Average demand")
ax[0, 0].set_title("Historical Biweekly Demand")
ax[0, 0].set_ylabel("Bicycles")
ax[0, 0].legend()

# 2. Actual vs predicted (56-day horizon)
f = forecasts[4]
dates = f.index + pd.Timedelta(days=56)

ax[0, 1].plot(dates, f["Actual"],
              label="Actual", color="#111827")
ax[0, 1].plot(dates, f["Forecast"],
              label="Seasonal OLS", color="#059669",
              linestyle="--")
ax[0, 1].set_title("56-Day Forecast: Actual vs Predicted")
ax[0, 1].set_ylabel("Bicycles")
ax[0, 1].legend()

# 3. MAE comparison
x = np.arange(4)
ax[1, 0].bar(x - 0.18, performance["MAE"], 0.36,
             color="#2563EB", label="Seasonal OLS")
ax[1, 0].bar(x + 0.18, performance["Naive MAE"], 0.36,
             color="#94A3B8", label="Naive")
ax[1, 0].set_xticks(x, performance["Days Ahead"])
ax[1, 0].set_title("Forecast MAE: Lower Is Better")
ax[1, 0].set_xlabel("Forecast Horizon (Days)")
ax[1, 0].set_ylabel("MAE (Bicycles)")
ax[1, 0].legend()

# 4. Out-of-sample R²
bars = ax[1, 1].bar(
    performance["Days Ahead"].astype(str),
    performance["Test R2"], color="#0F766E"
)
ax[1, 1].bar_label(bars, fmt="%.3f", padding=3)
ax[1, 1].set_ylim(0, 1)
ax[1, 1].set_title("Out-of-Sample R²")
ax[1, 1].set_xlabel("Forecast Horizon (Days)")
ax[1, 1].set_ylabel("R²")

for a in ax.flat:
    a.grid(alpha=0.15)
    a.set_axisbelow(True)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig("Q1_Forecasting_Dashboard.png",
            dpi=300, bbox_inches="tight")
plt.show()

#1:Biweekly Demand Forecasting

Sales transactions across three bicycle models were aggregated into **129 complete 14-day periods**, with an average demand of **15.80 bicycles per period**.

Four seasonal Ordinary Least Squares (OLS) models were developed to forecast demand **14, 28, 42, and 56 days ahead**, incorporating current demand, a time trend, and annual sine/cosine seasonality. A chronological 50:50 training/test split was used.

The models achieved test MAEs of **7.634, 6.826, 8.127, and 8.367 bicycles**, respectively. The 28-day model achieved the highest test R² (**0.670**). Compared with naïve forecasting, seasonal OLS reduced MAE by approximately **10.0% at 42 days** and **22.0% at 56 days**, but underperformed at shorter horizons.

**Managerial Implication:** Seasonal OLS is particularly useful for medium-term capacity planning, especially at the 42- and 56-day horizons. However, forecast uncertainty should be incorporated into inventory and production decisions to reduce the risk of shortages or excess capacity.
# In[8]:


# QUESTION 2 | MULTINOMIAL LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

price_cols = [
    "Price_Comfort_Mountain",
    "Price_Tough_Track",
    "Price_Ultimate_Mountain"
]

# Prepare features and target
X = df_transactions[price_cols]
y = df_transactions["Product"]

# Fit multinomial logistic regression
choice_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, solver="lbfgs")
)
choice_model.fit(X, y)

# Predict probabilities at assignment prices
scenario = pd.DataFrame(
    [[4200, 1950, 500]], columns=price_cols
)

probabilities = choice_model.predict_proba(scenario)[0]
products = choice_model.named_steps["logisticregression"].classes_

choice_results = pd.DataFrame({
    "Bicycle Model": products,
    "Probability (%)": probabilities * 100
}).sort_values("Probability (%)", ascending=False)

print("CUSTOMER CHOICE PROBABILITIES")
display(choice_results.round(2))
print("Total probability:", round(probabilities.sum(), 6))



# In[9]:


# QUESTION 2 | STEP 2: CUSTOMER CHOICE VISUALIZATION

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))

colors = ["#059669", "#2563EB", "#94A3B8"]

bars = ax.barh(
    choice_results["Bicycle Model"],
    choice_results["Probability (%)"],
    color=colors
)

ax.invert_yaxis()

ax.bar_label(bars, fmt="%.2f%%", padding=5)

ax.set_xlim(0, 65)
ax.set_xlabel("Predicted Choice Probability (%)")
ax.set_title(
    "Customer Bicycle Choice Probabilities | Multinomial Logistic Regression",
    fontsize=13,
    fontweight="bold"
)

ax.grid(axis="x", alpha=0.15)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    "Q2_Choice_Probabilities.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# In[10]:


# QUESTION 2 | QUANTITY-WEIGHTED ROBUSTNESS CHECK

weighted_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, solver="lbfgs")
)

weighted_model.fit(
    df_transactions[price_cols],
    df_transactions["Product"],
    logisticregression__sample_weight=df_transactions["quantity"]
)

weighted_prob = weighted_model.predict_proba(scenario)[0] * 100

comparison = pd.DataFrame({
    "Bicycle Model": products,
    "Unweighted (%)": probabilities * 100,
    "Quantity-Weighted (%)": weighted_prob
})

display(comparison.round(2))


#2: Multinomial Logistic Regression

A multinomial logistic regression model was developed to estimate customer choice probabilities using the prices of all three bicycle models as explanatory variables.

At the specified prices of **$4,200 (Comfort Mountain), $1,950 (Tough Track), and $500 (Ultimate Mountain)**, the predicted choice probabilities were **19.66%, 29.88%, and 50.46%**, respectively.

A quantity-weighted robustness check yielded corresponding probabilities of **20.70%, 29.18%, and 50.12%**, indicating that the predicted product mix remains relatively stable across both specifications.

**Managerial Implication:** Ultimate Mountain represents approximately half of the predicted product mix, followed by Tough Track and Comfort Mountain. These estimates can inform product-mix and inventory planning. However, the probabilities are conditional on purchasing one of the three bicycle models and do not measure total market demand.
# In[15]:


# =====================================================
# QUESTION 3 | STEP 1: DATA PREPARATION AND 50:50 SPLIT
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
df_transactions = pd.read_csv("sales_transactions.csv")
df = df_transactions.copy()
df["order_date"] = pd.to_datetime(df["order_date"])
df["delivery_date"] = pd.to_datetime(df["delivery_date"])

# Aggregate demand into complete 14-day periods
start = df["delivery_date"].min().normalize()
end = df["delivery_date"].max().normalize()

demand = (
    df.set_index("delivery_date")
      .resample("14D", origin=start)["quantity"]
      .sum().to_frame("demand")
)
demand = demand[demand.index + pd.Timedelta(days=13) <= end].copy()
demand["t"] = np.arange(len(demand))

# Chronological 50:50 split
split = len(demand) // 2
train_demand = demand.iloc[:split].copy()
test_demand = demand.iloc[split:].copy()
cutoff = test_demand.index.min()

# Align transaction-level data
final_end = demand.index.max() + pd.Timedelta(days=14)
train_tx = df[df["delivery_date"] < cutoff].copy()
test_tx = df[
    (df["delivery_date"] >= cutoff) &
    (df["delivery_date"] < final_end)
].copy()

# Validate split and quantities
assert train_tx["quantity"].sum() == train_demand["demand"].sum()
assert test_tx["quantity"].sum() == test_demand["demand"].sum()

print("Total transactions:", len(df))
print("Complete periods:", len(demand))
print("Training periods:", len(train_demand))
print("Testing periods:", len(test_demand))
print("Training transactions:", len(train_tx))
print("Testing transactions:", len(test_tx))
print("Test starts:", cutoff.date())

# Visualize 50:50 split
plt.figure(figsize=(11, 4))
plt.plot(train_demand.index, train_demand["demand"], label="Training")
plt.plot(test_demand.index, test_demand["demand"], label="Testing")
plt.axvline(cutoff, color="red", linestyle="--", label="50:50 Split")
plt.title("Biweekly Demand | Chronological 50:50 Split")
plt.xlabel("Delivery Period")
plt.ylabel("Bicycles per 14 Days")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()



# In[16]:


# =====================================================
# QUESTION 3 | STEP 2: OLS DEMAND FORECASTING
# =====================================================

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

season = 365.25 / 14
q3_models, q3_predictions, results = {}, {}, []

# Train 14-day and 56-day OLS models
for h in [1, 4]:
    data = demand.copy()
    data["target"] = data["demand"].shift(-h)
    data["sin"] = np.sin(2*np.pi*(data["t"]+h)/season)
    data["cos"] = np.cos(2*np.pi*(data["t"]+h)/season)

    features = ["demand", "t", "sin", "cos"]
    train = data.iloc[:split-h].dropna()
    test = data.iloc[split-h:len(data)-h]

    model = sm.OLS(
        train["target"],
        sm.add_constant(train[features], has_constant="add")
    ).fit()

    pred = np.maximum(model.predict(
        sm.add_constant(test[features], has_constant="add")
    ).to_numpy(), 0)

    actual = test_demand["demand"].to_numpy()

    q3_models[h] = model
    q3_predictions[h] = pd.DataFrame(
        {"Actual": actual, "Predicted": pred},
        index=test_demand.index
    )

    results.append({
        "Horizon": f"{h*14} days",
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred)
    })

# Forecast accuracy comparison
performance = pd.DataFrame(results)
display(performance.round(3))

# Visualize actual vs predicted demand
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

for ax, h in zip(axes, [1, 4]):
    f = q3_predictions[h]
    ax.plot(f.index, f["Actual"], label="Actual")
    ax.plot(f.index, f["Predicted"], "--", label="OLS Forecast")
    ax.set_title(f"{h*14}-Day Demand Forecast")
    ax.set_ylabel("Bicycles")
    ax.legend()
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()



# In[17]:


# =====================================================
# QUESTION 3 | STEP 3: MULTINOMIAL LOGISTIC REGRESSION
# =====================================================

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Define bicycle models and price variables
products = ["Comfort Mountain", "Tough Track", "Ultimate Mountain"]

price_cols = [
    "Price_Comfort_Mountain",
    "Price_Tough_Track",
    "Price_Ultimate_Mountain"
]

# Train multinomial logistic regression on training data only
choice_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, solver="lbfgs")
)

choice_model.fit(
    train_tx[price_cols],
    train_tx["Product"],
    logisticregression__sample_weight=train_tx["quantity"]
)

# Predict choice probabilities at assignment prices
reference_prices = np.array([4200, 1950, 500])

scenario = pd.DataFrame(
    [reference_prices],
    columns=price_cols
)

probabilities = choice_model.predict_proba(scenario)[0]

choice_results = pd.DataFrame({
    "Product": choice_model.classes_,
    "Price ($)": [
        dict(zip(products, reference_prices))[p]
        for p in choice_model.classes_
    ],
    "Choice Probability (%)": probabilities * 100
})

print("QUESTION 3 | CUSTOMER CHOICE PROBABILITIES")
display(choice_results.round(2))

# Visualize choice probabilities
plt.figure(figsize=(8, 4))
plt.bar(
    choice_results["Product"],
    choice_results["Choice Probability (%)"]
)
plt.title("Training-Based Bicycle Choice Probabilities")
plt.ylabel("Choice Probability (%)")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()

# Validation
assert np.isclose(probabilities.sum(), 1)
print("Validation passed: Probabilities sum to 100%.")



# In[18]:


# =====================================================
# QUESTION 3 | STEP 4: SHARED CAPACITY RESERVATION
# =====================================================

from scipy.stats import norm

# Training residuals from the 56-day OLS model
h = 4
data = demand.copy()
data["target"] = data["demand"].shift(-h)
data["sin"] = np.sin(2*np.pi*(data["t"]+h)/season)
data["cos"] = np.cos(2*np.pi*(data["t"]+h)/season)

features = ["demand", "t", "sin", "cos"]
train = data.iloc[:split-h].dropna()

residuals = q3_models[4].resid
sigma = residuals.std(ddof=1)

# Capacity threshold: 90% service level
service_level = 0.90
z = norm.ppf(service_level)

# Reserve capacity 56 days ahead
q3_capacity = q3_predictions[4].copy()
q3_capacity["Reserved Capacity"] = np.ceil(
    np.maximum(0, q3_capacity["Predicted"] + z*sigma)
).astype(int)

q3_capacity["Shortage"] = np.maximum(
    0, q3_capacity["Actual"] - q3_capacity["Reserved Capacity"]
)

q3_capacity["Unused Capacity"] = np.maximum(
    0, q3_capacity["Reserved Capacity"] - q3_capacity["Actual"]
)

# Summary
print("QUESTION 3 | CAPACITY RESERVATION")
print("Service level:", f"{service_level:.0%}")
print("Residual standard deviation:", round(sigma, 2))
print("Total reserved capacity:", q3_capacity["Reserved Capacity"].sum())
print("Actual test demand:", q3_capacity["Actual"].sum())
print("Potential shortage:", q3_capacity["Shortage"].sum())
print("Unused capacity:", q3_capacity["Unused Capacity"].sum())
print("Periods with sufficient capacity:",
      f"{(q3_capacity['Shortage'] == 0).mean():.1%}")

display(q3_capacity.head().round(2))

# Visualization
plt.figure(figsize=(11, 4))
plt.plot(q3_capacity.index, q3_capacity["Actual"], label="Actual Demand")
plt.plot(q3_capacity.index, q3_capacity["Reserved Capacity"],
         "--", label="Reserved Capacity")
plt.title("56-Day Capacity Reservation vs Actual Demand")
plt.xlabel("Selling Period")
plt.ylabel("Bicycles")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


# In[19]:


# =====================================================
# QUESTION 3 | STEP 5: PRICE AND QUANTITY OPTIMIZATION
# =====================================================

from scipy.optimize import minimize

# Product order follows the logistic model classes
products = list(choice_model.classes_)
price_map = dict(zip(
    ["Comfort Mountain", "Tough Track", "Ultimate Mountain"],
    [4200, 1950, 500]
))
reference = np.array([price_map[p] for p in products])

# Align prices with logistic model feature order
bounds = [
    (train_tx[col].min(), train_tx[col].max())
    for col in price_cols
]

def predict_shares(prices):
    scenario = pd.DataFrame(
        [prices], columns=price_cols
    )
    return choice_model.predict_proba(scenario)[0]

# Optimize expected revenue per purchasing customer
def objective(prices):
    shares = predict_shares(prices)
    prices_by_product = dict(zip(
        ["Comfort Mountain", "Tough Track", "Ultimate Mountain"],
        prices
    ))
    aligned = np.array([prices_by_product[p] for p in products])
    return -np.dot(aligned, shares)

result = minimize(
    objective,
    [4200, 1950, 500],
    bounds=bounds,
    method="L-BFGS-B"
)

assert result.success, result.message

best_prices = result.x
best_shares = predict_shares(best_prices)

# Production forecast constrained by reserved capacity
q3_plan = q3_capacity.copy()
q3_plan["Order Forecast"] = q3_predictions[1]["Predicted"]

q3_plan["Total Ordered"] = np.minimum(
    np.rint(q3_plan["Order Forecast"]).astype(int),
    q3_plan["Reserved Capacity"]
)

# Allocate integer quantities using largest remainders
def allocate(total, shares):
    raw = total * shares
    qty = np.floor(raw).astype(int)
    remainder = int(total - qty.sum())
    qty[np.argsort(-(raw - qty))[:remainder]] += 1
    return qty

quantities = np.array([
    allocate(total, best_shares)
    for total in q3_plan["Total Ordered"]
])

price_by_product = dict(zip(
    ["Comfort Mountain", "Tough Track", "Ultimate Mountain"],
    best_prices
))

for j, product in enumerate(products):
    q3_plan[f"Qty_{product}"] = quantities[:, j]
    q3_plan[f"Price_{product}"] = price_by_product[product]

aligned_prices = np.array([price_by_product[p] for p in products])

q3_plan["Planned Revenue"] = (
    quantities * aligned_prices
).sum(axis=1)

# Validate shared capacity constraint
assert (
    q3_plan["Total Ordered"] <= q3_plan["Reserved Capacity"]
).all()

assert (
    quantities.sum(axis=1) == q3_plan["Total Ordered"]
).all()

# Display optimal prices and quantities
summary = pd.DataFrame({
    "Product": products,
    "Optimal Price ($)": aligned_prices,
    "Choice Probability (%)": best_shares * 100,
    "Total Ordered": quantities.sum(axis=0)
})

print("QUESTION 3 | PRICE AND QUANTITY OPTIMIZATION")
display(summary.round(2))

print("Total reserved capacity:",
      q3_plan["Reserved Capacity"].sum())
print("Total ordered:", q3_plan["Total Ordered"].sum())
print("Planned production value: $",
      round(q3_plan["Planned Revenue"].sum(), 2))

display(q3_plan.head().round(2))


This optimization maximizes expected revenue per purchasing customer, conditional on purchase. It does not optimize total market demand or profit because production costs, reservation costs, and price-dependent purchase incidence are unavailable.
Also, the optimal prices may remain constant across test periods because the same fitted choice model and price bounds are used throughout. The best prices array follows the order of price cols, whereas the predicted probabilities follow choice_model.classes_. The code aligns prices and probabilities when calculating revenue and reporting results.
# In[20]:


# =====================================================
# QUESTION 3 | STEP 6: OUT-OF-SAMPLE EVALUATION
# =====================================================

# Evaluate planned production against historical demand
q3_eval = q3_plan.copy()

q3_eval["Potential Sales"] = np.minimum(
    q3_eval["Total Ordered"], q3_eval["Actual"]
)

q3_eval["Unmet Demand"] = np.maximum(
    0, q3_eval["Actual"] - q3_eval["Total Ordered"]
)

q3_eval["Excess Production"] = np.maximum(
    0, q3_eval["Total Ordered"] - q3_eval["Actual"]
)

# Performance summary
q3_summary = pd.DataFrame({
    "Metric": [
        "Test Periods",
        "Reserved Capacity",
        "Planned Production",
        "Actual Historical Demand",
        "Potential Sales",
        "Unmet Demand",
        "Excess Production",
        "Capacity Utilization (%)",
        "Planned Production Value ($)"
    ],
    "Value": [
        len(q3_eval),
        q3_eval["Reserved Capacity"].sum(),
        q3_eval["Total Ordered"].sum(),
        q3_eval["Actual"].sum(),
        q3_eval["Potential Sales"].sum(),
        q3_eval["Unmet Demand"].sum(),
        q3_eval["Excess Production"].sum(),
        100 * q3_eval["Total Ordered"].sum()
        / q3_eval["Reserved Capacity"].sum(),
        q3_eval["Planned Revenue"].sum()
    ]
})

print("QUESTION 3 | FINAL TEST PERFORMANCE")
display(q3_summary.round(2))

# Plot production against historical demand
plt.figure(figsize=(11, 4))
plt.plot(q3_eval.index, q3_eval["Actual"],
         label="Actual Demand")
plt.plot(q3_eval.index, q3_eval["Total Ordered"],
         "--", label="Planned Production")
plt.plot(q3_eval.index, q3_eval["Reserved Capacity"],
         ":", label="Reserved Capacity")

plt.title("Question 3 | Capacity, Production and Demand")
plt.xlabel("Selling Period")
plt.ylabel("Bicycles per 14 Days")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Verify production constraints
assert (q3_eval["Total Ordered"] <=
        q3_eval["Reserved Capacity"]).all()

# Export complete test-period decisions
q3_eval.to_csv(
    "Q3_Complete_Test_Period_Decisions.csv",
    index_label="Selling Period"
)

print("Validation passed: Capacity constraints satisfied.")



 #3: Capacity, Pricing, and Production Optimization

A chronological 50:50 training-test split was applied to 129 complete biweekly periods. Seasonal OLS models forecast demand 56 days ahead for shared capacity reservation and 14 days ahead for production planning. A training-based multinomial logistic regression estimated customer choice probabilities under alternative prices.

Across 65 test periods, the model reserved 2,034 bicycle units and planned production of 1,586 units, achieving 77.97% capacity utilization. Optimized prices were $4,516.20 for Comfort Mountain, $2,277 for Tough Track, and $778 for Ultimate Mountain. The resulting planned production value was approximately $6.88 million.

The framework satisfied the shared-capacity constraint and captured broad seasonal demand patterns. However, forecast errors created potential shortages and excess inventory. Furthermore, the optimized product mix was heavily concentrated in Comfort Mountain, highlighting limitations in the price-response model. Because total market demand and operating costs were not modeled, these decisions represent a revenue-oriented planning scenario rather than a proven profit-optimal solution.
# In[22]:


# =====================================================
# QUESTION 3 | STEP 7: FINAL VALIDATION AND EXPORT
# =====================================================

# Validate shared capacity and product quantities
assert (q3_plan["Total Ordered"] <=
        q3_plan["Reserved Capacity"]).all()

assert (q3_plan[[f"Qty_{p}" for p in products]].sum(axis=1)
        == q3_plan["Total Ordered"]).all()

assert len(q3_plan) == len(test_demand)

# Export complete test-period decisions and summary
q3_eval.to_csv(
    "Q3_Complete_65_Period_Decision_Plan.csv",
    index_label="Selling Date"
)

q3_summary.to_csv(
    "Q3_Final_Performance_Summary.csv",
    index=False
)

print("FINAL VALIDATION PASSED")
print("All 65 test periods evaluated.")
print("Capacity and quantity constraints satisfied.")
print("Decision plan and summary exported successfully.")



### Conclusion

The analysis demonstrates how seasonal OLS forecasting and multinomial logistic regression can be integrated to support advance capacity reservation, pricing, and production decisions. The proposed framework respects shared-capacity constraints and captures seasonal demand patterns, although forecasting errors create potential shortages and excess inventory. While the approach provides useful managerial insights, limitations in price-dependent demand and cost information mean the resulting decisions should be interpreted as illustrative rather than fully profit-optimal.
# In[ ]:


get_ipython().system('jupyter nbconvert --to python *.ipynb')


# In[ ]:




