from beta_geo_seasonal import BetaGeoModelWithSeasonality
from summary_functions import summary_data_from_transaction_data_season 
import pandas as pd

data = {
    'customer_id': [1, 1, 1, 2, 2, 3],
    'transaction_date': [
        '2023-01-01', '2023-02-15', '2023-03-20',
        '2023-01-05', '2023-04-10', '2023-02-20'
    ],
    'monetary_value': [100, 150, 200, 50, 75, 120],
    'high_season': [0, 1, 0, 0, 1, 0]
}
# Create DataFrame
transactions = pd.DataFrame(data)

# Convert 'transaction_date' to datetime
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

summary = summary_data_from_transaction_data_season(
    transactions,
    customer_id_col='customer_id',
    datetime_col='transaction_date',
    monetary_value_col='monetary_value',
    high_season_col='high_season',
    datetime_format='%Y-%m-%d',
    observation_period_end='2023-12-31',
    freq='D',
    include_first_transaction=True
)

# Rename the high_season_sum column back to high_season
summary.rename(columns={'high_season_tx_sum': 'high_season'}, inplace=True)

model = BetaGeoModelWithSeasonality(
    data=summary,
    model_config={
        "r_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
        "alpha_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
        "a_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
        "b_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
        "phi_prior": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}},
    },
    sampler_config={
        "draws": 1000,
        "tune": 1000,
        "chains": 2,
        "cores": 2,
    },
)
model.fit()
print(model.fit_summary())

expected_purchases = model.expected_purchases(future_t=10)
probability_alive = model.expected_probability_alive()
expected_purchases_new_customer = model.expected_purchases_new_customer(t=10)