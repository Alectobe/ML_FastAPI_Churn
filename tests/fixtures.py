import pandas as pd

SYNTHETIC_DATA = pd.DataFrame({
    "monthly_fee":        [10.0, 50.0, 30.0, 20.0, 60.0, 15.0, 45.0, 25.0, 55.0, 35.0],
    "usage_hours":        [5.0, 120.0, 60.0, 10.0, 150.0, 8.0, 90.0, 30.0, 110.0, 70.0],
    "support_requests":   [0, 5, 2, 1, 6, 0, 3, 1, 4, 2],
    "account_age_months": [1, 36, 12, 6, 48, 2, 24, 8, 30, 18],
    "failed_payments":    [0, 3, 1, 0, 4, 0, 2, 0, 3, 1],
    "autopay_enabled":    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    "region":             ["europe", "asia", "america", "africa", "europe",
                           "asia", "america", "africa", "europe", "asia"],
    "device_type":        ["mobile", "desktop", "tablet", "mobile", "desktop",
                           "tablet", "mobile", "desktop", "tablet", "mobile"],
    "payment_method":     ["card", "paypal", "crypto", "card", "paypal",
                           "crypto", "card", "paypal", "crypto", "card"],
    "churn":              [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
})

VALID_PREDICT_PAYLOAD = {
    "monthly_fee": 29.99,
    "usage_hours": 45.5,
    "support_requests": 2,
    "account_age_months": 12,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}