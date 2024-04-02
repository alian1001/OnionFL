import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set a seed for replicability
seed_value = 42
np.random.seed(seed_value)

# Parameters
num_households = 10  # Number of households (nodes)
num_days = 30  # Number of days to simulate
time_slots_per_day = 24  # Number of time slots per day (e.g., hourly data)
test_size = 0.2  # Proportion of data to be used as test set

# Generate synthetic data for each household and split into train and test sets
for household_id in range(num_households):
    household_data = []
    for day in range(num_days):
        # Simulate a daily usage pattern with a peak at a random time slot
        peak_time = np.random.choice(range(time_slots_per_day))
        usage_pattern = np.random.normal(loc=0.5, scale=0.2, size=time_slots_per_day)
        usage_pattern[peak_time] += np.random.normal(loc=1.0, scale=0.3)

        # Normalize to ensure non-negative values
        usage_pattern = np.clip(usage_pattern, 0, None)

        # Append data for each time slot
        for time_slot in range(time_slots_per_day):
            household_data.append({
                'household_id': household_id,
                'day': day,
                'time_slot': time_slot,
                'electricity_usage': usage_pattern[time_slot],
                'is_peak_usage': int(time_slot == peak_time)
            })

    # Create a DataFrame for the household
    household_df = pd.DataFrame(household_data)

    # Split into train and test sets
    train_df, test_df = train_test_split(household_df, test_size=test_size, random_state=seed_value)

    # Save to CSV
    train_df.to_csv(f'synthetic_household_{household_id}_train.csv', index=False)
    test_df.to_csv(f'synthetic_household_{household_id}_test.csv', index=False)

print(f"Generated synthetic train and test data for {num_households} households with seed {seed_value}.")
