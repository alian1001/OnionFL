import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_households = 10
num_days = 365
time_slots_per_day = 24

# Generate synthetic data
data = []
for household_id in range(num_households):
    for day in range(num_days):
        # Simulate a daily usage pattern with a peak at a random time slot
        peak_time = np.random.choice(range(time_slots_per_day))
        usage_pattern = np.random.normal(loc=0.5, scale=0.2, size=time_slots_per_day)
        usage_pattern[peak_time] += np.random.normal(loc=1.0, scale=0.3)

        # Normalize to ensure non-negative values
        usage_pattern = np.clip(usage_pattern, 0, None)

        # Append data for each time slot
        for time_slot in range(time_slots_per_day):
            data.append({
                'household_id': household_id,
                'day': day,
                'time_slot': time_slot,
                'electricity_usage': usage_pattern[time_slot],
                'is_peak_usage': int(time_slot == peak_time)
            })

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('datasets/synthetic_electricity_usage.csv', index=False)

# Display the first few rows
print(df.head())


# Select data for a specific household and day
household_id = 0
day = 0
household_data = df[(df['household_id'] == household_id) & (df['day'] == day)]

# Plot electricity consumption over time
plt.figure(figsize=(12, 6))
plt.plot(household_data['time_slot'], household_data['electricity_usage'], marker='o')
plt.title(f'Electricity Consumption for Household {household_id} on Day {day}')
plt.xlabel('Time Slot (Hour)')
plt.ylabel('Electricity Usage')
plt.grid(True)
plt.xticks(range(0, 24))
plt.show()