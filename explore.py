import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace with actual data)
data = {
    'baseline_value': np.random.randint(110, 160, 100),
    'accelerations': np.random.randint(0, 10, 100),
    'fetal_movement': np.random.randint(0, 10, 100),
    'uterine_contractions': np.random.randint(0, 10, 100),
    'light_decelerations': np.random.randint(0, 10, 100),
    'severe_decelerations': np.random.randint(0, 10, 100),
    'prolongued_decelerations': np.random.randint(0, 10, 100),
    'abnormal_short_term_variability': np.random.randint(0, 10, 100),
    'mean_value_of_short_term_variability': np.random.randint(0, 10, 100),
    'percentage_of_time_with_abnormal_long_term_variability': np.random.randint(0, 10, 100),
    'mean_value_of_long_term_variability': np.random.randint(0, 10, 100),
    'histogram_width': np.random.randint(0, 10, 100),
    'histogram_min': np.random.randint(0, 10, 100),
    'histogram_max': np.random.randint(0, 10, 100),
    'histogram_number_of_peaks': np.random.randint(0, 10, 100),
    'histogram_number_of_zeroes': np.random.randint(0, 10, 100),
    'histogram_mode': np.random.randint(0, 10, 100),
    'histogram_mean': np.random.randint(0, 10, 100),
    'histogram_median': np.random.randint(0, 10, 100),
    'histogram_variance': np.random.randint(0, 10, 100),
    'histogram_tendency': np.random.randint(0, 10, 100),
    'fetal_health': np.random.randint(1, 3, 100)
}
""""accelerations",
"prolongued_decelerations",
 "abnormal_short_term_variability",
   "percentage_of_time_with_abnormal_long_term_variability"
     and "mean_value_of_long_term_variability" 
     are the features with higher correlation with fetal_health.

# Columns in the data
      Index(['baseline value', 'accelerations', 'fetal_movement',
       'uterine_contractions', 'light_decelerations', 'severe_decelerations',
       'prolongued_decelerations', 'abnormal_short_term_variability',
       'mean_value_of_short_term_variability',
       'percentage_of_time_with_abnormal_long_term_variability',
       'mean_value_of_long_term_variability', 'histogram_width',
       'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
       'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
       'histogram_median', 'histogram_variance', 'histogram_tendency',
       'fetal_health'],
      dtype='object')
 """
# Convert to DataFrame
df = pd.DataFrame(data)

# # Plot the CTG signal
# plt.figure(figsize=(15, 10))

# # Plot baseline value (FHR)
# plt.subplot(3, 1, 1)
# plt.plot(df['baseline_value'], label='Baseline Value (FHR)')
# plt.title('Baseline Value (FHR)')
# plt.xlabel('Time')
# plt.ylabel('Heart Rate (bpm)')
# plt.legend()

# # Plot uterine contractions
# plt.subplot(3, 1, 2)
# plt.plot(df['uterine_contractions'], label='Uterine Contractions', color='orange')
# plt.title('Uterine Contractions')
# plt.xlabel('Time')
# plt.ylabel('Contractions')
# plt.legend()

# # Plot accelerations and decelerations
# plt.subplot(3, 1, 3)
# plt.plot(df['accelerations'], label='Accelerations', color='green')
# plt.plot(df['light_decelerations'], label='Light Decelerations', color='red')
# plt.plot(df['severe_decelerations'], label='Severe Decelerations', color='purple')
# plt.title('Accelerations and Decelerations')
# plt.xlabel('Time')
# plt.ylabel('Count')
# plt.legend()

# plt.tight_layout()
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace with actual data)
data = {
    'accelerations': np.random.randint(0, 10, 100),
    'prolongued_decelerations': np.random.randint(0, 10, 100),
    'abnormal_short_term_variability': np.random.randint(0, 10, 100),
    'percentage_of_time_with_abnormal_long_term_variability': np.random.randint(0, 100, 100),
    'mean_value_of_long_term_variability': np.random.randint(0, 10, 100)
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plot the specified columns
plt.figure(figsize=(15, 10))

# Plot accelerations
plt.subplot(3, 2, 1)
plt.plot(df['accelerations'], label='Accelerations')
plt.title('Accelerations')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()

# Plot prolongued decelerations
plt.subplot(3, 2, 2)
plt.plot(df['prolongued_decelerations'], label='Prolongued Decelerations', color='orange')
plt.title('Prolongued Decelerations')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()

# Plot abnormal short-term variability
plt.subplot(3, 2, 3)
plt.plot(df['abnormal_short_term_variability'], label='Abnormal Short-Term Variability', color='green')
plt.title('Abnormal Short-Term Variability')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()

# Plot percentage of time with abnormal long-term variability
plt.subplot(3, 2, 4)
plt.plot(df['percentage_of_time_with_abnormal_long_term_variability'], label='Percentage of Time with Abnormal Long-Term Variability', color='red')
plt.title('Percentage of Time with Abnormal Long-Term Variability')
plt.xlabel('Time')
plt.ylabel('Percentage')
plt.legend()

# Plot mean value of long-term variability
plt.subplot(3, 2, 5)
plt.plot(df['mean_value_of_long_term_variability'], label='Mean Value of Long-Term Variability', color='purple')
plt.title('Mean Value of Long-Term Variability')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
