#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns


# # Reading Foot_sensor Data
# 

# In[63]:


data = pd.read_csv('foot_sensor(P3_02).csv')
data['time_seconds'] = data['time_seconds'] = data.index * (1/120)
data['Acc_Z_derivative'] = np.gradient(data['Acc_Z'], data['time_seconds'])
print(data)

data.to_csv('foot_sensor(P3_02)_updated.csv', index=False)


# # Plotting Foot sensor data for visualisation 

# In[64]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

fig, axs = plt.subplots(2, 1, figsize=(25, 12))

# Plotting on the first subplot
axs[0].plot(data['time_seconds'], data['Acc_Z'], label='Acceleration Z')
axs[0].set_xlabel('Time (seconds)')
axs[0].set_ylabel('Acceleration Z')
axs[0].set_title('MultipleLocator(0.5)')
axs[0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
axs[0].set_title('Acceleration Z vs. Time (seconds)')


axs[0].legend()
axs[0].grid(True)

# Setting up the second subplot with specific x-axis locators
axs[1].plot(data['time_seconds'], data['Acc_Z'], label='Acceleration Z')
axs[1].set_xlabel('Time (seconds)')
axs[1].set_ylabel('Acceleration Z')
axs[1].set_title('MultipleLocator(0.5)')
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


# ##### Visual Conclusion of start time for IMU sensor is 5.6-5.65

# ## Using Derivative of acceleration to find the start time for IMU sensor recording 
# 
# 

# In[65]:


# Compute the derivative of Acc_Z with respect to time
data['Acc_Z_derivative'] = np.gradient(data['Acc_Z'], data['time_seconds'])

# Compute the change in the gradient (difference between consecutive values)
data['change_in_gradient'] = np.diff(data['Acc_Z_derivative'], prepend=data['Acc_Z_derivative'][0])

# Define the threshold
threshold = 10

# Create a mask for points where the change in gradient exceeds the threshold
mask = np.abs(data['change_in_gradient']) > threshold

# Skip the first 50 points for the mask and the corresponding results
skip_points = 50

# Adjust mask to skip the first 50 points
adjusted_mask = np.zeros_like(mask, dtype=bool)
adjusted_mask[skip_points:] = mask[skip_points:]

# Extract the points where the change in gradient is greater than the threshold
time_seconds_high_change = data['time_seconds'][adjusted_mask]
acc_z_high_change = data['Acc_Z'][adjusted_mask]
change_in_gradient_high_change = data['change_in_gradient'][adjusted_mask]

top_10_count = 10
time_seconds_top_10 = time_seconds_high_change.head(top_10_count)
acc_z_top_10 = acc_z_high_change.head(top_10_count)
change_in_gradient_top_10 = change_in_gradient_high_change.head(top_10_count)

# Find the index of the maximum change in gradient among these 10 values
max_change_index = change_in_gradient_top_10.abs().idxmax()

# Retrieve the corresponding time for the maximum change in gradient
max_time = time_seconds_top_10.loc[max_change_index]
max_change = change_in_gradient_top_10.loc[max_change_index]

# Display the results
print("Top 10 changes in gradient (excluding the first 50 points):")
for t, acc, change in zip(time_seconds_top_10, acc_z_top_10, change_in_gradient_top_10):
    print(f"Time: {t:.2f} s, Acceleration Z: {acc:.2f}, Change in Gradient: {change:.2f}")

print(f"\nAbsolute maximum change in gradient: {max_change:.2f} at Time: {max_time:.2f} s")


# # Cleaning of Bowl Arm sensor Data 
# 

# In[66]:


Bowling_arm2 = pd.read_csv('Bowling_arm2.csv')
print(Bowling_arm2)
Bowling_arm2['time_seconds'] = Bowling_arm2.index * (1/120)
Bowling_arm2_filtered = Bowling_arm2[Bowling_arm2['time_seconds'] >= max_time]
Bowling_arm2_filtered = Bowling_arm2_filtered.reset_index(drop=True)
Bowling_arm2_filtered['time_seconds'] = Bowling_arm2_filtered.index * (1/120)


Bowling_arm2_filtered = Bowling_arm2_filtered.loc[:, ~Bowling_arm2_filtered.columns.str.contains('^Unnamed')]

print(Bowling_arm2_filtered)

Bowling_arm2_filtered.to_csv('Bowling_arm2_filtered_reset.csv', index=False)
Bowling_arm2_filtered.to_csv('Bowling_arm2_filtered.csv', index=False)


# In[67]:


vicon_data2 = pd.read_csv('Vicon_Values(Pre-Cleaned)2.csv')
print(vicon_data2)
vicon_data2.rename(columns={'X': 'Rcar_X', 'Y': 'Rcar_Y', 'Z': 'Rcar_Z'}, inplace=True)
min_length = min(len(Bowling_arm2_filtered), len(vicon_data2))
Bowling_arm2_filtered = Bowling_arm2_filtered.iloc[:min_length].reset_index(drop=True)
vicon_data2 = vicon_data2.iloc[:min_length].reset_index(drop=True)

Bowling_arm2_filtered = pd.concat([Bowling_arm2_filtered, vicon_data2[['Rcar_X', 'Rcar_Y', 'Rcar_Z']]], axis=1)

# Display the updated DataFrame (optional)
print(Bowling_arm2_filtered)

# Save the updated DataFrame to a new CSV file
Bowling_arm2_filtered.to_csv('Bowling_arm2_with_vicon.csv', index=False)


# ## Filtering Out Noise in Acceleration Data Using a Pass Filter
# 
# - **Objective:** Filter out noise from acceleration data to obtain accurate results.
# - **Filter Types:**
#   - **High Pass Filter:**
#     - **Purpose:** Removes low-frequency noise and retains high-frequency signals.
#     - **Application:** Useful if noise predominantly consists of low-frequency components.
#   - **Low Pass Filter:**
#     - **Purpose:** Removes high-frequency noise and retains low-frequency signals.
#     - **Application:** Useful if noise predominantly consists of high-frequency components.
# - **Approach:**
#   - **Pre-Trials:** Determined that a high pass filter might be more effective.
#   - **Testing:** Plan to try both high pass and low pass filters to compare their performance.
# - **Goal:** Identify which filter provides the more accurate results
# 
# 

# #### Euler Rotation

# In[68]:


def euler_to_rotation_matrix(roll, pitch, yaw):
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    return R


# In[69]:


def gravity_vector_in_sensor_frame(roll, pitch, yaw):
    g = 9.81 
    g_global = np.array([0, 0, -g])
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    g_sensor = R.T @ g_global  
    return g_sensor


# In[70]:


def convert_and_correct_acceleration(acc_x, acc_y, acc_z, roll, pitch, yaw):
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    g_sensor = gravity_vector_in_sensor_frame(roll, pitch, yaw)
    acceleration_sensor_frame = np.array([acc_x, acc_y, acc_z])
    acceleration_corrected = acceleration_sensor_frame - g_sensor
    acceleration_global = R @ acceleration_corrected
    return acceleration_global


# In[71]:


def apply_conversion(row):
    acc_x, acc_y, acc_z = row['Acc_X'], row['Acc_Y'], row['Acc_Z']
    roll, pitch, yaw = row['Euler_X'], row['Euler_Y'], row['Euler_Z']
    return pd.Series(convert_and_correct_acceleration(acc_x, acc_y, acc_z, roll, pitch, yaw),
                     index=['Acc_Global_X', 'Acc_Global_Y', 'Acc_Global_Z'])

Bowling_arm2_filtered[['Acc_Global_X', 'Acc_Global_Y', 'Acc_Global_Z']] = Bowling_arm2_filtered.apply(apply_conversion, axis=1)

# Bowling_arm2_filtered['Acc_Y_derivative_Global'] = np.gradient(Bowling_arm2_filtered['Acc_Global_Y'], Bowling_arm2_filtered['time_seconds'])
# Bowling_arm2_filtered['Acc_Z_derivative_Global'] = np.gradient(Bowling_arm2_filtered['Acc_Global_Z'], Bowling_arm2_filtered['time_seconds'])
# Bowling_arm2_filtered['Acc_X_derivative_Global'] = np.gradient(Bowling_arm2_filtered['Acc_Global_X'], Bowling_arm2_filtered['time_seconds'])
# Bowling_arm2_filtered['Acc_Y_derivative'] = np.gradient(Bowling_arm2_filtered['Acc_Y'], Bowling_arm2_filtered['time_seconds'])
# Bowling_arm2_filtered['Acc_Z_derivative'] = np.gradient(Bowling_arm2_filtered['Acc_Z'], Bowling_arm2_filtered['time_seconds'])
# Bowling_arm2_filtered['Acc_X_derivative'] = np.gradient(Bowling_arm2_filtered['Acc_X'], Bowling_arm2_filtered['time_seconds'])
# print(Bowling_arm2_filtered)
Bowling_arm2_filtered.to_csv('Bowling_arm2_with_vicon_global_accelerations_corrected.csv', index=False)
print(Bowling_arm2_filtered)


# In[72]:


plt.figure(figsize=(10, 6))
plt.plot(Bowling_arm2_filtered['time_seconds'], Bowling_arm2_filtered['Acc_Global_Z'], label='Acc_Global_Z')
plt.xlabel('Time (s)')
plt.ylabel('Acc_Global_Z (m/s^2)')
plt.title('Global Z-Axis Acceleration vs Time')
plt.legend()
plt.grid(True)
plt.show()


# # Running Regression Models on Data

# In[75]:


print(Bowling_arm2_filtered.dtypes)
print(Bowling_arm2_filtered.head())
corr_matrix = Bowling_arm2_filtered.corr()
print(corr_matrix)
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[76]:


correlation_threshold = 0.1 

low_correlation_cols = [col for col in corr_matrix.columns if abs(corr_matrix.loc[col, 'Rcar_Z']) < correlation_threshold and col != 'Rcar_Z']

Bowling_arm2_filtered_cleaned = Bowling_arm2_filtered.drop(columns=low_correlation_cols)
print(Bowling_arm2_filtered_cleaned.head())
corr_matrix_cleaned = Bowling_arm2_filtered_cleaned.corr()
print(corr_matrix_cleaned)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_cleaned, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix (Filtered)')
plt.show()

# In[55]:

