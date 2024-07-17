import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt

# Function to convert Euler angles to a rotation matrix
def euler_to_rotation_matrix(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = R_z @ R_y @ R_x
    return R

def zero_velocity_update_single_axis(acc_data, gyro_data, acc_threshold, gyro_threshold, window_size):
    stationary = np.zeros(len(acc_data), dtype=int)
    for i in range(window_size, len(acc_data) - window_size):
        acc_window = acc_data[i-window_size:i+window_size]
        acc_mean = np.mean(acc_window, axis=0)
        if abs(acc_mean) > acc_threshold:
            gyro_window = gyro_data[i-window_size:i+window_size]
            gyro_mean = np.std(gyro_window, axis=0)
            if abs(gyro_mean) > gyro_threshold:
                stationary[i-window_size:i+window_size] = 10
    return stationary

def zero_velocity_update(acc_data, gyro_data, acc_threshold, gyro_threshold, window_size):
    stationary = np.zeros(len(acc_data), dtype=int)
    for i in range(window_size, len(acc_data) - window_size):
        acc_window = acc_data[i-window_size:i+window_size]
        acc_mean = np.mean(acc_window, axis=0)
        
        if np.all(np.abs(acc_mean) > acc_threshold):
            gyro_window = gyro_data[i-window_size:i+window_size]
            gyro_mean = np.std(gyro_window, axis=0)
            
            if np.all(np.abs(gyro_mean) > gyro_threshold):
                stationary[i-window_size:i+window_size] = 10
                
    return stationary

# Load the data from a CSV file
input_csv_path = '../data/raw/left_heel.csv'  # Replace with your input CSV file path
data = pd.read_csv(input_csv_path)

result = data

print("time",data['SampleTimeFine'])

time = data['SampleTimeFine']

initial_time = time[0]
adjusted_time = time - initial_time

# Convert to seconds if the original time is in microseconds
adjusted_time_in_seconds = adjusted_time / 1e6

print("\nAdjusted time (in seconds):")
print(adjusted_time_in_seconds)

data['SampleTimeFine'] = adjusted_time_in_seconds

# # print("data",data['acc_z'])

# # data['acc_z'] = data['acc_z'].apply(lambda x: x-0.25)

# Rotate each acceleration vector
global_acc_data = []

# Initialize list to store global acceleration data minus gravity
global_acc_data_minus_gravity = []

# Define gravity vector in global coordinate system
gravity_vector = np.array([0, 0, -9.80665])

for index, row in data.iterrows():
    roll = row["Euler_X"]
    pitch = row["Euler_Y"]
    yaw = row["Euler_Z"]
    
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    
    acc_device = np.array([row["Acc_X"], row["Acc_Y"], row["Acc_Z"]])

    acc_global = R @ acc_device

    acc_global_minus_gravity = acc_global - gravity_vector
    
    global_acc_data.append(acc_global)

    global_acc_data_minus_gravity.append(acc_global_minus_gravity)

global_acc_data_mean_value = np.mean(global_acc_data_minus_gravity[0:100],axis=0)

# Convert results to a DataFrame
global_acc_data_minus_gravity = np.array(global_acc_data_minus_gravity)
global_acc_df = pd.DataFrame(global_acc_data_minus_gravity, columns=["Acc_X_global", "Acc_Y_global", "Acc_Z_global"])

print("global_acc_df",global_acc_df)

# Combine with original data
result = pd.concat([data, global_acc_df], axis=1)

print("global_acc_data_mean_value",global_acc_data_mean_value)
result['Acc_X_global'] = result['Acc_X_global'].apply(lambda x: x-global_acc_data_mean_value[0])
result['Acc_Y_global'] = result['Acc_Y_global'].apply(lambda x: x-global_acc_data_mean_value[1])
result['Acc_Z_global'] = result['Acc_Z_global'].apply(lambda x: x-global_acc_data_mean_value[2])

# Define thresholds and window size for ZUPT
acc_threshold = 0.2 # m/s^2, this value can be tuned
gyro_threshold = 0.2  # rad/s, this value can be tuned
window_size = 30  # number of samples, this value can be tuned

# Apply zero velocity update algo
print(len(result['Acc_Z_global']))
zupt_result = zero_velocity_update(result[['Acc_X_global','Acc_Y_global','Acc_Z_global']].values, result[['Gyr_X','Gyr_Y','Gyr_Z']].values, acc_threshold, gyro_threshold, window_size)

result['Stationary'] = zupt_result

velocity = np.zeros(3, dtype='float64')
displacement = np.zeros(3, dtype='float64')
velocity_z = [0]  # Starting with an initial velocity of 0
velocity_y = [0]  # Starting with an initial velocity of 0
velocity_x = [0]  # Starting with an initial velocity of 0
displacement_z = [0]  # Starting with an initial displacement of 0
displacement_y = [0]  # Starting with an initial displacement of 0
displacement_x = [0]  # Starting with an initial displacement of 0
drift_offset = [0]  # Drift offset for velocity calculation
velocity_x_drift_removed = [0]  # Starting with an initial velocity of 0
velocity_y_drift_removed = [0]  # Starting with an initial velocity of 0
velocity_z_drift_removed = [0]  # Starting with an initial velocity of 0

dt = 1/120  # Assuming 60 Hz sampling rate

for i in range(1, len(result)):
    
    if result.loc[i, 'Stationary']:
        velocity_x.append(velocity_x[-1] + result['Acc_X_global'][i] * dt)
        velocity_y.append(velocity_y[-1] + result['Acc_Y_global'][i] * dt)
        velocity_z.append(velocity_z[-1] + result['Acc_Z_global'][i] * dt)
    else:
        velocity_x.append(0)
        velocity_y.append(0)
        velocity_z.append(0)

    displacement_x.append(displacement_x[-1] + velocity_x[i] * dt)
    displacement_y.append(displacement_y[-1] + velocity_y[i] * dt)
    displacement_z.append(displacement_z[-1] + velocity_z[i] * dt)


result['Displacement_y'] = displacement_y
result['Velocity_y'] = velocity_y

result['Displacement_x'] = displacement_x
result['Velocity_x'] = velocity_x

# # Save the results to a new CSV file
output_csv_path = '../data/processed/test1.csv'  # Replace with your output CSV file path
result.to_csv(output_csv_path, index=False)

# Assuming 'result' DataFrame is already populated with global acceleration data
# Create a figure with two subplots
# Create subplots
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 3)

# Plotting global accelerometer data
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(result['SampleTimeFine'], result['Acc_Y_global'], label='Acc_Y_global')
ax1.plot(result['SampleTimeFine'], result['Stationary'], label='Stationary')
ax1.set_xlabel('Index')
ax1.set_ylabel('Acceleration (m/s^2)')
ax1.set_title('Global Acceleration Components')
ax1.legend()
ax1.grid(True)

# Plotting velocity data
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(result['SampleTimeFine'], result['Velocity_y'], label='Velocity_y')
ax2.plot(result['SampleTimeFine'], result['Displacement_y'], label='Displacement_y')
ax2.set_xlabel('Index')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('Velocity and Displacement Y Components')
ax2.legend()
ax2.grid(True)

# Plotting velocity data for x-axis
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(result['SampleTimeFine'], result['Velocity_x'], label='Velocity_x')
ax3.plot(result['SampleTimeFine'], result['Displacement_x'], label='Displacement_x')
ax3.set_xlabel('Index')
ax3.set_ylabel('Velocity (m/s)')
ax3.set_title('Velocity and Displacement X Components')
ax3.legend()
ax3.grid(True)

# Create a subplot for the 3D plot
ax4 = fig.add_subplot(gs[1, :], projection='3d')

filtered_positions_x = result['Displacement_x'][result['Stationary'] == 10]
filtered_positions_y = result['Displacement_y'][result['Stationary'] == 10]
filtered_time = result['SampleTimeFine'][result['Stationary'] == 10]

# Plot the positions with time
ax4.plot(filtered_positions_x, filtered_positions_y, filtered_time, marker='o', markersize=1)  # Adjust markersize here
ax4.set_title('Foot Position on x-y Plane Over Time')
ax4.set_xlabel('x Position')
ax4.set_ylabel('y Position')
ax4.set_zlabel('Time')
ax4.grid(True)

# Adjust layout and display
plt.tight_layout()
plt.show()

