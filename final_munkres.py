import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors

# Define lists to store results
r_list = []
el_list = []
az_list = []
r=[]
az=[]
el=[]
class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1))
        self.Z1 = np.zeros((3, 1))  # Measurement vector
        self.Z2 = np.zeros((3, 1))
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 9.21  # 95% confidence interval for Chi-square distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            dt = time - self.Meas_Time
            self.vx = (self.Z2[0] - self.Z1[0]) / dt
            self.vy = (self.Z2[1] - self.Z1[1]) / dt
            self.vz = (self.Z2[2] - self.Z1[2]) / dt
            self.Meas_Time = time
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            if dt > 0:
                self.vx = (self.Z[0] - self.Sf[0]) / dt
                self.vy = (self.Z[1] - self.Sf[1]) / dt
                self.vz = (self.Z[2] - self.Sf[2]) / dt

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt**2) / 2.0
        T_3 = (dt**3) / 6.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q *= self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)
        print("Updated filter state (Sf):", self.Sf.flatten())

def form_measurement_groups(measurements, max_time_diff=50):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

def sph2cart(az, el, r):
    x = r * np.cos(np.radians(el)) * np.sin(np.radians(az))
    y = r * np.cos(np.radians(el)) * np.cos(np.radians(az))
    z = r * np.sin(np.radians(el))
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    az = np.degrees(np.arctan2(y, x))
    if az < 0.0:
        az += 360
    return r, az, el

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el


def cart2sph2(x:float,y:float,z:float,filtered_values_csv):
    for i in range(len(filtered_values_csv)):
    #print(np.array(y))
    #print(np.array(z))
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i]/np.sqrt(x[i]**2 + y[i]**2))*180/3.14)
        az.append(math.atan(y[i]/x[i]))
         
        if x[i] > 0.0:                
            az[i] = 3.14/2 - az[i]
        else:
            az[i] = 3*3.14/2 - az[i]       
        
        az[i]=az[i]*180/3.14 

        if(az[i]<0.0):
            az[i]=(360 + az[i])
    
        if(az[i]>360):
            az[i]=(az[i] - 360)   
      

        # print("Row ",i+1)
        # print("range:", r)
        # print("azimuth", az)
        # print("elevation",el)s
        # print()
  
    return r,az,el

def hungarian_algorithm(cost_matrix):
    cost_matrix = np.array(cost_matrix)
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))

def data_association(predictions, measurements):
    cost_matrix = []
    for pred in predictions:
        cost_row = []
        for meas in measurements:
            cost = np.linalg.norm(pred[:3] - meas[:3])  # Euclidean distance
            cost_row.append(cost)
        cost_matrix.append(cost_row)
    
    indexes = hungarian_algorithm(cost_matrix)
    return indexes

def main():
    file_path = 'ttk_84_test.csv'
    measurements = read_measurements_from_csv(file_path)

    csv_file_predicted = "ttk_84_test.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values
    measured_values_csv = df_predicted[['MT', 'MR', 'MA', 'ME']].values

    A = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3],filtered_values_csv)
    number = 1000
    result = np.divide(A[0], number)

    measurement_groups = form_measurement_groups(measurements, max_time_diff=50)

    kalman_filter = CVFilter()

    time_list = []
    r_list = []
    az_list = []
    el_list = []

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")
        predicted_states = []
        for i, (rng, azm, ele, mt) in enumerate(group):
            print(f"Measurement {i + 1}: (r={rng}, az={azm}, el={ele}, t={mt})")
            x, y, z = sph2cart(azm, ele, rng)
            if not kalman_filter.first_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            elif kalman_filter.first_rep_flag and not kalman_filter.second_rep_flag:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            else:
                kalman_filter.initialize_filter_state(x, y, z, kalman_filter.vx, kalman_filter.vy, kalman_filter.vz, mt)
                kalman_filter.predict_step(mt)
                predicted_states.append(kalman_filter.Sp)

        if predicted_states:
            indexes = data_association(predicted_states, group)
            for pred_idx, meas_idx in indexes:
                Z = np.array(group[meas_idx][:3]).reshape((3, 1))
                kalman_filter.update_step(Z)
                print(f"Measurement index {meas_idx}:")
                print("Predicted State (Sp):", predicted_states[pred_idx].flatten())
                print("Updated State (Sf):", kalman_filter.Sf.flatten())

                r_val, az_val, el_val = cart2sph(kalman_filter.Sf[0], kalman_filter.Sf[1], kalman_filter.Sf[2])
                r_list.append(r_val)
                az_list.append(az_val)
                el_list.append(el_val)
                time_list.append(group[meas_idx][3])

    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(time_list, r_list, label='Filtered Range (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], result, label='Filtered Range (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 1], label='Measured Range (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Range (r)', color='black')
    plt.title('Range vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(time_list, az_list, label='Filtered Azimuth (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[1], label='Filtered Azimuth (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 2], label='Measured Azimuth (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Azimuth (az)', color='black')
    plt.title('Azimuth vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(facecolor="white")
    plt.scatter(time_list, el_list, label='Filtered Elevation (code)', color='green', marker='*')
    plt.scatter(filtered_values_csv[:, 0], A[2], label='Filtered Elevation (track id 31)', color='red', marker='*')
    plt.scatter(measured_values_csv[:, 0], measured_values_csv[:, 3], label='Measured Elevation (code)', color='blue', marker='o')
    plt.xlabel('Time', color='black')
    plt.ylabel('Elevation (el)', color='black')
    plt.title('Elevation vs. Time', color='black')
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    mplcursors.cursor(hover=True)
    plt.show()

if __name__ == "__main__":
    main()
