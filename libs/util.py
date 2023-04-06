import numpy as np

def get_magnitude_and_angle(fft2_output):
    M_data = np.abs(fft2_output)
    theta_data = np.angle(fft2_output)
    return M_data, theta_data


def scale_values(data, new_min, new_max):
    actual_min = np.min(data)
    actual_max = np.max(data)
    return np.array(list(map(lambda x: (x - actual_min)/(actual_max - actual_min)*(new_max-new_min)+new_min, data)))
    