import numpy as np

rewards = np.load('')
time_taken = np.load('')
won = np.load('')
lost = np.load('')
end_position = np.load('')

def mean_confidence_interval(array):
    mean = np.mean(array)
    critical_value = 1.96
    n = 100_000
    standard_error = np.sqrt(np.var(array)/n)

    confidence_interval = (mean - critical_value * standard_error, mean + critical_value * standard_error)
    return confidence_interval