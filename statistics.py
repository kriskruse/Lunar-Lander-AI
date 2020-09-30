import numpy as np

rewards = np.load('rewards.npy')
time_taken = np.load('time_taken.npy')
won = np.load('won.npy')
lost = np.load('lost.npy')
end_position = np.load('end_position.npy')



def get_distance(end_positions):
    distance = []
    for i in end_positions:
        distance.append(np.sqrt(i[0]**2 + i[1]**2))
    return distance

def mean_confidence_interval(array):
    mean = np.mean(array)
    critical_value = 1.96
    n = 100_000
    standard_error = np.sqrt(np.var(array)/n)

    confidence_interval = (mean - critical_value * standard_error, mean + critical_value * standard_error)
    return confidence_interval

distance = get_distance(end_position)
all = [rewards, time_taken, distance]

for i in all:
    print(np.mean(i))
print("fuel")
print(mean_confidence_interval(rewards))
print("tid")
print(mean_confidence_interval(time_taken))

print("distance")
print(mean_confidence_interval(distance))


