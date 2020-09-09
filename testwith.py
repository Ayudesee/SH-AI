
training_data = []
with open('training_data_GRAY.npy') as f:
    for data in f.readlines():
        training_data.append(data)

print(training_data)

