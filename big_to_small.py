import numpy as np


file_name_big = 'C:/Users/Public/raw_data/raw_data_screen.npy'
counter_for_filename = 490
train_data = np.load(file_name_big, allow_pickle=True)
saving_data = []

for data in train_data:
    saving_data.append(data)

    if len(saving_data) % 500 == 0:
        print('{} : {} Mb'.format(counter_for_filename, int(counter_for_filename * 22.58)))
        file_name = 'C:/Users/Public/raw_data/raw_data_screen{}.npy'.format(counter_for_filename)
        np.save(file_name, saving_data)
        counter_for_filename += 1
        saving_data = []
