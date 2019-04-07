import os
import pandas as pd 

def get_file_distributions(path):
    print('getting frame information for videos in path', path, '......')
    dist = []
    names = os.listdir(path)
    for i in range(len(names)):
        file_name = path + names[i]
        dist.append(len(os.listdir(file_name)))
    print('frame information obtained')
    return dist


train_files = get_file_distributions('./data/train/')
valid_files = get_file_distributions('./data/valid/')

df_train = pd.DataFrame(train_files, columns=["train_frames"])
df_valid = pd.DataFrame(valid_files, columns=["valid_frames"])

total = df_train.join(df_valid, how='outer')

print('creating csv ...........')
total.to_csv('csv_data/frame_distribution.csv', index=False)
print('csv created')

