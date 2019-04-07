import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from snippet import getSnippet
from math import floor
from transition_network import TransitionCNN
from utilities import normalize_frame, print_shape
import pandas as pd
import os
import time
from TestVideo import TestVideo, return_start_and_end
from precision_and_recall import analyse_model


# get predictions of model for all videos in the rai data set for testing of model against RAI benchmark


device = 'cuda'

#load the two models
fifty_model = TransitionCNN()

#change model needed for testing here
fifty_model.load_state_dict(torch.load('./models/shot_boundary_detector_even_distrib.pt'))

fifty_model.to(device)

print(fifty_model.parameters)

fifty_model.eval()

detcted_frames = []

prediction_times = []

# change directory name to mach name of model for prediction save location
os.makedirs('predictions/shot_boundary_detector_even_distrib/', exist_ok=True)

text_files = os.listdir('test_files/rai')
print(text_files)

for vid, text_file in enumerate(text_files):
    path = './test_files/rai/' + text_file
    #load new video in full ten video test set
    test_video = TestVideo(path, sample_size=100, overlap=9)
    test_loader = DataLoader(test_video, batch_size=1, num_workers=1)
    print('video', vid+1, 'length:', len(test_video))

    print('number of lines/frames:', test_video.get_line_number())
    
    prediction_file_name = 'predictions/shot_boundary_detector_even_distrib/' + text_file
    print('prediction file name:', prediction_file_name)

    f = open(prediction_file_name, 'w+')


    print('computing and writing prediction for video:', text_file.replace('.txt', ''),'...............')

    video_indexes = []
    vals = np.arange(test_video.get_line_number())
    length = len(test_video)

    for i in range(length):
        s,e = return_start_and_end(i)
        video_indexes.append(vals[s:e])

    # frame predicted as shot boundary is written to text file
    start_time = time.time()
    for indx, batch in enumerate(test_loader):
        batch.to(device)
        batch = batch.type('torch.cuda.FloatTensor')
        predictions = fifty_model(batch)
        predictions = predictions.argmax(dim=1).cpu().numpy()
        for idx, prediction_set in enumerate(predictions):
            for i, prediction in enumerate(prediction_set):
                if prediction[0][0] == 0:
                    frame_index = video_indexes[indx][i+5]
                    f.write(str(frame_index) + '\n')
    f.close()

    end_time = time.time()
    prediction_time = end_time - start_time
    prediction_times.append(prediction_time)
    print('predictions complete for video:', text_file.replace('.txt', ''), '..............')

    print('')

df_pred_times = pd.DataFrame(prediction_times, columns=['prediction_time'])
df_pred_times.to_csv('csv_data/shot_boundary_detector_even_distrib_prediction_times.csv')


analyse_model('predictions/shot_boundary_detector_even_distrib/')







