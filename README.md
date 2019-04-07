# gygli_sbd_train_and_test
Training, Dataset creation and Benchmarking repository for Gygli convolutional neural network (https://arxiv.org/pdf/1705.08214.pdf) shot boundary detector

This repo is an extension of the repo found at https://github.com/oladeha2/shot_boudary_detector and is for those who desire to further train the model or are in need of a frame based dataset for training of their own shot boundary detectors. The model is implemented using pytorh. 

It contains fully commented scripts that do the following:
1. Create the training and validation set, which consists of approximately 980,000 videos based on the content of two YouTube playlists https://www.youtube.com/playlist?list=PLxf1dxhJ3H9orru0qzPy1j5VDa41c4x7Z and https://www.youtube.com/playlist?list=PLxf1dxhJ3H9pzLItmYdDeBQa0RE8zmsC3. Videos should be added to the second playlist for attempted expansion (create_data_set.py)
2. Training the network using the above dataset (train.py)
3. Testing the trained model. This outputs the precision, recall and F1 scores for each video and overall average (test_video_rai.py)

The remaining files are custom made utilities. 

The following technologies are required for successful usage of all code in the repo

1. Pytube
2. Moviepy
3. Numpy
4. Pandas
5. Pytorch
6. Cuda
7. Matplotlib
8. Pillow
9. BeautifulSoup


Contact ameenbello@gmail.com for any questions regarding the contents of the repo


