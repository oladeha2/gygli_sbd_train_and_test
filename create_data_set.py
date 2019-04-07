from moviepy.editor import *
import os
import numpy as np
import matplotlib.pyplot as plt
import requests as req
import numpy as np
from PIL import Image
from pytube import YouTube
from bs4 import BeautifulSoup as bs
from pytube import exceptions as ex 

from video_processing import six_four_crop_video

# download each video through the extracted link and resize the video to 64x64
def downoadLinksAndResizeVideos(links, path):
    cropped_videos = []
    """return an array of all the videos downloaded and resized videos are saved in the specified path
        create the path if it does not exist """

    if os.path.isdir(path) == False:
        os.makedirs(path)
        print('directory created')
    
    for idx, link in enumerate(links):
        print('-------------------------------------------------------------------------------------------------')
        file_name = 'clip' + str(idx + 1)
        try:
            yt = YouTube(link)
        except ex.RegexMatchError:
            print('Video Unavailable --> Regex Match Error')
            continue
        except ex.VideoUnavailable:
            print('Video Unavailable  --> Video Unavailable Error')
            continue
        except KeyError:
            print('Video Unavailable --> Key Error')
        else:
            yt.streams.filter(file_extension = 'mp4').first().download(path, file_name)
            print('Video Downloaded')

            #get each downloaded video and resize
            video = VideoFileClip(path + '/'+ file_name + '.mp4')
            print('duration', video.duration)
            print('fps', video.fps)
            print('frames', video.duration*video.fps)
            six_four_video = six_four_crop_video(video)
            w, h = six_four_video.size
            if w == 64 and h == 64:
                cropped_videos.append(six_four_video)
            print('-------------------------------------------------------------------------------------------------')
    return cropped_videos

# save largest multiple of ten frames from all videos and 
def extractAndSaveFrames(data_type, videoList):
    """
    Args:
        data_type = string ('train' or 'valid') determines where the data will be saved and in which file
    Frames for each video are extracted and saved in corrcet location
    """
    if (data_type == 'train') or (data_type == 'valid'):
        f = open(data_type + '.txt', 'w+')
    else:
        raise ValueError('First Argument must be string test or valid')

    # /data/<test or valid>/<index value>
    for i, video in enumerate(videoList):
        path = './data/' + data_type + '/' + str(i+1)
        frames = [frame for frame in video.iter_frames()]
        # cut the number of frames so data is always in groups of 10 
        frames = frames[0: len(frames) - len(frames)%10]
        if os.path.isdir(path) == False:
            os.makedirs(path)
        print('number of frames for current video ', (i+1), 'is: ', len(frames))
        for j, frame in enumerate(frames):
            frame_path = path + '/frame_' + str(j+1) + '.png'
            im = Image.fromarray(frame)
            im.save(frame_path)
            f.write(frame_path + '\n')
    f.close()

# load sample videos for testing. These videos will be downloaded instead in a specific manner for training and for validation

def getPlaylistLinks(playList):

    # get YouTube playlist and extract the link for each video
    getRequest = req.get(playList)
    pageSource = getRequest.text
    parser = bs(pageSource, 'html.parser')
    videoLinks = parser.find_all('a', {'dir': 'ltr'})
    len(videoLinks)

    videoList = []
    domain = "https://www.youtube.com"
    for link in videoLinks:
        tmp = domain + link.get('href')
        tmp = tmp.replace("&t=0s", '')
        if "watch" in tmp:
            videoList.append(tmp)
    return videoList

long_1 = getPlaylistLinks('https://www.youtube.com/playlist?list=PLxf1dxhJ3H9orru0qzPy1j5VDa41c4x7Z')
long_2 = getPlaylistLinks('https://www.youtube.com/playlist?list=PLxf1dxhJ3H9pzLItmYdDeBQa0RE8zmsC3')


print('playlist one', len(long_1))
print('playlist two', len(long_2))

videoList = long_1 + long_2 

print('full list length:', len(videoList))

# split the videos 80:20 ratio for training and validation sets
split = round(len(videoList)*0.8)
train = videoList[0:split]
valid = videoList[split:len(videoList)]

print('len train:', len(train))
print('len valid:', len(valid))


train_set = downoadLinksAndResizeVideos(train, './data/play_list_videos/train')
valid_set = downoadLinksAndResizeVideos(valid, './data/play_list_videos/valid')    

extractAndSaveFrames('train',train_set)
extractAndSaveFrames('valid',valid_set)

print('All frames and videos for training and valudation have been saved to the directory data/')
print('All frame paths for training and validation have been saved to train.txt and valid.txt')