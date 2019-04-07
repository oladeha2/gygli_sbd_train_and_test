import numpy as np
from PIL import Image
from random import choice
import cv2

def create_flash(img):
    """
    create an artificical flash in a frame by converting frame to YUV colour space
    increasing the Y values by a value of 200 where possible and converting back to RGB colour space
    """  
    value = 200  
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y,u,v = cv2.split(yuv)
    lim = 255 - value
    y[y > lim] = 255    
    y[y <= lim] += value
    final_yuv = cv2.merge((y,u,v))
    final_image = cv2.cvtColor(final_yuv, cv2.COLOR_YUV2RGB)
    return final_image

def create_flash_snippet(snippet):
    input_snippet = snippet    
    """
    create flashes in random consecutive positions in frames of input snippet
    """
    number_of_flash_frames = np.random.randint(4,11)
    possible_start_positions = 10 - number_of_flash_frames
    start = np.random.randint(0, possible_start_positions+1)
    end = start + number_of_flash_frames
    for i in range(start, end):
        input_snippet[i] = create_flash(input_snippet[i])
    return input_snippet

def no_transition(snippet, snippet_2=None):
    # also have approx --> want the falsh to not be as equally likke
    flash = np.random.choice([True, False], size=None, p=[0.3, 0.7])
    # flash = False #remove at some stage
    if flash:
        flash_snippet = create_flash_snippet(snippet)
        return flash_snippet, [1 for i in range(10)], 'no_transition'
    else:
        return snippet, [1 for i in range(10)], 'no_transition'

def hard_cut(snippet_1, snippet_2):
    # take the first five frames -- which is for the original current snippet
    return snippet_1[0:5] + snippet_2[5:10], [1,1,1,1,1,0,1,1,1,1], 'hard_cut'

def crop_cut(frames, next_frames=None):
    #50 percent full size crop ---> crop 64,64 --> 32,32
    out = [0,0,0,0,0,0,0,0,0,0]
    labels = [1,1,1,1,1,0,1,1,1,1]
    for i in range(len(frames)):
        if i <= 4:
            out[i] = frames[i]
        else:
            image = Image.fromarray(frames[i])
            width, height = image.size
            midpoint = width/2
            lower_limit = midpoint - 16  
            upper_limit = midpoint + 16
            crop_box = (lower_limit, lower_limit, upper_limit, upper_limit)
            cropped = image.resize((64,64),Image.BILINEAR, crop_box)
            out[i] = np.array(cropped)
    return out, labels, 'crop_cut'

def fade_in(frames, next_frames=None):
    #merge frames with a single black frame starting with full black frame ending in new shot
    out = []
    labels = []
    black_frame = np.multiply(frames[0],0)
    black_frame = Image.fromarray(black_frame)
    
    for i in range(len(frames)):
        if i < 9:
            current_frame = Image.fromarray(frames[i])
            cur_blend = Image.blend(black_frame, current_frame, 0.125*i)
            out.append(np.array(cur_blend))
            if i < 8:
                labels.append(0)
            else:
                labels.append(1)
        else:
            out.append(frames[i])
            labels.append(1)
    return out,labels, 'fade_in'

def fade_out(frames, next_frames=None):
    # merge frames with single black frame starting with current shot ending in blak frame
    out = [0,0,0,0,0,0,0,0,0,0]
    labels = [0,0,0,0,0,0,0,0,0,0]
    black_frame = np.multiply(frames[0], 0)
    black_frame = Image.fromarray(black_frame)
    
    for i in range(len(frames)-1,-1,-1):
        if(i > 1):
            labels[i] = 0
            current_frame = Image.fromarray(frames[i])
            blend = Image.blend(current_frame, black_frame, 0.125*i)
            out[i] = np.array(blend)
        else:
            out[i] = frames[i]
            labels[i] = 1
    return out, labels, 'fade_out'

def wipe(cur_shot, next_shot):
    out = []
    labels = []
    for i in range(10):
            if i == 0:
                labels.append(1)
            else:
                labels.append(0)
            sample = cur_shot[i]    
            sample = sample[:,round(i*(7.11)):64,:]
            shot = next_shot[i]
            shot[:,round(i*(7.11)):64,:] = sample
            out.append(shot)
    return out, labels, 'wipe'

def dissolve(shot, next_shot):
    # merge corresponding frames in both shots with increasing ratio 
    out = [] 
    labels = []
    for i in range(len(shot)):
            if i == 0:
                labels.append(1)
            else:
                labels.append(0)            
            frame = Image.fromarray(shot[i])
            frame_1 = Image.fromarray(next_shot[i])
            cur_blend = Image.blend(frame, frame_1, (i)*0.11)
            out.append(np.array(cur_blend))
    return out, labels, 'dissolve'

def selectAndReturnTransition(snippet_1, index, transition_decision, snippet_2=None):
    
    # set up dictionary for random selection of the transition required
    transitions_dict = {0: hard_cut,1: crop_cut,2: fade_in,
                        3: fade_out, 4: wipe, 5: dissolve}
    """
    Args:
        Pass in two snippets, ten frames each
        
    Functionality:    
        Function will firstly select true or false for whether transition is present
        then will choose value between 0 and 6 (not inclusive of 6) to select which
        transition is necessary. Integer and transitions are mapped as follows
        0 --> Hard Cut, 1 --> Crop Cut, 2 --> Fade In
        3 --> Fade Out, 4 --> Wipe, 5 --> Dissolve
        Implemented via the following dictionary:
        {0: hard_cut,1: crop_cut,2: fade_in,
         3: fade_out, 4: wipe, 5: dissolve}
    """

    # select whether a transition is necessary or not
    # print('transition:', transition_decision[index]['transition_decision'])
    # print('transition', transition)
    if transition_decision[index]['transition_present']:
        # currently only classifying for hard cuts with the entire data set
        a, labels, ident = transitions_dict.get(transition_decision[index]['transition_type'])(snippet_1, snippet_2)
    else:
        a, labels, ident = no_transition(snippet_1)
    return np.array(a), labels, ident


