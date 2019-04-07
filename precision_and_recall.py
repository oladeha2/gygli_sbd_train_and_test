import numpy as np
import os
import matplotlib.pyplot as plt

# calculate precicon recall and F1 score for model

def get_transition_truths(gt):

    transition_truths = []
    for idx, truth in enumerate(gt):
        if idx < len(gt) - 1:
            transition_truths.append([truth[1], gt[idx+1][0]])
    return transition_truths

def get_precision(prediction, gt):

    success = 1
    for truth in gt:
        if prediction > int(truth[0]) and prediction < int(truth[1]):
            success = 0
            return success
        else: 
            success = 1
    return success

def compute_precision(predictions, gt):

    precisions = []
    for prediction in predictions:
        precisions.append(get_precision(prediction=prediction, gt=gt))
    p_value = np.mean(np.array(precisions))
    return p_value

def get_recall(prediction, gt):

    success = 0
    for truth in gt:
        if prediction >= int(truth[0]) and prediction <= int(truth[1]):
            return 1
        else:
            success = 0
    return success

def compute_recall(predictions, gt):
    
    recalls = []
    for prediction in predictions:
        success = get_recall(prediction=prediction, gt=gt)
        recalls.append(success)
    r_value = np.mean(np.array(recalls))
    return r_value

def getF1(p, r):

    a = 2 * (p * r)
    b = p + r
    return a / b 


# predictions for moedel are saved in the directory 'predictions/<model_name>' pass directory into function to analyse its prediction performance after testing
def analyse_model(model_predictions):

    file_names = os.listdir(model_predictions)
    prediction_path = model_predictions
    ground_truth_path = 'ground_truths/'

    precs = []
    recs = []
    f1s = []

    for index, name in enumerate(file_names):
        vid_name = name.replace('.txt', '')
        pred_path = prediction_path + name
        gd_path = ground_truth_path + name
        print('VIDEO', index + 1, ':', vid_name)
        print(pred_path)
        print(gd_path)

        with open(gd_path) as f:
            ground_truths = f.readlines()

        g_truths = np.array([line.strip().replace('\t', ' ').split(' ') for line in ground_truths])

        with open(pred_path) as f:
            predictions = f.readlines()

        preds = [int(pred.strip()) for pred in predictions]
        print('pred len:', len(preds))

        print('g_truths len:', len(g_truths))

        t_truths = get_transition_truths(gt=g_truths)
        print('t_truths len:', len(t_truths))

        precision = compute_precision(predictions=preds, gt=g_truths)
        print('precision:', precision)
        precs.append(precision)

        recall = compute_recall(predictions=preds, gt=t_truths)
        print('recall:', recall)
        recs.append(recall)

        f1 = getF1(p=precision, r=recall)
        f1s.append(f1)
        print('F1 Score:', f1)

        print('')

    print('average precision:', np.mean(np.array(precs)))
    print('average recall:', np.mean(np.array(recs)))
    print('average F1 Score:', np.mean(np.array(f1s)))

    print('-----------------------------------------------------------')
