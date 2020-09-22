from dataset import data_path
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cv2

def white_pixel_distribution(path=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    n_white_pix = np.sum(img >= 240)
    return n_white_pix

def get_split(fold, num_splits=5):
    train_path = data_path / 'train' / 'angyodysplasia' / 'images'
    train_file_names = np.array(sorted(list(train_path.glob('*'))))

    mask_path = data_path / 'train' / 'angyodysplasia' / 'masks'
    mask_file_names = np.array(sorted(list(mask_path.glob('*'))))

    nonpathcount = 0
    pathcount = 0
    train_file_labels = []
    for f in mask_file_names:
        wp = white_pixel_distribution(f)
        if wp <= 0:
            train_file_labels.append(0)
            nonpathcount += 1
        else:
            train_file_labels.append(1)
            pathcount += 1
   
    kf = StratifiedKFold(n_splits=num_splits, random_state=2018)
    
    ids = list(kf.split(train_file_names, train_file_labels))

    train_ids, val_ids = ids[fold]

    if fold == -1:
        return train_file_names, train_file_names
    else:
        return train_file_names[train_ids], train_file_names[val_ids]


if __name__ == '__main__':
    ids = get_split(0)
