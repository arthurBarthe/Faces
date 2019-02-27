# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 18:17:11 2019

@author: Arthur
Function to load parts of the data set IMDB-WIKI into a pandas dataFrame
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
(Rasmus Rothe, Radu Timofte, Luc Van Gool)
"""

import scipy.io
import pandas as pd
import os.path
import numpy as np

def load_IMDB_WIKI(file_path, start = 0, nb = 5000, 
                   info_list = ['face_location']):
    """Load parts of the data set IMDB-WIKI into a pandas dataFrame. One can
    specify the meta information required through info_list"""
    df = pd.DataFrame()
    mat = scipy.io.loadmat(file_path)
    face_score = mat['wiki']['face_score'][0][0][0][start : start + nb]
    second_face_score = mat['wiki']['second_face_score'][0][0][0][start : start + nb]
    face_location_ = mat['wiki']['face_location'][0][0][0][start : start + nb]
    image_path_ = mat['wiki']['full_path'][0][0][0][start : start + nb]
    face_location = []
    image_path = []
    dir_name = os.path.dirname(file_path)
    for i in range(nb):
        if i % 500 == 0:
            print('Loading face data set: {} %'.format(i/nb*100))
        face_location.append(np.array(face_location_[i][0]))
        rel_path = image_path_[i][0].replace('/', '\\')
        image_path.append('{}\\{}'.format(dir_name, rel_path))
    df['face_score'] = pd.Series(face_score)
    df['second_face_score'] = pd.Series(second_face_score)
    df['face_location'] = face_location
    df['image'] = image_path
    return df

if __name__ == '__main__':
    f = 'D:\Data sets\Faces\wiki.tar\wiki\wiki\wiki.mat'
    df = load_IMDB_WIKI(f)
