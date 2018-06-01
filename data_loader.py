import os
import time
import pickle
import json

from sklearn.externals import joblib
from glob import glob

def load_dataset(cars_folder = os.path.join('data', 'vehicles'),
                 notcars_folder = os.path.join('data', 'non-vehicles')):

    print('Loading dataset...')
    t1 = time.time()

    cars_ptn = os.path.join(cars_folder, '**', '*.png')
    notcars_ptn = os.path.join(notcars_folder, '**', '*.png')

    cars = glob(cars_ptn, recursive=True)
    notcars = glob(notcars_ptn, recursive=True)

    t2 = time.time()

    print('Loading dataset...DONE ({} s, Car images: {}, Not-Car images: {})'.format(round(t2 - t1, 2), len(cars), len(notcars)))

    return cars, notcars

def load_search_params(file = 'search_params.json'):
    
    with open(file) as f:
        data = json.load(f)
    
    return data

def save_model(model, model_file = os.path.join('models', 'model.p')):
    print('Saving model to {}...'.format(model_file))
    joblib.dump(model, model_file)

def load_model(model_file = os.path.join('models', 'model.p')):
    print('Loading model from {}...'.format(model_file))
    return joblib.load(model_file)
