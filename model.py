import numpy as np
import cv2
import glob
import time

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from features_extractor import FeaturesExtractor
from data_loader import *
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

def extract_train_test(cars, notcars, params, rand_state = np.random.randint(0, 100), process_pool = None):

    color_space = params['color_space']
    spatial_size = params['spatial_size']
    hist_bins = params['hist_bins']
    orient = params['orient']
    pix_per_cell = params['pix_per_cell']
    cell_per_block = params['cell_per_block']

    t1 = time.time()

    fe = FeaturesExtractor(color_space=color_space, 
                           spatial_size=spatial_size, 
                           hist_bins=hist_bins,
                           orient=orient,
                           pix_per_cell=pix_per_cell,
                           cell_per_block=cell_per_block)

    car_features = fe.extract_features(cars, process_pool=process_pool)
    notcar_features = fe.extract_features(notcars, process_pool=process_pool)

    extraction_time = round(time.time() - t1, 2)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rand_state)

    return X_train, X_test, y_train, y_test, extraction_time

def scale(X_train, X_test):
     # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test

def train(X_train, y_train, rand_state):
  
    print('Feature Vector Size:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC(random_state = rand_state)
    
    t1 = time.time()
    
    print('Training on {} images...'.format(len(X_train)))
    
    svc.fit(X_train, y_train)

    t2 = time.time()

    train_time = round(t2 - t1, 2)

    print('Training on {} images...DONE ({} s)'.format(len(X_train), train_time))

    return svc, train_time

def test(model, X_test, y_test, n_predict = 100):
    test_accuracy = round(model.score(X_test, y_test), 4)
    # Check the score of the SVC
    print('Test Accuracy: {}'.format(test_accuracy))

    # Check prediction time
    t1 = time.time()
    _ = model.predict(X_test[0:n_predict])
    t2 = time.time()

    pred_time = round(t2 - t1, 5)

    print('Prediction time for {} samples: {} s'.format(n_predict, pred_time))

    return test_accuracy, pred_time

def search_best(cars, notcars, process_pool = None):

    search_params = {
        'color_space': ['YUV', 'YCrCb'],
        'orient': [11, 12, 16],
        'pix_per_cell': [8, 16],
        'cell_per_block': [2],
        'spatial_size': [None, (8, 8), (16, 16), (32, 32)],
        'hist_bins': [None, 16, 32]
    }

    rand_state = 42

    max_acc = 0
    max_acc_params = None

    print('Searching best params...')

    t1 = time.time()
    i = 0
    
    for color_space in search_params['color_space']:
        for orient in search_params['orient']:
            for pix_per_cell in search_params['pix_per_cell']:
                for cell_per_block in search_params['cell_per_block']:
                    for spatial_size in search_params['spatial_size']:
                        for hist_bins in search_params['hist_bins']:
                            
                            params = {
                                'color_space': color_space,
                                'orient': orient,
                                'pix_per_cell': pix_per_cell,
                                'cell_per_block': cell_per_block,
                                'spatial_size': spatial_size,
                                'hist_bins': hist_bins
                            }

                            X_train, X_test, y_train, y_test, extraction_time = extract_train_test(cars, notcars, params, rand_state = rand_state, process_pool = process_pool)

                            X_train, X_test = scale(X_train, X_test)

                            model, train_time = train(X_train, y_train, rand_state)

                            test_accuracy, pred_time = test(model, X_test, y_test)

                            i += 1
                            
                            if test_accuracy > max_acc:
                                max_acc = test_accuracy
                                params['extraction_time'] = extraction_time
                                params['train_time'] = train_time
                                params['pred_time'] = pred_time
                                max_acc_params = params

    t2 = time.time()

    print('Searching best params...DONE ({} s, {} combinations)'.format(round(t2 - t1, 2), i))
    print('Accuracy: {}'.format(max_acc))
    print('Best Params: {}'.format(max_acc_params))

    save_model(model, os.path.join('models', 'best_model.p'))
    save_model(max_acc_params, os.path.join('models', 'best_model_params.p'))

if __name__ == '__main__':

    cars, notcars = load_dataset()

    pool_size = os.cpu_count()

    if pool_size > 0:
        process_pool = Pool(pool_size)
    else:
        process_pool = None

    print('Using {} cores'.format(1 if process_pool is None else pool_size))

    search = False

    if search:
        search_best(cars, notcars, process_pool = process_pool)
    else:

        params = {
            'color_space': 'YCrCb',
            'orient': 12,  # HOG orientations
            'pix_per_cell': 16, # HOG pixels per cell
            'cell_per_block': 2, # HOG cells per block
            'spatial_size': (32, 32), # Spatial binning dimensions
            'hist_bins': 32,    # Number of histogram bins
        }

        rand_state = 42

        X_train, X_test, y_train, y_test, extraction_time = extract_train_test(cars, notcars, params, rand_state = rand_state, process_pool = process_pool)

        X_train, X_test = scale(X_train, X_test)

        model, train_time = train(X_train, y_train, rand_state)

        test_accuracy, pred_time = test(model, X_test, y_test)

        model_file = time.strftime('model-%Y%m%d-%H%M%S.p')

        params['extraction_time'] = extraction_time
        params['train_time'] = train_time
        params['pred_time'] = pred_time

        save_model(model, model_file=os.path.join('models', model_file))
        save_model(params, model_file=os.path.join('models', model_file.split('.')[0] + '_params.p'))