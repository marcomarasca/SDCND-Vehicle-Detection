import numpy as np
import cv2
import glob
import time
import argparse
import signal

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from features_extractor import FeaturesExtractor
from data_loader import *
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

def extract_features(cars, notcars, params, process_pool = None):

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

    return X, y, extraction_time

def scale(X_train, X_test):
     # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_train, X_test, X_scaler

from sklearn.model_selection import GridSearchCV

def train(X_train, y_train, rand_state):
  
    print('Feature Vector Size:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC(random_state = rand_state, dual=False, C=10**-2)

    t1 = time.time()
    
    print('Training on {} images...'.format(len(X_train)))
    
    svc.fit(X_train, y_train)

    t2 = time.time()

    train_time = round(t2 - t1, 2)

    print('Training on {} images...DONE ({} s)'.format(len(X_train), train_time))

    return svc, train_time

def test(model, X_test, y_test, n_predict = 100):
    accuracy = round(model.score(X_test, y_test), 4)
    # Check the score of the SVC
    print('Test Accuracy: {}'.format(accuracy))

    # Check prediction time
    t1 = time.time()
    _ = model.predict(X_test[0:n_predict])
    t2 = time.time()

    pred_time = round(t2 - t1, 5)

    print('Prediction time for {} samples: {} s'.format(n_predict, pred_time))

    return accuracy, pred_time

def train_and_test(cars, notcars, params, rand_state = None, process_pool = None):

    # Extract features
    X, y, ext_time = extract_features(cars, notcars, params, process_pool = process_pool)

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rand_state)

    # Scale data
    X_train, X_test, scaler = scale(X_train, X_test)

    # Train the Linear SVC
    model, train_time = train(X_train, y_train, rand_state = rand_state)

    # Test the predictions
    accuracy, pred_time = test(model, X_test, y_test)

    params['window'] = cv2.imread(cars[0]).shape[0]
    params['model'] = model
    params['scaler'] = scaler
    params['accuracy'] = accuracy
    params['extraction_time'] = ext_time
    params['train_time'] = train_time
    params['pred_time'] = pred_time

    return params

def parameters_search(cars, notcars, params_file = 'search_params.json', rand_state = None, process_pool = None):

    search_params = load_search_params(file = params_file)

    max_acc = 0
    max_acc_params = None

    print('Searching best parameters using space {}...'.format(search_params))

    t1 = time.time()
    experiments = 0
    
    for color_space in search_params['color_space']:
        for orient in search_params['orient']:
            for pix_per_cell in search_params['pix_per_cell']:
                for cell_per_block in search_params['cell_per_block']:
                    for spatial_size in search_params['spatial_size']:
                        for hist_bins in search_params['hist_bins']:

                            experiments += 1
                            
                            params = {
                                'color_space': color_space,
                                'orient': orient,
                                'pix_per_cell': pix_per_cell,
                                'cell_per_block': cell_per_block,
                                'spatial_size': spatial_size,
                                'hist_bins': hist_bins
                            }

                            model_params = train_and_test(cars, notcars, params, rand_state=rand_state, process_pool=process_pool)
                            
                            if model_params['accuracy'] > max_acc:
                                max_acc = model_params['accuracy']
                                max_acc_params = model_params

    t2 = time.time()

    print('Searching best params...DONE ({} s, {} experiments)'.format(round(t2 - t1, 2), experiments))
    print('Accuracy: {}'.format(max_acc))
    print('Best Params: {}'.format(max_acc_params))

    return model_params

def worker_init():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SVC Training')

    parser.add_argument(
        '--dir',
        type=str,
        default='data',
        help='Images folder'
    )
    parser.add_argument(
        '--rand_state',
        type=int,
        default=None,
        help='Random seed used for shuffling'
    )
    parser.add_argument(
        '--color_space',
        type=str,
        default='YCrCb',
        help='Color space conversion'
    )
    parser.add_argument(
        '--spatial_size',
        type=int,
        default=16,
        help='Spatial binning dimension, can be None to disable'
    )
    parser.add_argument(
        '--hist_bins',
        type=int,
        default=32,
        help='Number color histograms bins, can be None to disable'
    )
    parser.add_argument(
        '--orient',
        type=int,
        default=11,
        help='Number of HOG orientations'
    )
    parser.add_argument(
        '--pix_per_cell',
        type=int,
        default=16,
        help='Number of HOG pixels per cell'
    )
    parser.add_argument(
        '--cell_per_block',
        type=int,
        default=2,
        help='Number of HOG cells per block'
    )
    parser.add_argument(
        '--search',
        type=str,
        default=None,
        help='Performs a parameters search, the value is a json file with parameters space'
    )
    
    parser.add_argument('--disable-parallel', dest='parallel', action='store_false', help='Disable parallel processing (may decrease feature extraction speed)')
    parser.set_defaults(parallel=True)

    args = parser.parse_args()

    cars, notcars = load_dataset(cars_folder=os.path.join(args.dir, 'vehicles'), notcars_folder=os.path.join(args.dir, 'non-vehicles'))

    pool_size = os.cpu_count()

    if args.parallel is False or pool_size < 2:
        process_pool = None
    else:
        process_pool = Pool(pool_size, initializer = worker_init)

    print('Using {} cores'.format(1 if process_pool is None else pool_size))

    try:

        model_file = time.strftime('model-%Y%m%d-%H%M%S.p')

        if args.search is None:
            
            params = {
                'color_space': args.color_space,        # Color space
                'orient': args.orient,                  # HOG orientations
                'pix_per_cell': args.pix_per_cell,      # HOG pixels per cell
                'cell_per_block': args.cell_per_block,  # HOG cells per block
                'spatial_size': args.spatial_size,      # Spatial binning dimensions
                'hist_bins': args.hist_bins,            # Number of histogram bins
            }

            model_params = train_and_test(cars, notcars, params, rand_state=args.rand_state, process_pool=process_pool)
        else:
            model_params = parameters_search(cars, notcars, params_file=args.search, rand_state=args.rand_state, process_pool=process_pool)
     
        save_model(model_params, model_file=os.path.join('models', model_file))

    except Exception as e:
        if process_pool is not None:
            process_pool.terminate()
        raise e
