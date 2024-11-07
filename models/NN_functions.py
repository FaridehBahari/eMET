import os
import configparser
import pandas as pd
import numpy as np
from readFtrs_Rspns import set_gpu_memory_limit
import h5py
import time  
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense, Dropout, LeakyReLU, BatchNormalization, PReLU, Input, Add
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.losses import Poisson
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from readFtrs_Rspns import save_preds_tsv
import platform
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from ann_visualizer.visualize import ann_viz;
from sklearn.model_selection import train_test_split


if platform.system() == 'Windows':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
def build_NN_params(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # Load the hyperparameters from the config file
    param = {
        'activation1': config.get('architecture', 'activation1'),
        'activation_out': config.get('architecture', 'activation_out'),
        'res_connection': config.getboolean('architecture', 'res_connection'),
        'dropout': config.getfloat('architecture', 'dropout'),
        'learning_rate': config.getfloat('training', 'learning_rate'),
        'loss': config.get('training', 'loss'),
        'optimizer': config.get('training', 'optimizer'),
        'metrics': [x.strip() for x in config.get('training', 'metrics').split(',')],
        'epochs': config.getint('training', 'epochs'),
        'batch_size': config.getint('training', 'batch_size'),
        'save_interval': config.getint('training', 'save_interval'),
        'response': config.get('main', 'response'),
        'architecture': [int(x.strip()) for x in config.get('architecture', 'architecture').split(',')]
    }
    return param


   
def build_model_architecture(hyperparams, n_ftrs):
    
    optimizer = hyperparams['optimizer']
    learning_rate = hyperparams['learning_rate']
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    inputs = Input(shape=(n_ftrs,))
    x = inputs  # Initialize x with inputs
    
    # x = Dropout(0.2)(x)
    
    if hyperparams['architecture'] == [0]:
        x = Dense(1, activation=hyperparams['activation_out'])(x)
    else:
        for i, units in enumerate(hyperparams['architecture']):
            tmp_input = x  # Store the current output for the residual connection

            x = Dense(units)(x)
            x = BatchNormalization()(x)                

            if hyperparams['activation1'] == 'leaky_relu':
                x = LeakyReLU(alpha=0.2)(x)
            elif hyperparams['activation1'] == 'PReLU':
                x = PReLU()(x)
            else:
                x = Activation(hyperparams['activation1'])(x)

            if hyperparams['dropout'] != 0:
                x = Dropout(hyperparams['dropout'])(x)

            if hyperparams['res_connection']:
                
                if(tmp_input.shape[1] == units):
                    residual = tmp_input
                else:
                    residual = layers.Dense(units, activation=None)(tmp_input) 
                
                x = Add()([x, residual])

        x = Dense(1, activation=hyperparams['activation_out'])(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=hyperparams['loss'], optimizer=opt)

    return model


def get_last_batch_number(directory_path):
   
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter for files with names like "batch_N_model.h5"
    batch_files = [file for file in files if file.startswith("batch_") and file.endswith("_model.h5")]

    # Extract and store the batch numbers
    batch_numbers = []
    for file in batch_files:
        try:
            batch_number = int(file.split("_")[1])
            batch_numbers.append(batch_number)
        except ValueError:
            pass  # Skip files that don't match the expected naming pattern

    # Find the maximum batch number
    if batch_numbers:
        last_batch_number = max(batch_numbers)
        
    else:
        last_batch_number = 0
        
    return last_batch_number



def run_NN(X_train, Y_train, NN_hyperparams):
    
    path = NN_hyperparams['path_save']

    # Split the path by '/'
    parts = path.split('/')
    model_name = parts[-3]
    
    # Extract the desired portion
    if len(parts) >= 2:
        extracted_path = '/'.join(parts[:-2]) + '/'
    else:
        extracted_path = ''

    
    path_to_save = f'{extracted_path}model_plots'
    os.makedirs(path_to_save, exist_ok= True)
    
    
    if NN_hyperparams['response'] == "rate":
        Y_tr = (Y_train['nMut']) / (Y_train['length'])
    elif NN_hyperparams['response'] == "count":
        Y_tr = Y_train['nMut']
    else:
        raise ValueError("error")  # Use "raise" to raise an exception
    
    X_train, X_val, Y_tr, Y_val = train_test_split(X_train, Y_tr, 
                                                    test_size=0.12, 
                                                    random_state= 42)
    n_ftrs = int(X_train.shape[1])

    # Define a callback to save the model after each epoch
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(NN_hyperparams['path_save'], 'batch_{epoch}_model.h5'),
        save_best_only=False,  # Set to True if you want to save only the best model
        save_weights_only=False,  # Set to True if you want to save only the model weights
        monitor='val_loss',  # You can choose a metric to monitor, e.g., 'val_loss'
        mode='auto',  # Auto mode for monitoring
        save_freq=NN_hyperparams['save_interval']
    )

    # Build initial model
    model = build_model_architecture(NN_hyperparams, n_ftrs)
    print(model.summary())
    
    # ann_viz(model, filename= f'{path_to_save}/model_architecture',
    #         title='My first neural network')
    
    # Add the checkpoint callback to the list of callbacks
    callbacks_list = [checkpoint]
    
    history = model.fit(X_train, Y_tr,  validation_data=(X_val, Y_val), 
              epochs=NN_hyperparams['epochs'],
              batch_size=NN_hyperparams['batch_size'], verbose=1,
              callbacks=callbacks_list)  # Pass the callbacks list to the fit method
   
    # model.fit(X_train, Y_tr, epochs=NN_hyperparams['epochs'],
    #           batch_size=NN_hyperparams['batch_size'], verbose=1,
    #           callbacks=callbacks_list) 
   
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    
    #plt.show()
    plt.savefig(f'{path_to_save}/{model_name}_train_val_loss_plot')
    plt.close()
    
    model_data = {'model': model,
                  'N': Y_train.N[0],
                  'response_type': NN_hyperparams['response']}

    return model_data



   

def predict_NN(model, X_test, length_elems):
    
    response = model['response_type']
    prediction_df = None
    M = model['model']
    prediction = M.predict(X_test, verbose = 1)
    N = model['N']
    
    if response == "rate":
        prediction = prediction/ N  
    elif response == "count":
        log_offset = N * length_elems
        prediction = prediction[:,0] /np.exp(log_offset)
    else:
        ValueError("error")
    
    prediction_df = pd.DataFrame({'predRate': prediction.ravel()}, 
                                 index=X_test.index)
    return prediction_df

def nn_model_info(save_name, *args):
    NN_hyperparams = build_NN_params(args[0])
    model_dict = {"save_name" : save_name,
                  "Args" : NN_hyperparams,
                  "run_func": run_NN,
                  "predict_func": predict_NN,
                  "save_func": save_nn,
                  "check_file_func": check_file_nn
                  }
    
    return model_dict


def save_nn(fitted_Model, path_save, save_name, iteration = None, save_model = True): 
    
    save_preds_tsv(fitted_Model, path_save, save_name, iteration)
    
    if save_model:
        M = fitted_Model.model['model']
        # Save the model 
        if iteration is not None:
            save_path_model = f'{path_save}/{save_name}/rep_train_test/{save_name}_model_{iteration}.h5'
        else:
            save_path_model = f'{path_save}/{save_name}/{save_name}_model.h5'
            
        M.save(save_path_model)
        
        
def check_file_nn(path_save, save_name):
    file_name = f'{path_save}/{save_name}/{save_name}_model.h5'
    return file_name