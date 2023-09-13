import logging
import pandas as pd
import json
import os
import dill

p = os.path.expanduser(r'~/airflow_hw')
path = os.environ.get('PROJECT_PATH', f'{p}')



def model_func():
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as f:
        model = dill.load(f)
    return model


def files(name):
    with open(f'{path}/data/test/{name}', 'rb') as file:
        data = json.load(file)
    return data


def pred():
    model = model_func()
    test_files = os.listdir(f'{path}/data/test')
    name = []
    predict = []
    for file in test_files:
        df = pd.DataFrame(files(file), index=[0])
        pred = model.predict(df)
        name.append(file.split('.')[0])
        predict.append(pred[0])
    return name, predict


def predict():
    name, predict = pred()
    dictionary = {'car_id': name, ' Predict': predict}
    df = pd.DataFrame(dictionary)
    df.to_csv(f'{path}/data/predictions/data.csv', index=False)
    logging.info(f'Predict is saved as {df}')


if __name__ == '__main__':
    predict()
