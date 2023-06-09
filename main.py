import sys
from pathlib import Path
import os

from model import PaddedTransformer
from importlib import import_module
import configparser

import plotly.graph_objects as go

def plot(series, name, output_path):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(series))), y=series,
                             mode='lines+markers',
                             name=name))

    fig.update_layout(title=name+ ' vs pad', width=800, height=480)
    fig.update_xaxes(range=(0, 514))
    fig.update_yaxes(range=(min(series)*0.9, max(series)*1.1))

    fig.write_html(output_path / f'results_{name}.html')


def main(config_file):
    config = configparser.ConfigParser()


    config.read(config_file)
    task_type = config['TASK']['taskType']

    model_class = 'AutoModelFor' + task_type
    model_class = getattr(import_module('transformers', model_class), model_class)
    model_name = config["MODEL"]['modelName']

    transformer = PaddedTransformer(model_name, model_class)

    dataset_name = config['DATASET']['datasetName']
    dataset_class = getattr(import_module('dataset', dataset_name), dataset_name)
    dataset = dataset_class()

    task_class = getattr(import_module('task', task_type), task_type)
    task = task_class(transformer, dataset)

    res = task.inference()


    output_path = Path(os.path.join('./results', config['TASK']['experimentName']))

    if not output_path.exists():
        output_path.mkdir(parents=True)

    res.to_csv(output_path / 'results.csv')

    for col in res.columns:
        plot(res[col], col, output_path)




if __name__ == "__main__":
    main(sys.argv[1])
