"""
This is the GUI script for Dynamic Clustering with GNNs.

It is a code similar to https://github.com/tristanwallis/smlm_clustering/blob/main/nastic_gui.py
but rather simple.
"""
import os
from threading import Thread, Event

import PySimpleGUI as sg
import matplotlib.pyplot as plt
import colorama


def start():
    from . import CONSTANTS
    sg.theme('DARKGREY11')
    colorama.init()
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'{colorama.Fore.GREEN}=================================================={colorama.Style.RESET_ALL}')
    print(f'{colorama.Fore.GREEN}{CONSTANTS.METHOD_NAME} {CONSTANTS.LAST_UPDATE} initialising...{colorama.Style.RESET_ALL}')
    print(f'{colorama.Fore.GREEN}=================================================={colorama.Style.RESET_ALL}')
    print(F'Preparing GUI. It may takes a while... (Ignore Tensorflow Warnings)')
    #MODULE IMPORTS
    from time import sleep

    from tensorflow.errors import NotFoundError
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from .algorithm.ClusterDetector import ClusterDetector
    from .algorithm.LocalizationClassifier import LocalizationClassifier
    from .utils import predict_on_dataset
    
    sg.Print('Debug window open...', text_color='white', background_color='green', font='Courier 10')

    layout = [
        [
        sg.Text(
            text=f"{CONSTANTS.METHOD_NAME} (Last Update: {CONSTANTS.LAST_UPDATE})",
            font=('Arial Bold', 12),
            size=12,
            expand_x=True,
            justification='center')
        ],
        [
        sg.Text("Spatial Dimension"),
        sg.Listbox(['um', 'nm'], size=(10, 1), enable_events=False, key='-SPATIAL-DIMENSION-')
        ],
        [
        sg.Text("ROI Width"),
        sg.In(size=(25, 1), enable_events=False, key="-WIDTH-"),
        ],
        [
        sg.Text("ROI Height"),
        sg.In(size=(25, 1), enable_events=False, key="-HEIGHT-"),
        ],
        [
        sg.Text("GNN1 Model File (e.g., 'gnn1.h5')"),
        sg.In(size=(25, 1), enable_events=False, key="-GNN1-MODEL-"),
        sg.FileBrowse(),
        ],
        [
        sg.Text("GNN1 Threshold File (e.g., 'gnn1.txt')"),
        sg.In(size=(25, 1), enable_events=False, key="-GNN1-THRESHOLD-"),
        sg.FileBrowse(),
        ],
        [
        sg.Text("GNN2 Model File (e.g., 'gnn2.h5')"),
        sg.In(size=(25, 1), enable_events=False, key="-GNN2-MODEL-"),
        sg.FileBrowse(),
        ],
        [
        sg.Text("Localization Dataset"),
        sg.In(size=(25, 1), enable_events=True, key="-LOCALIZATION-DATASET-FILE-"),
        sg.FileBrowse(),
        sg.Checkbox('Is a TRXYT file?', False, key='-TRXYT-FILE-OPTION-')
        ],
        [
        sg.Text("Path to save analyzed dataset"),
        sg.In(size=(25, 1), enable_events=False, key="-PATH-TO-SAVE-RESULTS-"),
        sg.FolderBrowse(),
        ],
        [sg.Button('Analyze'), sg.Exit()],
    ]

    window = sg.Window(f"{CONSTANTS.METHOD_NAME} (Last Update: {CONSTANTS.LAST_UPDATE})", layout)

    i = 0

    while True:
        event, values = window.read()
        print(event, values)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Analyze':
            sg.Print('Checking ROI Dimensions...', text_color='white', background_color='blue', font='Courier 10')

            try:
                roi_height = float(values['-HEIGHT-'])
                roi_width = float(values['-WIDTH-'])

                if roi_height == 0 or roi_width == 0:
                    raise Exception()
            except Exception as e:
                sg.Print('Please, select correct ROI dimensions.', text_color='white', background_color='red', font='Courier 10')
                continue
            
            sg.Print('Loading GNN1 files...', text_color='white', background_color='blue', font='Courier 10')
            localization_classifier = LocalizationClassifier(roi_height, roi_width)

            try:
                localization_classifier.load_keras_model(values['-GNN1-MODEL-'], raise_error=True)
            except Exception as e:
                sg.Print(f'Please, select a valid .h5 file for GNN1.', text_color='white', background_color='red', font='Courier 10')
                continue        

            try:
                localization_classifier.load_threshold(values['-GNN1-THRESHOLD-'], raise_error=True)
            except Exception as e:
                sg.Print(f'Please, select a valid threshold file for GNN1.', text_color='white', background_color='red', font='Courier 10')
                continue

            sg.Print('Loading GNN2 files...', text_color='white', background_color='blue', font='Courier 10')
            cluster_detector = ClusterDetector(roi_height, roi_width)

            try:
                cluster_detector.load_keras_model(values['-GNN2-MODEL-'], raise_error=True)
            except Exception as e:
                sg.Print(f'Please, select a valid .h5 file for GNN2.', text_color='white', background_color='red', font='Courier 10')
                continue

            sg.Print('Loading localization dataset...', text_color='white', background_color='blue', font='Courier 10')

            try:
                if values['-TRXYT-FILE-OPTION-']:
                    dataset = pd.read_csv(values['-LOCALIZATION-DATASET-FILE-'], sep=' ', header=None)
                    dataset.columns= [
                        CONSTANTS.PARTICLE_ID_COLUMN_NAME,
                        CONSTANTS.X_POSITION_COLUMN_NAME,
                        CONSTANTS.Y_POSITION_COLUMN_NAME,
                        CONSTANTS.TIME_COLUMN_NAME
                    ]
                else:
                    dataset = pd.read_csv(values['-LOCALIZATION-DATASET-FILE-'])
            except Exception as e:
                sg.Print(f'Please, select a valid dataset. If a .trxyt file is not used, a .csv dataset with at least columns [x,y,t] is expected.', text_color='white', background_color='red', font='Courier 10')
                continue

            if not os.path.isdir(values['-PATH-TO-SAVE-RESULTS-']):
                sg.Print(f'Directory to save results does not exist...', text_color='white', background_color='red', font='Courier 10')
                continue
            
            sg.Print(f'All is OK', text_color='white', background_color='green', font='Courier 10')
            sg.Print(f'Analyzing...', text_color='white', background_color='green', font='Courier 10')

            try:
                dataset = predict_on_dataset(dataset, localization_classifier, cluster_detector)
                new_file_name = os.path.basename(values['-LOCALIZATION-DATASET-FILE-']).split('/')[-1]+".full_prediction.csv"
                dataset.to_csv(os.path.join(values['-PATH-TO-SAVE-RESULTS-'], new_file_name), index=False)
                sg.Print(f'FINISHED...', text_color='white', background_color='green', font='Courier 10')
            except Exception as e:
                sg.popup_error(e)

    window.close()
