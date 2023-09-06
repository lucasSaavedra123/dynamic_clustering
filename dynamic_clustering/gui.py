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

#CONSTANTS
METHOD_NAME = 'Dynamic Clustering with GNNs'
LAST_UPDATE = '2023-08-29'


if __name__ == '__main__':
    sg.theme('DARKGREY11')
    colorama.init()
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'{colorama.Fore.GREEN}=================================================={colorama.Style.RESET_ALL}')
    print(f'{colorama.Fore.GREEN}{METHOD_NAME} {LAST_UPDATE} initialising...{colorama.Style.RESET_ALL}')
    print(f'{colorama.Fore.GREEN}=================================================={colorama.Style.RESET_ALL}')
    print(F'Preparing GUI. It may takes a while... (Ignore Tensorflow Warnings)')
    #MODULE IMPORTS
    from time import sleep

    from tensorflow.errors import NotFoundError
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from ClusterDetector import ClusterDetector
    from LocalizationClassifier import LocalizationClassifier
    from dynamic_clustering.CONSTANTS import *

    #CONSTANTS
    METHOD_NAME = 'Dynamic Clustering with GNNs'
    LAST_UPDATE = '2023-08-28'

    layout = [
        [
        sg.Text(
            text=f"{METHOD_NAME} (Last Update: {LAST_UPDATE})",
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
        [sg.ProgressBar(100, orientation='h', expand_x=True, size=(20, 20),  key='-PBAR-')],
        [sg.Button('Analyze'), sg.Exit()],
        [sg.Text('Current Status:', font=('Arial', 9)), sg.StatusBar('Click on "Analyze" to start...', key='-STATUS-', font=('Arial', 9))]
    ]

    window = sg.Window(f"{METHOD_NAME} (Last Update: {LAST_UPDATE})", layout)

    i = 0

    while True:
        event, values = window.read()
        print(event, values)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        elif event == 'Analyze':
            window['-STATUS-'].update('Checking ROI Dimensions...')

            try:
                roi_height = float(values['-HEIGHT-'])
                roi_width = float(values['-WIDTH-'])

                if roi_height == 0 or roi_width == 0:
                    raise Exception()
            except Exception as e:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(f'Please, select correct ROI dimensions.')   
                continue

            window['-STATUS-'].update('Loading GNN1 files...')
            localization_classifier = LocalizationClassifier(roi_height, roi_width)

            try:
                localization_classifier.load_keras_model(values['-GNN1-MODEL-'], raise_error=True)
            except Exception as e:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(f'Please, select a valid .h5 file for GNN1.')
                continue        

            try:
                localization_classifier.load_threshold(values['-GNN1-THRESHOLD-'], raise_error=True)
            except Exception as e:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(e)
                continue

            window['-STATUS-'].update('Loading GNN2 files...')
            cluster_detector = ClusterDetector(roi_height, roi_width)

            try:
                cluster_detector.load_keras_model(values['-GNN2-MODEL-'], raise_error=True)
            except Exception as e:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(f'Please, select a valid .h5 file for GNN2.')
                continue

            window['-STATUS-'].update('Loading localization dataset...')
            try:
                if values['-TRXYT-FILE-OPTION-']:
                    dataset = pd.read_csv(values['-LOCALIZATION-DATASET-FILE-'], sep=' ', header=None)
                    dataset.columns= [PARTICLE_ID_COLUMN_NAME, X_POSITION_COLUMN_NAME, Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]
                else:
                    dataset = pd.read_csv(values['-LOCALIZATION-DATASET-FILE-'])
            except Exception as e:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(f'Please, select a valid dataset. If a .trxyt file is not used, a .csv dataset with at least columns [x,y,t] is expected.')                

    window.close()
