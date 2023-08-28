"""
This is the GUI script for Dynamic Clustering with GNNs.

It is a code similar to https://github.com/tristanwallis/smlm_clustering/blob/main/nastic_gui.py
but rather simple.
"""
import PySimpleGUI as sg
import colorama
import os

#CONSTANTS
METHOD_NAME = 'Dynamic Clustering with GNNs'
LAST_UPDATE = '2023-08-28'

if __name__ == '__main__':
    sg.theme('DARKGREY11')
    colorama.init()
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'{colorama.Fore.GREEN}=================================================={colorama.Style.RESET_ALL}')
    print(f'{colorama.Fore.GREEN}{METHOD_NAME} {LAST_UPDATE} initialising...{colorama.Style.RESET_ALL}')
    print(f'{colorama.Fore.GREEN}=================================================={colorama.Style.RESET_ALL}')

    #MODULE IMPORTS
    from time import sleep

    from tensorflow.errors import NotFoundError

    from ClusterDetector import ClusterDetector
    from LocalizationClassifier import LocalizationClassifier

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
            window['-STATUS-'].update('Loading GNN1 files...')
            localization_classifier = LocalizationClassifier(10,10)

            try:
                localization_classifier.load_keras_model(values["-GNN1-MODEL-"], raise_error=True)
            except NotFoundError as exception:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(f'Please, select a model file for GNN1.')
                continue
            except OSError as exception:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(f'Please, select a valid .h5 file for GNN1.')
                continue

            try:
                localization_classifier.load_threshold(values["-GNN1-THRESHOLD-"], raise_error=True)
            except Exception as e:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(e)
                continue

            window['-STATUS-'].update('Loading GNN2 files...')
            cluster_detector = ClusterDetector(10,10)

            try:
                cluster_detector.load_keras_model(values["-GNN2-MODEL-"], raise_error=True)
            except NotFoundError as exception:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(f'Please, select a model file for GNN2.')
                continue
            except OSError as exception:
                window['-STATUS-'].update('Something failed...')
                sg.popup_error(f'Please, select a valid .h5 file for GNN2.')
                continue

            window['-STATUS-'].update('Loading localization dataset...')

    window.close()
