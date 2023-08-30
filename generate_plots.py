from utils import styled_plotting

T_LIMIT = [7, 8]
SUB_ROI_X_LIMIT = [1.5,3.5]
SUB_ROI_Y_LIMIT = [1,3]

datasets = [
    "D:\\UCA\\03-Clustering Dynamics-WORKING\\STORM Datasets\\Control_BTX_smlm_dataset_NASTIC_20230830-154646\\result.tsv",
    "D:\\UCA\\03-Clustering Dynamics-WORKING\\STORM Datasets\\Control_BTX_smlm_dataset_SEGNASTIC_20230830-155623\\result.tsv",
    "D:\\UCA\\03-Clustering Dynamics-WORKING\\STORM Datasets\\Control_BTX_smlm_dataset.csv.full_prediction.csv_revalidated.csv",
]

for dataset in datasets:
    styled_plotting(
        dataset,
        t_limit=T_LIMIT,
        save_plot=True,
        with_clustering=True,
        plot_trajectories=True
    )

    styled_plotting(
        dataset,
        t_limit=T_LIMIT,
        x_limit=SUB_ROI_X_LIMIT,
        y_limit=SUB_ROI_Y_LIMIT,
        save_plot=True,
        with_clustering=True,
        plot_trajectories=True
    )
