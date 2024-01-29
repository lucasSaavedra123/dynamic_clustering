from dynamic_clustering.utils import styled_plotting

T_LIMIT = [15, 20]
SUB_ROI_X_LIMIT = [7,9]
SUB_ROI_Y_LIMIT = [2,4]

datasets = [
    "D:\\UCA\\Projects\\00-Clustering Dynamics-FINISHING\\STORM Datasets\\Control_BTX_smlm_dataset_NASTIC_20230830-154646\\result.tsv",
    "D:\\UCA\\Projects\\00-Clustering Dynamics-FINISHING\\STORM Datasets\\Control_BTX_smlm_dataset_SEGNASTIC_20230830-155623\\result.tsv",
    "D:\\UCA\\Projects\\00-Clustering Dynamics-FINISHING\\STORM Datasets\\Control_BTX_smlm_dataset.csv.full_prediction.csv_revalidated.csv",
]

for dataset in datasets:
    styled_plotting(
        dataset,
        t_limit=T_LIMIT,
        save_plot=True,
        with_clustering=True,
        plot_trajectories=True,
        background_color='w',
        trajectory_color=None,#'black',
        colorbar_color='black',
        trajectory_alpha=1
    )

    styled_plotting(
        dataset,
        t_limit=T_LIMIT,
        x_limit=SUB_ROI_X_LIMIT,
        y_limit=SUB_ROI_Y_LIMIT,
        save_plot=True,
        with_clustering=True,
        plot_trajectories=True,
        background_color='w',
        trajectory_color=None,#'black',
        colorbar_color='black',
        trajectory_alpha=1
    )
