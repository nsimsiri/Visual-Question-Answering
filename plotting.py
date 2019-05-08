import os, sys
import matplotlib.pyplot as plt

from bokeh.plotting import figure, output_file, show, save

LOSS_OVER_N_EPOCHS_DICT_KEYS = ["train_loss", "test_loss"]
SCORE_KEY_MAP = {'precision': 0, 'recall': 1, 'f1': 2}

def validate_loss_over_n_dict_keys(loss_over_n_epochs: dict):
    assert all([key in LOSS_OVER_N_EPOCHS_DICT_KEYS for key in loss_over_n_epochs.keys()])


def plot_loss_over_n_epochs(loss_over_n_epochs: dict,
                            title=None,
                            file_path=None,
                            hard_key = None,
                            fig_size: tuple = (10, 6)):

#     validate_loss_over_n_dict_keys(loss_over_n_epochs)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    if title:
        ax.set_title(title)
        
    if hard_key is None:
        hard_key = next(iter(loss_over_n_epochs.keys()))
    n_epochs = len(loss_over_n_epochs[hard_key])

    for key in loss_over_n_epochs:
        # If nothing to plot just skip that split.
        if len(loss_over_n_epochs[key]) == 0:
            continue
        ax.plot(range(1, n_epochs + 1), loss_over_n_epochs[key], label=key)

    plt.legend()

    if file_path:
        file_path = os.path.join(PLOTTING_ROOT, file_path)
        print("File Path: ", file_path)
        fig.savefig(file_path)

    plt.show()


def plot_score_over_n_epochs(scores_over_n_epochs: dict,
                             score_type='f1',
                             title=None,
                             file_path=None,
                             fig_size: tuple = (10, 6)):
    assert score_type in SCORE_KEY_MAP.keys(), "Invalid Score type."

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('{} Score'.format(score_type))
    if title:
        ax.set_title(title)

    f1_score_key = SCORE_KEY_MAP[score_type]

    first_key = next(iter(scores_over_n_epochs.keys()))
    n_epochs = len(scores_over_n_epochs[first_key])

    for key in scores_over_n_epochs:
        f1_score = []
        if len(scores_over_n_epochs[key]) == 0:
            continue
        for epoch in range(n_epochs):
            f1_score.append(scores_over_n_epochs[key][epoch][f1_score_key])

        ax.plot(range(1, n_epochs + 1), f1_score, label=key)

    plt.legend()
    plt.show()

def get_empty_stat_over_n_epoch_dictionaries():
    loss_over_epochs = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    scores_over_epochs = {
        "train_scores": [],
        "val_scores": [],
        "test_scores": [],
        "overall_scores": []
    }

    return loss_over_epochs, scores_over_epochs


def plot_line_chart_using_bokeh(x_axis_data: list, y_axis_data: list, colors: list,
                                title: str, output_file_name: str,
                                plot_height=350, plot_width=800,
                                line_alpha=0.5, line_width=1,
                                x_label='Time', y_label='Value',
                                show_fig=True):
    assert len(x_axis_data) == len(y_axis_data) and len(x_axis_data) == len(
        y_axis_data), "Length miss-match for x-axis or y-axis data."

    p = figure(x_axis_type="datetime", title=title, plot_height=plot_height, plot_width=plot_width)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_alpha = 0.5
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.multi_line(x_axis_data, y_axis_data, line_color=colors, line_width=line_width, line_alpha=line_alpha)
    output_file(output_file_name)
    if show_fig:
        show(p)