import matplotlib.pyplot as plt
import numpy as np

def plot_figure(path, historty, args):

    print(f"Saving loss historty figure... ", end='')
    
    lr = args.learning_rate
    lradj = args.lradj

    epochs_length = range(1, len(historty['scaled']['train'])+1)

    x_limit = int(len(historty['scaled']['train']) / 10) * 10 + 10.01
    # y1_limit = min(max(historty['scaled']['test']) + 300.01, 3000.01)
    y1_limit = 200.01

    x_ticks_major = np.arange(0, x_limit, 10)
    x_ticks_major[0] = 1
    x_ticks_minor = np.arange(0, x_limit, 1)
    x_ticks_minor[0] = 1
    y_ticks_1_major = np.arange(0, y1_limit, 100)
    y_ticks_1_minor = np.arange(0, y1_limit, 20)

    fig, axs = plt.subplots(2)
    fig.set_size_inches(8, 10)
    fig.suptitle('Learning Rates & Losses Comparition')

    axs[0].plot(epochs_length, historty['learning_rate'], 'b-', linewidth = 1, label=f"lr: {lr} / lradj: {lradj}")
    axs[0].legend()
    axs[0].set(ylabel='Learning Rate', title='Learning Rates')
    axs[0].set_xlim(1, x_limit)
    axs[0].set_xticks(x_ticks_major)
    axs[0].set_xticks(x_ticks_minor, minor=True)
    axs[0].grid()
    axs[0].grid(which='minor', alpha=0.3)

    axs[1].plot(epochs_length, historty['scaled']['train'], 'b-', linewidth = 1, label='Train')
    axs[1].plot(epochs_length, historty['scaled']['vali'], 'r-', linewidth = 1, label='Vali')
    axs[1].plot(epochs_length, historty['scaled']['test'], 'y-', linewidth = 1, label='Test')
    axs[1].legend()
    axs[1].set(xlabel='Epochs', ylabel='Loss', title='Losses')
    axs[1].set_xlim(1, x_limit)
    axs[1].set_xticks(x_ticks_major)
    axs[1].set_xticks(x_ticks_minor, minor=True)
    axs[1].set_ylim(0, y1_limit)
    axs[1].set_yticks(y_ticks_1_major)
    axs[1].set_yticks(y_ticks_1_minor, minor=True)
    axs[1].grid()
    axs[1].grid(which='minor', alpha=0.3)

    plt.savefig(f"{path}/historty.png", dpi=200)
    # plt.show()

    print("Done.\n")
    return