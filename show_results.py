import os
import argparse
import numpy as np

result_directory = 'results'
result_list = os.listdir(result_directory)
result_list = sorted(result_list)

def show_metrics():
    print('')
    for result in result_list:
        try: data = np.load(f"results/{result}/metrics.npy")
        except: continue
        print(f"Metrics of {result}:")
        print(f"MSE: {data[1]:13.8f} | MAE: {data[0]:11.8f}\n")
    return

def show_pred():
    for result in result_list:
        try:
            predictions = np.load(f"results/{result}/predictions.npy")
        except:
            continue
        print(f"Predictions & truths of {result}:")
        for i in range(len(predictions[-1, :, 0])):
            print(f"{float(predictions[-1][i])} | ", end='')
        print('\n')
    return

def show_pred_truths():
    for result in result_list:
        try:
            predictions = np.load(f"results/{result}/predictions.npy")
            truths      = np.load(f"results/{result}/truths.npy")
        except:
            continue
        print(f"Predictions & truths of {result}:")
        for i in range(len(predictions[-1, :, 0])):
            print(f"{float(predictions[-1][i])}, {float(truths[-1][i])} | ", end='')
        print('\n')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--act', type=str, default='metrics')
    args = parser.parse_args()

    if args.act == 'metrics': show_metrics()
    elif args.act == 'pred': show_pred()
    elif args.act == 'pred_truths': show_pred_truths()
    else: print('Wrong args.')