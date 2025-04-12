import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from main import loadDataFrame

import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf


STEP = 1
LAGS = int(200000 / STEP)


def render(data):
    data = data[10000:][:1440]
    df = pd.DataFrame({'x': np.arange(len(data)), 'y': data})

    canvas = ds.Canvas(plot_width=len(data), plot_height=1000)
    agg = canvas.line(df, 'x', 'y')
    img = tf.shade(agg)

    img.to_pil().show()


def main():
    print("Loading")
    dataframe = loadDataFrame("../household_power_consumption.txt")
    targetIndex: int = dataframe.columns.get_loc("总功率")
    matrix = dataframe.values
    data = matrix[:, targetIndex][::STEP]
    del dataframe
    del matrix

    print("Rendering")
    render(data)

    print("Analyzing")
    plot_acf(data, lags=LAGS, fft=True)
    plt.title("ACF")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    plot_pacf(data, lags=LAGS)
    plt.title("PACF")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
