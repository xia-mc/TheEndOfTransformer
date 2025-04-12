import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from main import loadDataFrame


LAGS = 100


def main():
    print("Loading")
    dataframe = loadDataFrame("../household_power_consumption.txt")
    targetIndex: int = dataframe.columns.get_loc("总功率")
    matrix = dataframe.values
    del dataframe

    print("Analyzing")
    plot_acf(matrix[:, targetIndex], lags=LAGS, fft=True)
    plt.title("ACF")
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    plot_pacf(matrix[:, targetIndex], lags=LAGS)
    plt.title("PACF")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
