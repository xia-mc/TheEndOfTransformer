import gc
import math
import typing

import numpy
import pandas
import torch.nn
from torch import Tensor
from tqdm import tqdm
from pandas import DataFrame
from torch.nn import MSELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Const import *
from Model import Model
from ModelDataSet import ModelDataSet

DATE_KEY = "日期"
TIME_KEY = "时刻"
FIXED_DATETIME_KEY = "datetime"

DATA_LENGTH = 50
PREDICT_LENGTH = 10
VERIFY_SPLIT = 0.8

BATCH_SIZE = 64
EPOCHS = 20
LOSS_RATE = 1e-3


def loadDataFrame(path: str) -> DataFrame:
    """读取数据和归一化"""
    dataframe: DataFrame = pandas.read_csv(path, sep=";", na_values="?")
    dataframe[FIXED_DATETIME_KEY] = pandas.to_datetime(dataframe[DATE_KEY] + " " + dataframe[TIME_KEY], dayfirst=True)
    dataframe.drop(columns=["日期", "时刻", "电压", "电流", "厨房用电功率", "洗衣间用电功率"], inplace=True)
    dataframe.set_index(FIXED_DATETIME_KEY, inplace=True)
    # 提取时间特征，GPT 说这能让模型更好的理解周期性
    hour = dataframe["hour"] = dataframe.index.hour
    minute = dataframe["minute"] = dataframe.index.minute
    dayOfWeek = dataframe["dayOfWeek"] = dataframe.index.dayofweek
    dataframe["hourSin"] = numpy.sin(2 * PI * hour / 24)
    dataframe["hourCos"] = numpy.cos(2 * PI * hour / 24)
    dataframe["minuteSin"] = numpy.sin(2 * PI * minute / 60)
    dataframe["minuteCos"] = numpy.cos(2 * PI * minute / 60)
    dataframe["dayOfWeekSin"] = numpy.sin(2 * PI * dayOfWeek / 7)
    dataframe["dayOfWeekCos"] = numpy.cos(2 * PI * dayOfWeek / 7)
    dataframe = dataframe.astype(DATA_TYPE)
    dataframe.interpolate(inplace=True)  # 孩子们我才知道panda可以自动补充缺失值👍
    return dataframe


def generateSeq(vector: Matrix, targetIndex: int) -> tuple[Array[Matrix], Array[Matrix]]:
    """
    我不知道这么写滑动窗口是不是内存占用会非常高🤔
    实测占用1GB左右，可以接受
    """
    inData: list[Matrix] = []
    outData: list[Matrix] = []
    for i in tqdm(range(len(vector) - DATA_LENGTH - PREDICT_LENGTH), leave=True, desc="Generate Sequences", ncols=100):
        dataBegin = i
        predictBegin = i + DATA_LENGTH
        inData.append(vector[dataBegin:dataBegin + DATA_LENGTH])
        outData.append(vector[predictBegin:predictBegin + PREDICT_LENGTH, targetIndex])
    return typing.cast(tuple[Array[Matrix], Array[Matrix]], (numpy.array(inData), numpy.array(outData)))


def train(model: Model, trainLoader: DataLoader, verifyLoader: DataLoader, epochs: int, lossRate: float):
    device: torch.device = torch.device("cpu")
    model.to(device)
    optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=lossRate)
    lossMethod: MSELoss = MSELoss()

    with tqdm(total=epochs * 2, leave=True, ncols=100) as progress:
        for i in range(epochs):
            progress.set_description(f"Train Model (Epoch: {i + 1})")

            # train
            model.train()
            progressTL = tqdm(total=len(trainLoader), ncols=100)
            trainLoss = 0.0
            for inBatch, outBatch in trainLoader:
                # check
                assert not torch.isnan(inBatch).any()
                assert not torch.isinf(inBatch).any()
                assert not torch.isnan(outBatch).any()
                assert not torch.isinf(outBatch).any()

                progressTL.set_description(f"Train Loss (Loss: {trainLoss:.4f})")

                inBatch, outBatch = inBatch.to(device), outBatch.to(device)
                pred = model(inBatch)
                loss: Tensor = lossMethod(pred, outBatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trainLoss += loss.item()

                progressTL.update()
            progress.update()

            # verify
            model.eval()
            progressVL = tqdm(total=len(trainLoader), ncols=100)
            verifyLoss = 0.0
            with torch.no_grad():
                for inBatch, outBatch in verifyLoader:
                    progressVL.set_description(f"Verify Loss (Loss: {verifyLoss:.4f})")

                    inBatch, outBatch = inBatch.to(device), outBatch.to(device)
                    pred = model(inBatch)
                    loss: Tensor = lossMethod(pred, outBatch)
                    verifyLoss += loss.item()
            verifyLoss /= len(verifyLoader)

            progressTL.close()
            progressVL.close()
            progress.update()
            tqdm.write(f"Epoch {i + 1}/{epochs}, Train Loss: {trainLoss:.4f}, Verify Loss: {verifyLoss:.4f}.")


def predict(model: Model, inputSeq: Matrix) -> Vector:
    model.eval()
    device: torch.device = next(model.parameters()).device
    with torch.no_grad():
        input_seq = torch.tensor(inputSeq, dtype=DTYPE).unsqueeze(0).to(device)
        pred: Tensor = model(input_seq)
        return typing.cast(Vector, pred.cpu().numpy().flatten())


def main():
    tqdm.write("初始化数据...")
    dataframe = loadDataFrame("../household_power_consumption.txt")
    targetIndex: int = dataframe.columns.get_loc("总功率")  # 预测的目标列
    matrix: Matrix = typing.cast(Matrix, dataframe.values)
    assert not numpy.isnan(matrix).any()
    assert not numpy.isinf(matrix).any()
    del dataframe  # 节省内存

    gc.collect()
    inData, outData = generateSeq(matrix, targetIndex)  # 用于训练
    del matrix

    # 准备训练数据
    tqdm.write("准备训练数据...")
    gc.collect()
    assert len(inData) == len(outData)
    split = int(len(inData) * VERIFY_SPLIT)
    trainData = ModelDataSet(inData[:split], outData[:split])
    verifyData = ModelDataSet(inData[split:], outData[split:])
    trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
    verifyLoader = DataLoader(verifyData, batch_size=BATCH_SIZE)
    del outData

    # 准备模型对象
    tqdm.write("准备模型...")
    model = Model(featureCount=len(inData[0][0]), predictLength=PREDICT_LENGTH)

    # 训练
    tqdm.write("开始训练...")
    gc.collect()
    train(model, trainLoader, verifyLoader, EPOCHS, LOSS_RATE)
    del trainData, verifyData, trainLoader, verifyLoader

    # 推理
    tqdm.write("开始推理...")
    gc.collect()
    print(predict(model, inData[-1]))


# 能训练了，但速度只有4it/s，这得训练一年。
if __name__ == '__main__':
    main()
