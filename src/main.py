import gc
import os
import time
import typing
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy.version
import pandas
import torch.nn
from pandas import DataFrame
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from Const import *
from Model import Model
from ModelDataSet import ModelDataSet
from Scaler import Scaler, Scaler3D

DATA_LENGTH = 1440
PREDICT_LENGTH = 10
VERIFY_SPLIT = 0.8

DATA_KEEP = 0.01
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
OUTPUT_DIR = (Path(__file__).parent.parent if Path(__file__).parent.name == "src" else Path(__file__).parent) / "out"
MODEL_OUTPUT = OUTPUT_DIR / "model.pth"
IN_SCALER_OUTPUT = OUTPUT_DIR / "in_scaler.pkl"
OUT_SCALER_OUTPUT = OUTPUT_DIR / "out_scaler.pkl"

TRAIN_MODE = False
TQDM_DISABLED = False


def loadDataFrame(path: str) -> DataFrame:
    """读取数据和归一化"""
    dataframe: DataFrame = pandas.read_csv(path, sep=";", na_values="?")
    dataframe["datetime"] = pandas.to_datetime(dataframe["日期"] + " " + dataframe["时刻"], dayfirst=True)
    dataframe.drop(columns=["日期", "时刻", "电压", "电流", "厨房用电功率", "洗衣间用电功率"], inplace=True)
    dataframe.set_index("datetime", inplace=True)
    # 提取时间特征；年不重要
    dataframe["timestamp"] = dataframe.index.asi8 / 1e9
    month = dataframe["month"] = dataframe.index.month
    day = dataframe["day"] = dataframe.index.days_in_month
    hour = dataframe["hour"] = dataframe.index.hour
    minute = dataframe["minute"] = dataframe.index.minute
    dayOfWeek = dataframe["dayOfWeek"] = dataframe.index.dayofweek
    dataframe["monthSin"] = numpy.sin(2 * PI * month / 12)
    dataframe["monthCos"] = numpy.cos(2 * PI * month / 12)
    dataframe["daySin"] = numpy.sin(2 * PI * day / 31)
    dataframe["dayCos"] = numpy.cos(2 * PI * day / 31)
    dataframe["hourSin"] = numpy.sin(2 * PI * hour / 24)
    dataframe["hourCos"] = numpy.cos(2 * PI * hour / 24)
    dataframe["minuteSin"] = numpy.sin(2 * PI * minute / 60)
    dataframe["minuteCos"] = numpy.cos(2 * PI * minute / 60)
    dataframe["dayOfWeekSin"] = numpy.sin(2 * PI * dayOfWeek / 7)
    dataframe["dayOfWeekCos"] = numpy.cos(2 * PI * dayOfWeek / 7)
    dataframe = dataframe.astype(DATA_TYPE)
    dataframe.interpolate(inplace=True)  # 孩子们我才知道panda可以自动补充缺失值👍

    return dataframe


def generateSeq(vector: Matrix, targetIndex: int) -> tuple[Array[Matrix], Array[Vector]]:
    """
    我不知道这么写滑动窗口是不是内存占用会非常高🤔
    这里其实可以直接创建两个numpy array而不是python list，但是前者有点难写ww
    """
    inData: list[Matrix] = []
    outData: list[Matrix] = []
    for i in tqdm(range(int((len(vector) - DATA_LENGTH - PREDICT_LENGTH) * DATA_KEEP)), desc="Generate Sequences",
                  ncols=100, disable=TQDM_DISABLED):
        dataBegin = i
        predictBegin = i + DATA_LENGTH
        inData.append(vector[dataBegin:dataBegin + DATA_LENGTH])
        outData.append(vector[predictBegin:predictBegin + PREDICT_LENGTH, targetIndex])
    return typing.cast(tuple[Array[Matrix], Array[Vector]], (numpy.asarray(inData), numpy.asarray(outData)))


def train(model: Model, trainLoader: DataLoader, verifyLoader: DataLoader, epochs: int, learningRate: float):
    device: torch.device = torch.device("cpu")
    model.to(device)
    optimizer: Optimizer = AdamW(model.parameters(), lr=learningRate)
    lossMethod: MSELoss = MSELoss()

    progress = tqdm(total=epochs * 2, desc="Train Model", ncols=100, disable=TQDM_DISABLED)
    progressTL = tqdm(total=len(trainLoader), desc="Train Loss", ncols=100, leave=False, disable=TQDM_DISABLED)
    progressVL = tqdm(total=len(trainLoader), desc="Verify Loss", ncols=100, leave=False, disable=TQDM_DISABLED)
    for i in range(epochs):
        progress.set_description(f"Train Model (Epoch: {i + 1}/{epochs})")
        progressTL.reset()
        progressVL.reset()

        # train
        model.train()
        trainLoss = 0.0
        for inBatch, outBatch in trainLoader:
            # check
            assert not torch.isnan(inBatch).any()
            assert not torch.isinf(inBatch).any()
            assert not torch.isnan(outBatch).any()
            assert not torch.isinf(outBatch).any()

            inBatch, outBatch = inBatch.to(device), outBatch.to(device)
            pred = model(inBatch)
            loss: Tensor = lossMethod(pred, outBatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progressTL.set_description(f"Train Loss (Loss: {loss.item():.4f})")
            trainLoss += loss.item()

            progressTL.update()
        trainLoss /= len(trainLoader)
        progress.update()

        # verify
        model.eval()
        verifyLoss = 0.0
        with torch.no_grad():
            for inBatch, outBatch in verifyLoader:

                inBatch, outBatch = inBatch.to(device), outBatch.to(device)
                pred = model(inBatch)
                loss: Tensor = lossMethod(pred, outBatch)
                progressVL.set_description(f"Verify Loss (Loss: {loss.item():.4f})")
                verifyLoss += loss.item()

                progressVL.update()
        verifyLoss /= len(verifyLoader)
        progress.update()

        tqdm.write(f"Epoch {i + 1}/{epochs}, Train Loss: {trainLoss:.4f}, Verify Loss: {verifyLoss:.4f}.")
    progressTL.close()
    progressVL.close()
    progress.close()


def predict(model: Model, inputSeq: Matrix, scaler: Scaler) -> Vector:
    model.eval()
    device: torch.device = next(model.parameters()).device
    with torch.no_grad():
        input_seq = torch.tensor(inputSeq, dtype=DTYPE).unsqueeze(0).to(device)
        pred: Tensor = model(input_seq)
        pred_np = pred.detach().cpu().numpy()   # shape: (1, 10)
        result: Vector = scaler.inverseTransform(pred_np).flatten()
        return result


def formatDuration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)

    result = ""
    if hours > 0:
        result += f"{hours}小时"
    if minutes > 0:
        result += f"{minutes}分"
    result += f"{sec}秒"

    return result


def main():
    startTime = time.time()
    tqdm.write(f"PyTorch 版本: {torch.__version__} ({torch.version.__version__})")
    tqdm.write(f"Numpy 版本: {numpy.__version__} ({numpy.version.__version__})")
    tqdm.write(f"Pandas 版本: {pandas.__version__}")
    tqdm.write(f"时间窗口: {DATA_LENGTH}")
    tqdm.write(f"预测长度: {PREDICT_LENGTH}")
    tqdm.write(f"数据保留比例: {DATA_KEEP}")
    tqdm.write(f"负载大小: {BATCH_SIZE}")
    tqdm.write(f"训练轮次: {EPOCHS}")
    tqdm.write(f"学习率: {LEARNING_RATE}\n")

    tqdm.write("初始化数据...")
    dataframe = loadDataFrame("../household_power_consumption.txt")
    timeIndex: int = dataframe.columns.get_loc("timestamp")  # 时间戳列
    targetIndex: int = dataframe.columns.get_loc("总功率")  # 预测的目标列
    matrix: Matrix = typing.cast(Matrix, dataframe.values)
    assert not numpy.isnan(matrix).any()
    assert not numpy.isinf(matrix).any()
    # del dataframe  # 节省内存

    # 准备模型所需的数据
    tqdm.write("准备模型所需的数据...")
    gc.collect()
    inData, outData = generateSeq(matrix, targetIndex)  # 用于训练
    assert len(inData) == len(outData)
    del matrix
    lastTimestamp = int(inData[-1][-1][timeIndex])
    # 老实了，我一开始还不知道scaler有什么用，结果transformer对0.x-2.x这么小范围的值不敏感。
    inScaler = Scaler3D(TRAIN_MODE, IN_SCALER_OUTPUT, inData.shape[2])
    outScaler = Scaler(TRAIN_MODE, OUT_SCALER_OUTPUT)
    inData = inScaler.transform(inData)
    outData = outScaler.transform(outData)
    lastInData = inData[-1]

    # 准备模型对象
    tqdm.write("准备模型...")
    gc.collect()
    model = Model(featureCount=len(inData[0][0]), predictLength=PREDICT_LENGTH)

    def compileModel() -> None:
        nonlocal inData, outData
        if TRAIN_MODE:
            split = int(len(inData) * VERIFY_SPLIT)
            trainData = ModelDataSet(inData[:split], outData[:split])
            verifyData = ModelDataSet(inData[split:], outData[split:])
            trainLoader = DataLoader(
                trainData, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=os.cpu_count(), pin_memory=True
            )
            verifyLoader = DataLoader(
                verifyData, batch_size=BATCH_SIZE,
                num_workers=os.cpu_count(), pin_memory=True
            )
            del inData, outData

            # 训练
            tqdm.write("开始训练...")
            gc.collect()
            train(model, trainLoader, verifyLoader, EPOCHS, LEARNING_RATE)
            del trainData, verifyData, trainLoader, verifyLoader

            # 保存模型以备后用
            tqdm.write("训练结束！")
            torch.save(model.state_dict(), MODEL_OUTPUT)
            tqdm.write(f"模型已保存到: {MODEL_OUTPUT}")
            print(f"模型总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        else:
            # 加载模型
            tqdm.write("加载预训练的模型...")
            model.load_state_dict(torch.load(MODEL_OUTPUT, weights_only=True))

    compileModel()

    # 推理
    tqdm.write("开始推理...")
    gc.collect()
    result: Vector = predict(model, lastInData, outScaler)
    del model
    timestamp = lastTimestamp
    tqdm.write(f"\n推理结束！预测结果：")
    # 模型本身的输出不包含时间戳，只包含总功率。因此我们需要额外计算时间戳
    for powerValue in result:

        timestamp += 60
        timeObj = pandas.to_datetime(datetime.fromtimestamp(timestamp))
        timeStr = timeObj.strftime("%d/%m/%Y;%H:%M:%S")

        tqdm.write(f"    时间: {timeStr}")
        tqdm.write(f"        预测总功率: {powerValue:.3f}")
        try:
            realPower: Optional[float] = dataframe.at[timeObj, "总功率"]
            tqdm.write(f"        实际总功率: {realPower}")
            tqdm.write(f"        预测偏差: {powerValue - realPower:.3f}")
        except KeyError:
            tqdm.write(f"        实际总功率: 未知")

    tqdm.write(f"'用电预测1'已完成。总耗时：{formatDuration(time.time() - startTime)}")


# 能训练了，但速度只有4it/s，这得训练一年。
# 问了 ChatGPT ，尝试优化性能；优化完还是4it/s，性能基本没有变化
# 尝试用float16优化性能；但是纯CPU训练不支持半精度，只能float32
# 删数据，降训练轮数；这回能跑了，几分钟就训练出来了。但是效果不佳
# 训练了一晚上，效果不佳，就算1440时间窗口好像还是不太行。
# 很可能是因为删数据删的太过了，1%的数据Transformer啥也学不到；去租GPU时间吧。
# 用Google Colab训练出来了，连T4 GPU都跑了4个多小时。不过verify loss没有显著下降，0.01还是太高了。等会跑一下推理
# 跑了一下效果也很差...几乎和最开始1%数据超小时间窗口的效果差不多...wtf?
# 这一切都是骗局！没有任何规律！Transformer什么都学不到！这就是个平滑随机数！跑了acf相关性只有2个元素、2分钟！一整天的时间窗口全喂了狗！
# 傻孩子们，快跑啊！
if __name__ == '__main__':
    main()
