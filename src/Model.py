from torch import Tensor
from torch.nn import Module, Linear, TransformerEncoderLayer, TransformerEncoder

DMODEL = 64
HEADS = 4
LAYERS = 4


class Model(Module):
    def __init__(self, featureCount: int, predictLength: int):
        super().__init__()
        self.input = Linear(featureCount, DMODEL)
        self.output = Linear(DMODEL, predictLength)
        layer = TransformerEncoderLayer(DMODEL, HEADS, batch_first=True)
        self.encoder = TransformerEncoder(layer, LAYERS)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        前向传播
        :param tensor: [batchSize, seqSize, featureCount]
        :return: [batchSize, predictSize]
        """
        tensor: Tensor = self.input(tensor)
        tensor = tensor.permute(1, 0, 2)  # [seqSize, batchSize, featureCount]
        tensor: Tensor = self.encoder(tensor)
        tensor = tensor.permute(1, 0, 2)  # [batchSize, seqSize, featureCount]
        # 预测
        result = self.output(tensor[:, -1])
        return result
