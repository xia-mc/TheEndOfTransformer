from typing import TypeAlias

import numpy
import torch

PI = numpy.pi
DATA_TYPE: TypeAlias = float
DTYPE: TypeAlias = torch.float32
Array: TypeAlias = numpy.ndarray
Vector: TypeAlias = Array[DTYPE]
Matrix: TypeAlias = Array[Vector]
