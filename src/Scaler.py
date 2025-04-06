from pathlib import Path

import joblib
from sklearn.preprocessing import MinMaxScaler

from Const import Matrix, Array


class Scaler:
    def __init__(self, trainMode: bool, path: Path):
        self._trainMode = trainMode
        self._path = path
        if trainMode:
            self._scaler: MinMaxScaler = MinMaxScaler()
        else:
            self._scaler: MinMaxScaler = joblib.load(self._path)

    def _dump(self) -> None:
        self._path.parent.mkdir(exist_ok=True)
        joblib.dump(self._scaler, self._path)

    def transform(self, data: Matrix) -> Matrix:
        if self._trainMode:
            result = self._scaler.fit_transform(data)
            self._dump()
        else:
            result = self._scaler.transform(data)
        return result

    def inverseTransform(self, data: Matrix) -> Matrix:
        return self._scaler.inverse_transform(data)


class Scaler3D:
    def __init__(self, trainMode: bool, path: Path, size: int):
        self._trainMode = trainMode
        self._path = path
        self._size = size
        self._scalers: list[MinMaxScaler] = []

        if trainMode:
            self._scalers = [MinMaxScaler() for _ in range(size)]
        else:
            dir_ = self._path.parent / self._path.with_suffix("").name
            assert dir_.exists()
            for i in range(size):
                scaler = joblib.load(dir_ / f"{i}{self._path.suffix}")
                self._scalers.append(scaler)

    def _dump(self) -> None:
        self._path.parent.mkdir(exist_ok=True)
        dir_ = self._path.parent / self._path.with_suffix("").name
        dir_.mkdir(exist_ok=True)
        for i, scaler in enumerate(self._scalers):
            joblib.dump(scaler, dir_ / f"{i}{self._path.suffix}")

    def transform(self, data: Array[Matrix]) -> Array[Matrix]:
        data = data.copy()
        assert data.shape[2] == self._size

        for i in range(self._size):
            sub_data = data[:, :, i].reshape(-1, 1)
            if self._trainMode:
                data[:, :, i] = self._scalers[i].fit_transform(sub_data).reshape(data[:, :, i].shape)
            else:
                data[:, :, i] = self._scalers[i].transform(sub_data).reshape(data[:, :, i].shape)

        if self._trainMode:
            self._dump()

        return data

    def inverseTransform(self, data: Array[Matrix]) -> Array[Matrix]:
        data = data.copy()
        assert data.shape[2] == self._size

        for i in range(self._size):
            sub_data = data[:, :, i].reshape(-1, 1)
            data[:, :, i] = self._scalers[i].inverse_transform(sub_data).reshape(data[:, :, i].shape)

        return data
