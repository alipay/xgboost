from typing import Iterator

from ...data_units import XGBoostResult, XGBoostRecord
from ...data_source import DataSource


# TODO: implement this
class CsvDataSource(DataSource):
    def read(self) -> Iterator[XGBoostRecord]:
        pass

    def write(self, result_iter: Iterator[XGBoostResult]):
        pass
