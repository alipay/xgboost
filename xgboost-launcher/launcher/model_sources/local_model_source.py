from typing import List

from ..model_source import ModelSource


class LocalModelSource(ModelSource):
    def read_buffer(self, model_path: str) -> bytes:
        with open(model_path, 'rb') as f:
            return f.read()

    def write_buffer(self, buf: bytes, model_path: str):
        with open(model_path, 'wb') as f:
            f.write(buf)

    def read_lines(self, model_path: str) -> List[str]:
        with open(model_path, 'r') as f:
            return f.readlines()

    def write_lines(self, lines: List[str], model_path: str):
        with open(model_path, 'w') as f:
            f.writelines(lines)
