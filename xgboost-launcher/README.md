### XGBoost-Launcher
An extendable, cloud-native XGBoost based machine learning pipeline. With user-defined DataSources and ModelSources, XGBoost-Launcher is able to work under various environments.

#### DataSource API
A handler about data reading/writing, which is compatible with both single-machine and distributed runtime.
 

```python
class DataSource:
    def __init__(self, 
                 rank: int, 
                 num_worker: int,
                 column_conf: configs.ColumnFields,
                 source_conf):
        pass
        
    @abstractmethod
    def read(self) -> Iterator[XGBoostRecord]:
        pass

    @abstractmethod
    def write(self, result_iter: Iterator[XGBoostResult]):
        pass
```

#### ModelSource API
A handler by which XGBLauncher save/load model(booster) and related information.

```python
class ModelSource:
    def __init__(self, source_conf):
        pass

    @abstractmethod
    def read_buffer(self, model_path: str) -> bytes:
        pass

    @abstractmethod
    def write_buffer(self, buf: bytes, model_path: str):
        pass

    @abstractmethod
    def read_lines(self, model_path: str) -> List[str]:
        pass

    @abstractmethod
    def write_lines(self, lines: List[str], model_path: str):
        pass
```

#### install XGBoost-Launcher

XGBoost-Launcher is a python package, which should install separately.

Due to auto-training integration, for now, XGBoost-Launcher is only compatible with Ant-XGBoost.

By the way, XGBoost-Launcher requires python >= 3.6.

Running below codes, XGBoost-Launcher and Ant-XGBoost it depended on will be installed.
__Meanwhile, the setup program will uninstall existing xgboost in current environment, if is not compatible with XGBoost-Launcher.__ 

```bash
cd xgboost-launcher
python setup.py install
```




 
