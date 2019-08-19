### [incubating] xgboost-launcher
XGBoost-Launcher is an another python wrapper of xgboost, which is extendable, adaptive and docker-friendly.

One major purpose of xgboost-launcher is running xgboost on k8s clusters, with the orchestration of [xgboost-operator](https://github.com/kubeflow/xgboost-operator).
So, xgboost-launcher is designed to package and run xgboost pipelines as jobs, which can be run either in local or in distributed runtime.

What's more, with user-defined data sources and model sources, xgboost-launcher is able to work under various systems.

<br>

#### How to install

XGBoost-Launcher is a separate python package (required python >= 3.6), which depends on xgboost python package.

Due to auto-training integration, xgboost-launcher is depended on [ant-xgboost](https://pypi.org/manage/project/ant-xgboost).

```bash
pip install xgboost-launcher
```

<br>

#### How to use

Running with unified configure is a recommended way to launch xgboost-launcher, because it is easy to package for remote cluster (e.g. k8s, yarn) submission. 

##### run locally with unified configure

Here is a demo to run a standalone job with unified configure.

First, we provide a demo with [ODPS](https://github.com/aliyun/aliyun-odps-python-sdk) data source._
```python
from launcher import launch_c
# train pipeline
with open('train_demo.yaml', 'r') as f:
    conf_str = f.read()
launch_c(conf_str)

# predict pipeline
with open('predict_demo.yaml', 'r') as f:
    conf_str = f.read()
launch_c(conf_str) 
```

* train_demo.yaml
```yaml
job: train
train_conf:
  xgboost_conf:
    num_boost_round: 100
    params:
      max_depth: 5
      eta: 0.1
      tree_method: hist
      num_class: 3
      objective: 'multi:softprob'
  data_conf:
    column_format:
      features:
        columns: sparse_kv_feature
        feature_num: 200
        item_delimiter: ','
        is_sparse: true
      label: label
    data_source:
      name: odps
      config:
        access_id: '*****'
        access_key: '*****'
        project: '*****'
        endpoint: '*****'
        input_table: train_table'
    valid_data_source:
      name: odps
      config:
        access_id: '*****'
        access_key: '*****'
        project: '*****'
        endpoint: '*****'
        input_table: 'valid_table'
  model_conf:
    model_path: xgb_launcher_test_path/test_model_123
    model_source:
      name: oss
      config:
        access_id: '*****'
        access_key: '*****'
        endpoint: '*****'
        bucket: '*****'
    dump_conf:
      path: xgb_launcher_test_path/test_model_123_info
      dump_format: json
      is_dump_fscore: true
```
* predict_demo.yaml
```yaml
job: predict
predict_conf:
  data_conf:
    builder:
      batch_size: 1024
    column_format:
      features:
        columns: sparse_kv_feature
        feature_num: 200
        item_delimiter: ','
        is_sparse: true
      append_columns: ['uuid']
      result_columns:
        result_column: 'ret_col'
        probability_column: 'prob_col'
        detail_column: 'detail_col'
        leaf_column: 'leaf_encoding_col'
    data_source:
      name: odps
      config:
        access_id: '*****'
        access_key: '*****'
        project: '*****'
        endpoint: '*****'
        input_table: 'prediction_dataset'
        output_table: 'prediction_result_set'
  model_conf:
    model_path: xgb_launcher_test_path/test_model_123
    model_source:
      name: oss
      config:
        access_id: '*****'
        access_key: '*****'
        endpoint: '*****'
        bucket: '*****'
```

Second, we also provide a local csv data source demo based on the iris dataset.
```python
from launcher import launch_c
# train pipeline
with open('train_demo_csv.yaml', 'r') as f:
    conf_str = f.read()
launch_c(conf_str)

# predict pipeline
with open('predict_demo_csv.yaml', 'r') as f:
    conf_str = f.read()
launch_c(conf_str) 
```

* train_demo_csv.yaml
```yaml
job: train
train_conf:
  xgboost_conf:
    num_boost_round: 100
    params:
      max_depth: 3
      eta: 0.1
      tree_method: hist
      num_class: 3
      objective: 'multi:softprob'
  data_conf:
    data_source:
      name: csv
      config:
          input_table: iris.csv
          output_table: 'prediction_result.csv'
    column_format:
      features:
          columns: 'Sepal.Length,Sepal.Width,Petal.Length,Petal.Width'
          feature_num: 4
          item_delimiter: ','
          kv_delimiter: ':'
          is_sparse: false
      label: Species    
  model_conf:
    model_path: test_model_123
    model_source:
      name: csv
      config:
```
* predict_demo_csv.yaml
```yaml
job: predict
predict_conf:
  data_conf:
    data_source:
      name: csv
      config:
          input_table: iris.csv
          output_table: 'prediction_result.csv'
    column_format:
      features:
          columns: 'Sepal.Length,Sepal.Width,Petal.Length,Petal.Width'
          feature_num: 4
          item_delimiter: ','
          kv_delimiter: ':'
          is_sparse: false
      append_columns: ['real_label']
      result_columns:
        result_column: 'ret_col'
        probability_column: 'prob_col'
        detail_column: 'detail_col'
        leaf_column: 'leaf_encoding_col'    
  model_conf:
    model_path: test_model_123
    model_source:
      name: csv
      config:
```
* iris.csv
```csv
5.1,3.5,1.4,0.2,1
4.9,3.0,1.4,0.2,1
4.7,3.2,1.3,0.2,1
6.4,3.2,4.5,1.5,0
6.9,3.1,4.9,1.5,0
5.5,2.3,4.0,1.3,0
5.8,2.8,5.1,2.4,2
6.4,3.2,5.3,2.3,2
6.5,3.0,5.5,1.8,2
```

##### run on k8s via [xgboost-operator](https://github.com/kubeflow/xgboost-operator)
coming soon......

<br>

#### Adapting xgboost-launcher to your own system
It is convenient to make a xgboost-launcher adaption for your own system.
Firstly, write your own `DataSource` and `ModelSource`. Then, register them.

 
##### DataSource
```python
from typing import Iterator
from launcher import XGBoostRecord, XGBoostResult, config_fields

class DataSource:
    def __init__(self, 
                 rank: int, 
                 num_worker: int,
                 column_conf: config_fields.ColumnFields,
                 source_conf):
        pass
        
    @abstractmethod
    def read(self) -> Iterator[XGBoostRecord]:
        pass

    @abstractmethod
    def write(self, result_iter: Iterator[XGBoostResult]):
        pass
```
##### ModelSource
```python
from typing import List

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
##### register sources
```python
from typing import NamedTuple, Iterator
from launcher import DataSource, ModelSource, register_data_source, register_model_source, XGBoostResult, XGBoostRecord

# register a mock data source
class MockDataConfig(NamedTuple):
    field_x: str
    field_y: int
    field_z: float


class MockDataSource(DataSource):
    def read(self) -> Iterator[XGBoostRecord]:
        pass

    def write(self, result_iter: Iterator[XGBoostResult]):
        pass
        
        
register_data_source('mock_data_source', MockDataConfig, MockDataSource)


# register a mock model source
class MockModelConfig(NamedTuple):
    field_a: str
    field_b: int
    field_c: float


class MockModelSource(ModelSource):
    def read_buffer(self, model_path: str) -> bytes:
        pass

    def write_buffer(self, buf: bytes, model_path):
        pass

    def read_lines(self, model_path: str) -> List[str]:
        pass

    def write_lines(self, lines: List[str], model_path):
        pass


register_model_source('mock_model_source', MockModelConfig, MockModelSource)
```
##### a real demo
coming soon......
