#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Iterator, NamedTuple

from ..data_source import DataSource
from ..data_units import XGBoostResult, XGBoostRecord


# TODO: implement this
class CsvFields(NamedTuple):
    pass


# TODO: implement this
class CsvDataSource(DataSource):
    def read(self) -> Iterator[XGBoostRecord]:
        pass

    def write(self, result_iter: Iterator[XGBoostResult]):
        pass
