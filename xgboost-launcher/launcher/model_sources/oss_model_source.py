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

import os
import tempfile
from typing import List, NamedTuple

import oss2

from ..model_source import ModelSource


class OSSFields(NamedTuple):
    access_id: str
    access_key: str
    bucket: str
    endpoint: str


class OssModelSource(ModelSource):
    def __init__(self, source_conf):
        super().__init__(source_conf)
        assert isinstance(source_conf, OSSFields)
        auth = oss2.Auth(source_conf.access_id, source_conf.access_key)
        self._bkt = oss2.Bucket(
            auth=auth,
            endpoint=source_conf.endpoint,
            bucket_name=source_conf.bucket)

    def read_buffer(self, model_path: str) -> bytes:
        tmp_file = os.path.join(tempfile.mkdtemp(), 'oss_tmp')
        self._bkt.get_object_to_file(model_path, tmp_file)
        with open(tmp_file, 'rb') as f:
            return f.read()

    def write_buffer(self, buf: bytes, model_path: str):
        self._bkt.put_object(model_path, buf)

    def read_lines(self, model_path: str) -> List[str]:
        tmp_file = os.path.join(tempfile.mkdtemp(), 'oss_tmp')
        with open(tmp_file, 'r') as f:
            return f.readlines()

    def write_lines(self, lines: List[str], model_path: str):
        self._bkt.put_object(model_path, '\n'.join(lines))
