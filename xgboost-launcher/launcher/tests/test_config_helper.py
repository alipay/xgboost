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

from typing import NamedTuple

import pytest

from ..config_helper import load_config, dump_config, field_keys_equal, fields_equal


class Education(NamedTuple):
    school: str
    degree: str
    gpa: str
    other_performance: str = 'No record'

    @classmethod
    def convert_degree(cls, value):
        if value.lower() == 'bachelor':
            return 'Bachelor'
        if value.lower() == 'master':
            return 'Master'
        if value.lower() == 'phd' or value.lower() == 'p.h.d':
            return 'PHD'
        raise NameError('Unknown degree!')

    @classmethod
    def convert_gpa(cls, values):
        return '%.2f' % values


class PhysicalStatus(NamedTuple):
    height: int
    weight: int


class Teacher(NamedTuple):
    name: str


class Student(NamedTuple):
    name: str
    gender: str
    age: int
    physical: PhysicalStatus
    education: Education
    mentor: Teacher = Teacher('unknown')

    @classmethod
    def convert_gender(cls, value):
        if value.lower().startswith('f'):
            return 'Female'
        if value.lower().startswith('m'):
            return 'Male'
        raise NameError('Unknown gender')

    @classmethod
    def convert_physical(cls, value):
        return {'height': value[0], 'weight': value[1]}


def test_load_config():
    education = load_config(
        Education,
        **{'school': 'Harvard', 'degree': 'bacheLOR', 'gpa': 3.5})
    assert education == Education('Harvard', 'Bachelor', '3.50')

    with pytest.raises(NameError, match='Unknown degree!'):
        load_config(Education, **{'school': 'A', 'degree': 'B', 'gpa': 3.5})

    student_conf = {
        'name': 'Linda',
        'gender': 'F',
        'age': 22,
        'education': {
            'school': 'Standford',
            'degree': 'Master',
            'gpa': 3.75,
            'other_performance': 'xxxx'
        },
        'physical': [168, 55],
        'mentor': {'name': 'Alfred Hitchcock'}
    }
    student = load_config(Student, **student_conf)
    assert student.name == 'Linda'
    assert student.gender == 'Female'
    assert student.age == 22
    assert student.education.school == 'Standford'
    assert student.education.degree == 'Master'
    assert student.education.gpa == '3.75'
    assert student.education.other_performance == 'xxxx'
    assert student.physical.height == 168
    assert student.physical.weight == 55
    assert student.mentor.name == 'Alfred Hitchcock'

    # test None value skipping
    cpy_conf = student_conf.copy()
    cpy_conf['education']['other_performance'] = None
    student = load_config(Student, **cpy_conf)
    assert student.education.other_performance == 'No record'

    # test empty kwargs skipping
    cpy_conf = student_conf.copy()
    cpy_conf.pop('mentor')
    student = load_config(Student, **cpy_conf)
    assert student.mentor.name == 'unknown'


def test_dump_config():
    student = Student(
        name='Mike', gender='M', age=30,
        education=Education('MIT', 'PHD', '3.25', '1 year gap'),
        physical=PhysicalStatus(182, 78))
    assert dump_config(student) == {
        'name': 'Mike', 'gender': 'M', 'age': 30,
        'education': {
            'school': 'MIT', 'degree': 'PHD', 'gpa': '3.25',
            'other_performance': '1 year gap'
        },
        'physical': {'height': 182, 'weight': 78},
        'mentor': {'name': 'unknown'}
    }


def test_fields_equal():
    p1 = PhysicalStatus(170, 70)
    p2 = PhysicalStatus(160, 60)
    assert field_keys_equal(p1, p2)
    assert not fields_equal(p1, p2)

    e1 = Education('u.c. Berkeley', 'Master', '3.35')
    e2 = Education('u.c. Berkeley', 'Master', '3.35')
    assert fields_equal(e1, e2)
    assert field_keys_equal(e1, e2)
