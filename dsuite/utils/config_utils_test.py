# Copyright 2019 The D'Suite Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for config_utils."""

import unittest

from dsuite.utils.config_utils import merge_configs


class AddConfigsTest(unittest.TestCase):
    """Tests for `merge_configs`."""

    def test_identity(self):
        config = {'a': 1, 'b': {'c': 2}}
        result = merge_configs(config, {})
        self.assertIsNot(config, result)
        self.assertDictEqual(config, result)

        reverse_result = merge_configs({}, config)
        self.assertIsNot(config, reverse_result)
        self.assertDictEqual(config, reverse_result)

    def test_disjoint(self):
        result = merge_configs({'a': 1}, {'b': 2})
        self.assertDictEqual(result, {'a': 1, 'b': 2})

    def test_overwrite(self):
        result = merge_configs({'a': 1, 'b': 2}, {'b': 3})
        self.assertDictEqual(result, {'a': 1, 'b': 3})

    def test_multiple(self):
        result = merge_configs({
            'a': 1,
            'b': 2
        }, {
            'b': 3,
            'c': 4
        }, {
            'a': 5,
            'd': 6
        })
        self.assertDictEqual(result, {'a': 5, 'b': 3, 'c': 4, 'd': 6})

    def test_nested(self):
        result = merge_configs({
            'a': {
                'b': 1,
            },
        }, {
            'a': {
                'c': 2,
            },
        }, {
            'a': {
                'b': 3,
                'd': 4,
            },
        })
        self.assertDictEqual(result, {'a': {'b': 3, 'c': 2, 'd': 4}})

    def test_multiple_nested(self):
        result = merge_configs({
            'a': {
                'b': {
                    'c': 1,
                },
                'd': 2,
            },
        }, {
            'a': {
                'b': {
                    'c': 2,
                    'e': 3,
                }
            },
            'f': {
                'g': 4
            },
        })
        self.assertDictEqual(result, {
            'a': {
                'b': {
                    'c': 2,
                    'e': 3,
                },
                'd': 2,
            },
            'f': {
                'g': 4
            },
        })


if __name__ == '__main__':
    unittest.main()
