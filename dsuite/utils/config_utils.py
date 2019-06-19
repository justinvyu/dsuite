# Copyright 2019 The DSuite Authors.
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

"""Utility methods for working with configurations."""

import collections
import copy
from typing import Any, Dict


def merge_configs(*configs) -> Dict[str, Any]:
    """Adds the given dictionaries together.

    Args:
        *configs: A sequence of dictionaries to recursively merge together.

    Returns:
        The total merged dictionary of all of the given dictionaries.
    """
    if not configs:
        raise ValueError('At least one dictionary must be given.')
    total_config = copy.deepcopy(configs[0])
    for config in configs[1:]:
        _merge_config(total_config, config)
    return total_config


def _merge_config(config_a: Dict[str, Any], config_b: Dict[str, Any]):
    """Merges the second dictionary into the first dictionary."""
    for key, value_b in config_b.items():
        if key not in config_a:
            config_a[key] = copy.deepcopy(value_b)
            continue

        value_a = config_a[key]
        is_dict_a = isinstance(value_a, collections.Mapping)
        is_dict_b = isinstance(value_b, collections.Mapping)

        # Ensure either both values are dictionaries or neither.
        if is_dict_a ^ is_dict_b:
            raise ValueError(
                "Cannot merge dictionary with non-dictionary value: {}, {}"
                .format(value_a, value_b))

        # Recursively merge nested-dictionaries.
        if is_dict_a:
            assert is_dict_b
            _merge_config(value_a, value_b)
        else:
            # Override non-dictionary fields.
            config_a[key] = copy.deepcopy(value_b)
