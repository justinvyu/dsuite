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

"""Helper methods to locate asset files."""

import os

_MODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))

_MODULE_PARENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def get_asset_path(path: str):
    """Returns the absolute path of the given fully-qualified resource path.

    Example:
        >>> get_asset_path('dsuite/dclaw/assets/')

    Args:
        path: The fully-qualified path to the resource, with components
            separated by slashes.
    """
    if path.startswith('dsuite'):
        return os.path.join(_MODULE_DIR, path)
    elif path.startswith('dsuite-scenes'):
        return os.path.join(_MODULE_DIR, path)
    raise ValueError('Unknown path root: ' + path)
