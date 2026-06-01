# Copyright 2026 Google LLC
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

"""Compatibility layer for legacy omics helpers.

Prefer native omics support in `tunix.models.qwen3`.
"""

from tunix.models.omics.model import OmicsAgnosticLM
from tunix.models.omics.model import OmicsConfig
from tunix.models.omics.model import OmicsEmbedderAdapter
from tunix.models.omics.model import load_omics_model
