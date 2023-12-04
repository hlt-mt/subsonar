# Copyright 2023 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from subsonar.sonar_metric import SonarAudioTextMetric

import random

import torch


class FakeSonarAudioTextMetric(SonarAudioTextMetric):
    """
    Fake SonarAudioTextMetric to test the SonarAudioTextMetric class without
    downloading Sonar pretrained models.
    """
    def __init__(self, fake_text_predictor, fake_audio_predictor, fake_txt_lang="eng"):
        self.audio_encoder = fake_audio_predictor
        self.text_encoder = fake_text_predictor
        self.text_lang = fake_txt_lang


class FixedValuePredictor:
    def __init__(self, out_seq_len=1024, fixed_value=1.):
        self.out_seq_len = out_seq_len
        self.fixed_value = fixed_value

    def predict(self, list_inputs, source_lang=None):
        return torch.full((len(list_inputs), self.out_seq_len), self.fixed_value)


class NanFixedValuePredictor(FixedValuePredictor):
    def __init__(self, out_seq_len=1024, fixed_value=1., nan_per_batch=1):
        super().__init__(out_seq_len, fixed_value)
        self.nan_per_batch = nan_per_batch

    def predict(self, list_inputs, source_lang=None):
        if len(list_inputs) == 1:
            return torch.full((1, self.out_seq_len), torch.nan)
        else:
            fixed_out = super().predict(list_inputs, source_lang=source_lang)
            assert len(list_inputs) >= self.nan_per_batch
            idxs_to_set_to_nan = random.sample(range(len(list_inputs)), self.nan_per_batch)
            fixed_out[idxs_to_set_to_nan, :] = torch.nan
            return fixed_out
