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
import logging
import unittest

import torch

from tests.fake_utils import FakeSonarAudioTextMetric, FixedValuePredictor, NanFixedValuePredictor


class NanEmbeddingsTestCase(unittest.TestCase):
    def setUp(self):
        self.sonar_metric_single_nan_audio = FakeSonarAudioTextMetric(
            FixedValuePredictor(out_seq_len=8), NanFixedValuePredictor(out_seq_len=8))
        self.sonar_metric_all_nan_audio = FakeSonarAudioTextMetric(
            FixedValuePredictor(out_seq_len=8),
            NanFixedValuePredictor(out_seq_len=8, nan_per_batch=8))
        self.sonar_metric_single_nan_text = FakeSonarAudioTextMetric(
            NanFixedValuePredictor(out_seq_len=8), FixedValuePredictor(out_seq_len=8))
        self.sonar_metric_all_nan_text = FakeSonarAudioTextMetric(
            NanFixedValuePredictor(out_seq_len=8, nan_per_batch=8),
            FixedValuePredictor(out_seq_len=8))
        self.sonar_metric_single_nan_both = FakeSonarAudioTextMetric(
            NanFixedValuePredictor(out_seq_len=8), NanFixedValuePredictor(out_seq_len=8))
        self.sonar_metric_all_nan_both = FakeSonarAudioTextMetric(
            NanFixedValuePredictor(out_seq_len=8, nan_per_batch=8),
            NanFixedValuePredictor(out_seq_len=8, nan_per_batch=8))

    def test_nan_to_zero_single_audio(self):
        score = self.sonar_metric_single_nan_audio.score("a", torch.zeros((1, 1)))
        self.assertEqual(0.0, score)
        score = self.sonar_metric_all_nan_audio.score("a", torch.zeros((1, 1)))
        self.assertEqual(0.0, score)

    def test_nan_to_zero_single_text(self):
        score = self.sonar_metric_single_nan_text.score("a", torch.zeros((1, 1)))
        self.assertEqual(0.0, score)
        score = self.sonar_metric_all_nan_text.score("a", torch.zeros((1, 1)))
        self.assertEqual(0.0, score)

    def test_nan_to_zero_batch_audio(self):
        fake_audios = [torch.zeros((1, 1))] * 8
        fake_texts = ["a", "b", "c", "d", "e", "f", "g", "h"]
        scores = self.sonar_metric_single_nan_audio.batch_score(fake_texts, fake_audios)
        self.assertIn(0.0, scores)
        self.assertAlmostEqual(7.0, sum(scores), places=6)
        scores = self.sonar_metric_all_nan_audio.batch_score(fake_texts, fake_audios)
        self.assertListEqual([0.0] * 8, scores)

    def test_nan_to_zero_batch_text(self):
        fake_audios = [torch.zeros((1, 1))] * 8
        fake_texts = ["a", "b", "c", "d", "e", "f", "g", "h"]
        scores = self.sonar_metric_single_nan_text.batch_score(fake_texts, fake_audios)
        self.assertIn(0.0, scores)
        self.assertAlmostEqual(7.0, sum(scores), places=6)
        scores = self.sonar_metric_all_nan_text.batch_score(fake_texts, fake_audios)
        self.assertListEqual([0.0] * 8, scores)

    def test_nan_to_zero_batch_both(self):
        fake_audios = [torch.zeros((1, 1))] * 8
        fake_texts = ["a", "b", "c", "d", "e", "f", "g", "h"]
        scores = self.sonar_metric_single_nan_both.batch_score(fake_texts, fake_audios)
        self.assertIn(0.0, scores)
        # it can be either 7.0 if the NaN is in the same position or 6.0 if it is not
        self.assertGreater(sum(scores), 5.9)
        self.assertLess(sum(scores), 7.1)
        scores = self.sonar_metric_all_nan_both.batch_score(fake_texts, fake_audios)
        self.assertListEqual([0.0] * 8, scores)

    def test_warning(self):
        with self.assertLogs(level=logging.WARNING) as warn_cm:
            _ = self.sonar_metric_single_nan_text.score("a", torch.zeros((1, 1)))
        self.assertEqual(len(warn_cm.output), 1)
        self.assertIn("Found NaN in text embeddings", warn_cm.output[0])

        fake_audios = [torch.zeros((1, 1))] * 8
        fake_texts = ["a", "b", "c", "d", "e", "f", "g", "h"]
        with self.assertLogs(level=logging.WARNING) as warn_cm:
            _ = self.sonar_metric_single_nan_audio.batch_score(fake_texts, fake_audios)
        self.assertEqual(len(warn_cm.output), 1)
        self.assertIn("Found NaN in audio embeddings", warn_cm.output[0])

        with self.assertLogs(level=logging.WARNING) as warn_cm:
            _ = self.sonar_metric_single_nan_both.batch_score(fake_texts, fake_audios)
        self.assertEqual(len(warn_cm.output), 2)
        self.assertIn("Found NaN in text embeddings", warn_cm.output[0])
        self.assertIn("Found NaN in audio embeddings", warn_cm.output[1])


if __name__ == '__main__':
    unittest.main()
