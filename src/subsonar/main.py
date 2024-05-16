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
import argparse
from importlib.metadata import version
import logging

import torch
from tqdm import tqdm
from typing import List, Optional

from subsonar.sonar_metric import SonarAudioTextMetric
from subsonar.srt_reader import SrtReader


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def main(
        srt_files: List[str],
        audio_files: List[str],
        audio_lang: str,
        text_lang: str,
        batch_size: int,
        width: float,
        bootstrap_ci_alpha: Optional[float] = None):
    scores = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = SonarAudioTextMetric(audio_lang, text_lang, device=device)
    for srt_file, audio_file in zip(srt_files, audio_files):
        LOGGER.info(f"Scoring {srt_file}...")
        srt_reader = SrtReader(audio_file, srt_file, )
        batch = []
        for block in tqdm(srt_reader):
            batch.append(block)
            if len(batch) >= batch_size:
                scores.extend(
                    metric.batch_score([b.text for b in batch], [b.audio for b in batch]))
                batch = []
        if len(batch) > 0:
            scores.extend(metric.batch_score([b.text for b in batch], [b.audio for b in batch]))
    overall_score = metric.merge_scores(scores)
    if bootstrap_ci_alpha is not None:
        raise NotImplementedError("CI has not been implemented yet")
    print(f"SubSONAR v{version('subsonar')} = {'{:.{}f}'.format(overall_score, width)}")


def cli_script():
    """
    Wrapper function for CLI script that reads arguments and invokes the main function,
    which computes SubSONAR.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--srt-files', type=str, nargs='+', required=True,
        help="the SRT file(s) for which the metrics should be computed")

    parser.add_argument(
        '--audio-files', type=str, nargs='+', required=True,
        help="the path to the audio files corresponding to each SRT")

    parser.add_argument(
        '--width', '-w', type=int, default=4,
        help='floating point width.')

    parser.add_argument(
        '--audio-lang', '-al', type=str, required=True,
        help='language of the speech in the audio file in three-letter code (e.g. eng).')

    parser.add_argument(
        '--text-lang', '-tl', type=str, required=True,
        help='language of the text in the SRT file in Flores 200 format (e.g. eng_Latn).')

    parser.add_argument(
        '--batch-size', '-bs', type=int, default=10,
        help='batch size to use in the inference phase.')

    parser.add_argument(
        '--bootstrap-ci-alpha', '-bsa', type=float, required=False,
        help='the p-value to use to build a confidence interval for the estimated metric.')

    parsed_args = parser.parse_args()
    LOGGER.info(f"Starting evaluation with arguments: {parsed_args}")
    assert len(parsed_args.srt_files) == len(parsed_args.audio_files), \
        f"SRT (len={len(parsed_args.srt_files)}) and audio (len={len(parsed_args.audio_files)}) " \
        "files should match"
    main(**parsed_args.__dict__)


if __name__ == '__main__':
    cli_script()
