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
from dataclasses import dataclass
import srt
import torchaudio
import torch as torch
from typing import List


@dataclass
class SrtBlock:
    text: str
    audio: torch.Tensor


class SrtReader:
    def __init__(self, wav_file: str, srt_file: str, device=torch.device("cpu")):
        """
        Reader of the SRT file and the corresponding WAV.
        """
        self.wav_content, self.sr = torchaudio.load(wav_file)
        assert self.sr == 16000, "Only 16 kHz audios are supported"
        with open(srt_file, 'r') as srt_fp:
            self.subtitles: List[srt.Subtitle] = list(srt.parse(srt_fp))
        self.device = device

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self) -> SrtBlock:
        if self._counter < len(self.subtitles):
            subtitle = self.subtitles[self._counter]
            start_frame = int(subtitle.start.total_seconds() * self.sr)
            end_frame = int(subtitle.end.total_seconds() * self.sr)
            audio_slice = self.wav_content[:, start_frame:end_frame].to(self.device)
            text_content = subtitle.content.replace("\n", " ").replace("  ", " ").strip()
            self._counter += 1
            return SrtBlock(text_content, audio_slice)
        raise StopIteration

    def __len__(self):
        return len(self.subtitles)
