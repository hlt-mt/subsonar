# SONAR Subtitling Evaluator

Code to evaluate the quality of SRT files using the multilingual multimodal [SONAR sentence embedding model](https://github.com/facebookresearch/SONAR).

The evaluation accounts for the semantic similarity (computed as a cosine similarity)
between each subtitle block and the corresponding audio to which the block is assigned to
(through the timestamps in the SRT). The returned scores range in `[-1, 1]`
where the higher, the better.


## Installation

Ensure that you have `libsndfile` installed in you environment.
Then, run:

```bash
pip install SubSONAR
```

or, in the source root of this repository:

```bash
pip install -e .
```

The installation has been tested with python 3.8 and 3.10.

## Usage

Example usage for Italian SRTs and English audios of two files (1 and 2):

```bash
subsonar \
  --srt-files 1.srt 2.srt \
  --audio-files 1.wav 2.wav \
  --text-lang ita_Latn --audio-lang eng \
  -bs 32
```

Please set the batch size `bs` according to your GPU capacity.

The available languages for the speech encoder (`--audio-lang`) can be found in the
[SONAR repository](https://github.com/facebookresearch/SONAR/blob/main/README.md#supported-languages-and-download-links),
while the text encoder (`--text-lang`) supports the
[200 languages of NLLB](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/scripts/flores200/langs.txt).

## License

**SONAR Subtitling Evaluator** is licensed under [Apache Version 2.0](LICENSE).

However, the SONAR encoders have a dedicated license that can be found
in [their repository LICENSE](https://github.com/facebookresearch/SONAR/blob/main/LICENSE.md).
Please check the license for the encoders you are using.

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{gaido-et-al-2024-sbaam,
title = {{SBAAM! Eliminating Transcript Dependency in Automatic Subtitling}},
author = {Gaido, Marco and Papi, Sara and Negri, Matteo and Cettolo, Mauro and Bentivogli, Luisa},
booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
year = "2024",
address = "Bangkok, Thailand",
}
```
