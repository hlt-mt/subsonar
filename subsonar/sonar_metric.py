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
from typing import List
import logging

from sonar.inference_pipelines import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
import torch as torch
from torch.nn import functional as F


LOGGER = logging.getLogger(__name__)


class SonarAudioTextMetric:
    # taken from https://github.com/facebookresearch/SONAR
    AUDIO_LANGS = {
        "arb",  # Standard Arabic
        "ben",  # Bengali
        "cat",  # Catalan
        "ces",  # Czech
        "cmn",  # Mandarin Chinese
        "cym",  # Welsh
        "dan",  # Danish
        "deu",  # German
        "eng",  # English
        "est",  # Estonian
        "fin",  # Finnish
        "fra",  # French
        "hin",  # Hindi
        "ind",  # Indonesian
        "ita",  # Italian
        "jpn",  # Japanese
        "kan",  # Kannada
        "kor",  # Korean
        "mlt",  # Maltese
        "nld",  # Dutch
        "pes",  # Iranian Persian
        "pol",  # Polish
        "por",  # Portuguese
        "ron",  # Romanian
        "rus",  # Russian
        "slk",  # Slovak
        "spa",  # Spanish
        "rus",  # Russian
        "swe",  # Swedish
        "swh",  # Swahili
        "tam",  # Tamil
        "tel",  # Telugu
        "tgl",  # Tagalog
        "tha",  # Thai
        "tur",  # Turkish
        "ukr",  # Ukrainian
        "urd",  # Urdu
        "uzn",  # Northern Uzbek
        "vie",  # Vietnamese
    }

    # taken from https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/modeling/scripts/flores200/langs.txt  # noqa:E501,W505
    TEXT_LANGS = set("ace_Arab,ace_Latn,acm_Arab,acq_Arab,aeb_Arab,afr_Latn,ajp_Arab,aka_Latn,"
                     "amh_Ethi,apc_Arab,arb_Arab,ars_Arab,ary_Arab,arz_Arab,asm_Beng,ast_Latn,"
                     "awa_Deva,ayr_Latn,azb_Arab,azj_Latn,bak_Cyrl,bam_Latn,ban_Latn,bel_Cyrl,"
                     "bem_Latn,ben_Beng,bho_Deva,bjn_Arab,bjn_Latn,bod_Tibt,bos_Latn,bug_Latn,"
                     "bul_Cyrl,cat_Latn,ceb_Latn,ces_Latn,cjk_Latn,ckb_Arab,crh_Latn,cym_Latn,"
                     "dan_Latn,deu_Latn,dik_Latn,dyu_Latn,dzo_Tibt,ell_Grek,eng_Latn,epo_Latn,"
                     "est_Latn,eus_Latn,ewe_Latn,fao_Latn,pes_Arab,fij_Latn,fin_Latn,fon_Latn,"
                     "fra_Latn,fur_Latn,fuv_Latn,gla_Latn,gle_Latn,glg_Latn,grn_Latn,guj_Gujr,"
                     "hat_Latn,hau_Latn,heb_Hebr,hin_Deva,hne_Deva,hrv_Latn,hun_Latn,hye_Armn,"
                     "ibo_Latn,ilo_Latn,ind_Latn,isl_Latn,ita_Latn,jav_Latn,jpn_Jpan,kab_Latn,"
                     "kac_Latn,kam_Latn,kan_Knda,kas_Arab,kas_Deva,kat_Geor,knc_Arab,knc_Latn,"
                     "kaz_Cyrl,kbp_Latn,kea_Latn,khm_Khmr,kik_Latn,kin_Latn,kir_Cyrl,kmb_Latn,"
                     "kon_Latn,kor_Hang,kmr_Latn,lao_Laoo,lvs_Latn,lij_Latn,lim_Latn,lin_Latn,"
                     "lit_Latn,lmo_Latn,ltg_Latn,ltz_Latn,lua_Latn,lug_Latn,luo_Latn,lus_Latn,"
                     "mag_Deva,mai_Deva,mal_Mlym,mar_Deva,min_Latn,mkd_Cyrl,plt_Latn,mlt_Latn,"
                     "mni_Beng,khk_Cyrl,mos_Latn,mri_Latn,zsm_Latn,mya_Mymr,nld_Latn,nno_Latn,"
                     "nob_Latn,npi_Deva,nso_Latn,nus_Latn,nya_Latn,oci_Latn,gaz_Latn,ory_Orya,"
                     "pag_Latn,pan_Guru,pap_Latn,pol_Latn,por_Latn,prs_Arab,pbt_Arab,quy_Latn,"
                     "ron_Latn,run_Latn,rus_Cyrl,sag_Latn,san_Deva,sat_Olck,scn_Latn,shn_Mymr,"
                     "sin_Sinh,slk_Latn,slv_Latn,smo_Latn,sna_Latn,snd_Arab,som_Latn,sot_Latn,"
                     "spa_Latn,als_Latn,srd_Latn,srp_Cyrl,ssw_Latn,sun_Latn,swe_Latn,swh_Latn,"
                     "szl_Latn,tam_Taml,tat_Cyrl,tel_Telu,tgk_Cyrl,tgl_Latn,tha_Thai,tir_Ethi,"
                     "taq_Latn,taq_Tfng,tpi_Latn,tsn_Latn,tso_Latn,tuk_Latn,tum_Latn,tur_Latn,"
                     "twi_Latn,tzm_Tfng,uig_Arab,ukr_Cyrl,umb_Latn,urd_Arab,uzn_Latn,vec_Latn,"
                     "vie_Latn,war_Latn,wol_Latn,xho_Latn,ydd_Hebr,yor_Latn,yue_Hant,zho_Hans,"
                     "zho_Hant,zul_Latn".split(","))

    def __init__(self, audio_lang: str, text_lang: str, device=torch.device("cpu")):
        """
        Scorer of the similarity between an audio and a text using the SONAR model.
        The audio language should be expressed in the three-letter codes of the ISO 639-3
        standard, while the text language should be expressed in the FLORES 200 language code.
        """
        assert audio_lang in SonarAudioTextMetric.AUDIO_LANGS, \
            f"{audio_lang} is not supported. Supported langs are: {self.AUDIO_LANGS}"
        assert text_lang in SonarAudioTextMetric.TEXT_LANGS, \
            f"{text_lang} is not supported. Supported langs are: {self.TEXT_LANGS}"
        self.audio_encoder = SpeechToEmbeddingModelPipeline(
            encoder=f"sonar_speech_encoder_{audio_lang}", device=device)
        self.text_lang = text_lang
        self.text_encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device)

    @torch.inference_mode()
    def score(self, text: str, audio: torch.Tensor) -> float:
        """
        Scores the similarity between the given audio and text using the cosine similarity of their
        SONAR embeddings.

        :param text: text to score
        :param audio: audio in format [channels, time] with sampling rate 16 kHz.
        :return: a float in the range [-1, 1]
        """
        text_embedding = self.text_encoder.predict([text], source_lang=self.text_lang)
        speech_embedding = self.audio_encoder.predict([audio])
        self.nan_to_zero(text_embedding, text, emb_type="text")
        self.nan_to_zero(speech_embedding, text, emb_type="audio")
        return F.cosine_similarity(text_embedding, speech_embedding).item()

    @torch.inference_mode()
    def batch_score(self, texts: List[str], audios: List[torch.Tensor]) -> List[float]:
        """
        Scores the similarity between the given audio and text using the cosine similarity of their
        SONAR embeddings.

        :param texts: list of texts to score
        :param audios: list of audios in format [channels, time] with sampling rate 16 kHz.
        :return: a list of floats in the range [-1, 1]
        """
        text_embeddings = self.text_encoder.predict(texts, source_lang=self.text_lang)
        speech_embeddings = self.audio_encoder.predict(audios)
        self.nan_to_zero(text_embeddings, texts, emb_type="text")
        self.nan_to_zero(speech_embeddings, texts, emb_type="audio")
        return F.cosine_similarity(text_embeddings, speech_embeddings).tolist()

    def nan_to_zero(self, embeddings, text, emb_type="audio"):
        """
        Sets to 0.0 embeddings that are NaN.
        This happens when the audio slice is less than 40 ms or with empty text.
        """
        nan_mask = embeddings.isnan()
        if nan_mask.any():
            LOGGER.warning(
                f"Found NaN in {emb_type} embeddings for: {text}\nSetting to zero. This usually "
                "happens when the audio slice is less than 40 ms or with empty text. Please check "
                "your data.")
            embeddings.masked_fill_(nan_mask, 0.0)

    def merge_scores(self, scores: List[float]) -> float:
        return sum(scores) / len(scores)
