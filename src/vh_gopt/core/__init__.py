from vh_gopt.core.arpa_infer import ARPA39, detect_blank_id
from vh_gopt.core.prosody import (
    ENERGY_DIM,
    PROSODY_DIM,
    phone_prosody,
    phone_prosody_from_segs,
    phone_segments,
)
from vh_gopt.core.vendors import (
    align_sequence_dp,
    align_words_to_canon,
    get_canonical_word_phones,
    parse_iflytek_phones,
    parse_speechace_phones,
    parse_speechsuper_phones,
    IPA2ARPA_FULL,
)

PHONE_LIST = sorted(list(ARPA39))
PHONE2ID = {p: i for i, p in enumerate(PHONE_LIST)}
