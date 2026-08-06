import re

LANGUAGE_NAMES = {'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada'}
LANGUAGE_PROMPTS = {'en': 'English', 'hi': 'Hindi (Hindi)', 'kn': 'Kannada (Kannada)'}
_DEVANAGARI_LO, _DEVANAGARI_HI = 0x0900, 0x097F
_KANNADA_LO,    _KANNADA_HI    = 0x0C80, 0x0CFF

def _count_block(text, lo, hi):
    return sum(1 for ch in text if lo <= ord(ch) <= hi)

def detect_language(text):
    if not text or not text.strip():
        return ('en', 1.0, 'English')
    stripped = text.strip()
    total = len(stripped)
    kn = _count_block(stripped, _KANNADA_LO,    _KANNADA_HI)
    hi = _count_block(stripped, _DEVANAGARI_LO, _DEVANAGARI_HI)
    kn_r = kn / total
    hi_r = hi / total
    if kn_r >= 0.08:
        return ('kn', round(min(1.0, kn_r * 4.0), 4), 'Kannada')
    if hi_r >= 0.08:
        return ('hi', round(min(1.0, hi_r * 4.0), 4), 'Hindi')
    ascii_r = sum(1 for ch in stripped if ord(ch) < 128) / total
    return ('en', round(min(0.99, 0.80 + 0.19 * ascii_r), 4), 'English')

def get_language_flag(lang_code):
    return {'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada'}.get(lang_code, '?')

def get_response_language_instruction(lang_code):
    if lang_code == 'hi':
        return ('Respond in Hindi. Keep traditional terms like Panchagavya, Bijamrita in '
                'their original Sanskrit/English form, but explain all context in Hindi.')
    if lang_code == 'kn':
        return ('Respond in Kannada. Keep traditional terms like Panchagavya, Bijamrita in '
                'their original form, but explain all context in Kannada.')
    return 'Respond in English. Be precise and use domain-specific terminology as documented.'