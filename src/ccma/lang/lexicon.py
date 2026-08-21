"""Romanised-Hindi lexicon for the contact-centre domain.

Why a hand-curated lexicon at all, when we have neural LID?

Off-the-shelf LID (fastText lid.176, CLD3, Whisper's own detector) operates on
*script* statistics and labels romanised Hindi as English ~85% of the time. That
single failure poisons everything downstream: the utterance gets routed to an
English intent classifier, English sentiment lexicons, and English compliance
phrase banks. A 300-entry lexicon of the words that actually occur in support
calls fixes the majority of those cases for zero inference cost, and gives the
neural model a prior to fuse with rather than replacing it.

Each entry maps canonical romanisation -> (Devanagari, spelling variants).
Variants matter enormously: agents and ASR both produce `nahi / nahin / nhi /
nahee`, and a lexicon that only knows one spelling is worse than useless because
it produces *confidently wrong* labels on the others.
"""

from __future__ import annotations

from .script import strip_diacritics

# canonical -> (devanagari, [surface variants])
_HINDI_LEXICON: dict[str, tuple[str, list[str]]] = {
    # --- pronouns / function words -----------------------------------------
    "main": ("मैं", ["mai", "mein", "me1", "man"]),
    "aap": ("आप", ["ap", "aapko", "apko", "aapka", "apka", "aapki", "apki", "aapne", "apne"]),
    "hum": ("हम", ["ham", "humko", "hamko", "humara", "hamara", "humein", "hamein"]),
    "tum": ("तुम", ["tumhe", "tumhara", "tumko"]),
    "yeh": ("यह", ["ye", "yah", "yeh"]),
    "woh": ("वह", ["wo", "vah", "voh", "wah"]),
    "kya": ("क्या", ["kia", "kyaa", "kyaa"]),
    "kyun": ("क्यों", ["kyu", "kyon", "kyun", "kiu"]),
    "kaise": ("कैसे", ["kese", "kaisay", "kaisa", "kaisi"]),
    "kab": ("कब", ["kab"]),
    "kahan": ("कहाँ", ["kaha", "kahan", "kahaan"]),
    "kaun": ("कौन", ["kon", "kaun", "kaunsa", "konsa"]),
    "kitna": ("कितना", ["kitni", "kitne", "kitna", "kitnaa"]),
    "nahi": ("नहीं", ["nahin", "nhi", "nahee", "nahii", "nai", "nahiin"]),
    "haan": ("हाँ", ["han", "haa", "haanji", "hanji", "ji"]),
    "aur": ("और", ["or", "aur", "our"]),
    "lekin": ("लेकिन", ["lekin", "magar", "par"]),
    "abhi": ("अभी", ["abhi", "abi"]),
    "phir": ("फिर", ["fir", "phir", "phr"]),
    "bhi": ("भी", ["bhi", "bi"]),
    "hi": ("ही", ["hii"]),
    "sab": ("सब", ["sabh", "sabhi"]),
    "kuch": ("कुछ", ["kuchh", "kch", "kuch"]),
    "koi": ("कोई", ["koi", "koyi"]),
    "bahut": ("बहुत", ["bohot", "bahot", "bhut", "bahut", "buhut"]),
    "thoda": ("थोड़ा", ["thora", "thoda", "thodi", "thodasa"]),
    "jaldi": ("जल्दी", ["jaldi", "jldi"]),
    "turant": ("तुरंत", ["turant", "turnt"]),
    "bilkul": ("बिल्कुल", ["bilkul", "bilkool"]),
    "matlab": ("मतलब", ["matlab", "mtlb"]),
    "sirf": ("सिर्फ़", ["sirf", "srf", "keval"]),
    "wapas": ("वापस", ["wapas", "vapas", "wapis"]),
    "dobara": ("दोबारा", ["dubara", "dobara", "dubra"]),
    # --- verbs --------------------------------------------------------------
    "hai": ("है", ["hai", "hay", "he"]),
    "hain": ("हैं", ["hain", "hein", "h"]),
    "tha": ("था", ["tha", "thi", "the"]),
    "hoga": ("होगा", ["hoga", "hogi", "honge", "hogaa"]),
    "karo": ("करो", ["karo", "kro", "kariye", "karein", "kijiye", "kijiyega"]),
    "kar": ("कर", ["kar", "kr", "karke", "karna", "karni", "karne", "karta", "karti", "karte"]),
    "hua": ("हुआ", ["hua", "hui", "huye", "hue"]),
    "diya": ("दिया", ["diya", "diyaa", "di", "dena", "dijiye", "de"]),
    "liya": ("लिया", ["liya", "li", "lena", "lijiye", "le"]),
    "mila": ("मिला", ["mila", "mili", "milta", "milti", "milega", "milegi", "milne"]),
    "gaya": ("गया", ["gaya", "gayi", "gaye", "gya"]),
    "raha": ("रहा", ["raha", "rahi", "rahe", "rha"]),
    "bola": ("बोला", ["bola", "boli", "bole", "bolna", "boliye", "bol"]),
    "kaha": ("कहा", ["kaha", "kehna", "kehta", "kehte", "kahiye"]),
    "samajh": ("समझ", ["samajh", "smjh", "samjha", "samjhi", "samjhaya", "samajhna"]),
    "dekh": ("देख", ["dekh", "dekha", "dekhiye", "dekhna", "dekhte", "dekhkar"]),
    "sun": ("सुन", ["sun", "suna", "suniye", "sunna", "sunte"]),
    "chahiye": ("चाहिए", ["chahiye", "chahie", "chaiye", "chahta", "chahti", "chahte"]),
    "sakta": ("सकता", ["sakta", "sakti", "sakte", "sakoge", "sakenge"]),
    "paye": ("पाए", ["paya", "payi", "paye", "pana"]),
    "lagta": ("लगता", ["lagta", "lagti", "lagte", "laga", "lagi", "lage", "lagega"]),
    "hone": ("होने", ["hone", "hona", "honi", "ho"]),
    "aaya": ("आया", ["aaya", "aayi", "aye", "aana", "aayega", "ayega", "aa"]),
    "jaana": ("जाना", ["jana", "jaana", "jayega", "jaayega", "jaiye", "ja"]),
    "band": ("बंद", ["band", "bnd", "bandh"]),
    "chalu": ("चालू", ["chalu", "chaalu", "chal", "chalta", "chalti", "chala"]),
    "ruk": ("रुक", ["ruk", "ruka", "rukiye", "rukna", "rukega"]),
    "bhej": ("भेज", ["bhej", "bheja", "bhejiye", "bhejna", "bhejo"]),
    "check": ("चेक", ["chek"]),  # English loanword, but Hindi-inflected in practice
    # --- nouns: money / billing --------------------------------------------
    "paisa": ("पैसा", ["paise", "paisa", "paisay", "pese"]),
    "rupaye": ("रुपये", ["rupay", "rupaye", "rupee", "rupaiya", "rs"]),
    "bill": ("बिल", ["bil"]),
    "bakaya": ("बकाया", ["bakaya", "bakaaya"]),
    "kaat": ("काट", ["kat", "kata", "kaat", "kate", "kategi", "katautee", "kataute"]),
    "wapsi": ("वापसी", ["wapsi", "vapsi", "refund"]),
    "jama": ("जमा", ["jama", "jamaa"]),
    "khata": ("खाता", ["khata", "khaata", "khate"]),
    "shulk": ("शुल्क", ["shulk", "charge"]),
    "galat": ("ग़लत", ["galat", "galath", "glt"]),
    "sahi": ("सही", ["sahi", "shi", "thik", "theek"]),
    # --- nouns: service / support ------------------------------------------
    "samasya": ("समस्या", ["samasya", "smasya", "problem", "dikkat", "dikkat"]),
    "dikkat": ("दिक्कत", ["dikkat", "dikat", "dikkaten"]),
    "shikayat": ("शिकायत", ["shikayat", "shikaayat", "complaint"]),
    "seva": ("सेवा", ["seva", "sewa"]),
    "sipahi": ("सिपाही", ["sipahi"]),
    "adhikari": ("अधिकारी", ["adhikari", "officer", "senior"]),
    "manager": ("मैनेजर", ["manejar", "mainejar"]),
    "number": ("नंबर", ["nambar", "numbar"]),
    "connection": ("कनेक्शन", ["kanekshan"]),
    "network": ("नेटवर्क", ["netwrk", "netvark"]),
    "recharge": ("रिचार्ज", ["richarj"]),
    "plan": ("प्लान", ["plaan"]),
    "data": ("डेटा", ["deta"]),
    "speed": ("स्पीड", ["spid"]),
    "signal": ("सिग्नल", ["signl"]),
    "order": ("ऑर्डर", ["ardar"]),
    "delivery": ("डिलीवरी", ["dilivery", "dilivari"]),
    "samay": ("समय", ["samay", "time", "waqt", "vakt"]),
    "din": ("दिन", ["din", "dinn"]),
    "hafta": ("हफ़्ता", ["hafta", "hafte", "haftey"]),
    "mahina": ("महीना", ["mahina", "mahine", "maheena"]),
    "ghanta": ("घंटा", ["ghanta", "ghante", "ganta"]),
    "kal": ("कल", ["kal", "kl"]),
    "aaj": ("आज", ["aaj", "aj"]),
    "baar": ("बार", ["bar", "baar", "baar"]),
    "aakhri": ("आख़िरी", ["aakhri", "akhri", "last"]),
    # --- politeness / escalation -------------------------------------------
    "kripya": ("कृपया", ["kripya", "kripaya", "please"]),
    "dhanyavaad": ("धन्यवाद", ["dhanyavad", "dhanyawad", "shukriya", "shukria"]),
    "maaf": ("माफ़", ["maaf", "maf", "maafi", "mafi", "sorry"]),
    "pareshan": ("परेशान", ["pareshan", "preshan", "pareshaan"]),
    "gussa": ("ग़ुस्सा", ["gussa", "gusa"]),
    "naraz": ("नाराज़", ["naraz", "naaraz", "naraaz"]),
    "band_karo": ("बंद करो", ["bandkaro"]),
    "faida": ("फ़ायदा", ["faida", "fayda"]),
    "nuksan": ("नुक़सान", ["nuksan", "nuqsan", "nukshan"]),
    "kanoon": ("क़ानून", ["kanoon", "kanun", "legal"]),
    "court": ("कोर्ट", ["kort", "adalat"]),
    "upbhokta": ("उपभोक्ता", ["upbhokta", "consumer"]),
    "forum": ("फ़ोरम", ["foram"]),
    "senior": ("सीनियर", ["siniyar"]),
    "baat": ("बात", ["bat", "baat", "baatein", "baaten"]),
    "sahayata": ("सहायता", ["sahayata", "madad", "help"]),
    "madad": ("मदद", ["madat", "madad"]),
    "intezaar": ("इंतज़ार", ["intezar", "intzar", "intezaar", "wait"]),
    "pichle": ("पिछले", ["pichle", "pichhle", "pichla"]),
    "aisa": ("ऐसा", ["aisa", "aise", "esa", "ese"]),
    "itna": ("इतना", ["itna", "itni", "itne"]),
    "zyada": ("ज़्यादा", ["zyada", "jyada", "zada", "jada"]),
    "kam": ("कम", ["kam", "km"]),
    "theek": ("ठीक", ["thik", "theek", "tik"]),
    "achha": ("अच्छा", ["acha", "achha", "accha", "acchi", "achhi"]),
    "bura": ("बुरा", ["bura", "buri", "kharab", "kharaab"]),
    "kharab": ("ख़राब", ["kharab", "kharaab", "krb"]),
    "record": ("रिकॉर्ड", ["rikord", "rekord"]),
    "call": ("कॉल", ["kal2", "kall"]),
    "sunwai": ("सुनवाई", ["sunwai", "sunvai"]),
}

# Latin-script tokens that are unambiguously English in a support-call context.
# Used as a *negative* signal so `is / us / to / me` are not grabbed by the
# Hindi lexicon's homograph entries.
_ENGLISH_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "to", "of", "in", "on", "at", "for", "with", "from", "by", "about", "into",
    "and", "or", "but", "if", "so", "because", "as", "than", "then", "there",
    "not", "no", "yes", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "do", "does", "did", "have", "has", "had",
    "what", "when", "where", "which", "who", "whom", "why", "how",
    "sir", "madam", "please", "thank", "thanks", "sorry", "okay", "ok",
    "account", "payment", "amount", "charge", "charged", "refund", "bill",
    "billing", "issue", "problem", "service", "customer", "support", "help",
    "call", "calling", "called", "number", "mobile", "phone", "email",
    "internet", "network", "signal", "speed", "data", "plan", "recharge",
    "order", "delivery", "delivered", "cancel", "cancelled", "complaint",
    "manager", "supervisor", "escalate", "escalation", "team", "department",
    "minutes", "hours", "days", "week", "month", "today", "tomorrow", "yesterday",
    "check", "checking", "confirm", "confirmed", "update", "updated", "status",
    "sure", "right", "fine", "good", "bad", "very", "really", "just", "only",
    "recorded", "recording", "quality", "purposes", "training", "monitored",
    "verify", "verification", "security", "otp", "password", "card", "credit",
    "debit", "bank", "transaction", "failed", "successful", "pending", "again",
    "still", "already", "yet", "now", "soon", "wait", "waiting", "hold",
    "system", "technical", "error", "reset", "restart", "working", "work",
}

# Tokens that are valid words in *both* languages. Never resolved by lookup —
# always sent to neighbour-context voting in lid.py.
_HOMOGRAPHS: set[str] = {
    "to", "me", "is", "us", "the", "he", "do", "so", "in", "at", "bat", "hi",
    "kar", "car", "par", "ban", "man", "din", "sun", "un", "an", "ha", "na",
    "kam", "am", "up", "bahar", "sab", "tab", "jab", "ab", "ye", "we", "or",
    "are", "chip", "lag", "gay", "way", "bag", "big", "log", "mall", "call",
}

_VARIANT_TO_CANONICAL: dict[str, str] = {}
_VARIANT_TO_DEVANAGARI: dict[str, str] = {}

for _canon, (_deva, _variants) in _HINDI_LEXICON.items():
    for _surface in {_canon, *_variants}:
        _key = strip_diacritics(_surface).lower()
        _VARIANT_TO_CANONICAL.setdefault(_key, _canon)
        _VARIANT_TO_DEVANAGARI.setdefault(_key, _deva)


def normalize_key(token: str) -> str:
    return strip_diacritics(token).lower().strip("'’-")


def is_romanized_hindi(token: str) -> bool:
    """Confident lexicon hit, excluding homographs."""
    key = normalize_key(token)
    return key in _VARIANT_TO_CANONICAL and key not in _HOMOGRAPHS


def is_english_word(token: str) -> bool:
    key = normalize_key(token)
    return key in _ENGLISH_STOPWORDS and key not in _HOMOGRAPHS


def is_homograph(token: str) -> bool:
    return normalize_key(token) in _HOMOGRAPHS


def canonical_form(token: str) -> str | None:
    return _VARIANT_TO_CANONICAL.get(normalize_key(token))


def to_devanagari(token: str) -> str | None:
    return _VARIANT_TO_DEVANAGARI.get(normalize_key(token))


def lexicon_size() -> int:
    return len(_VARIANT_TO_CANONICAL)
