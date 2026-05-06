#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tti.py — TextToIntent  ·  Version 3.1.0
=======================================================================
Composant TTI (Text-To-Intent) du système ASE.
Pipeline NLU français → Intent(s) structuré(s).

CHANGELOG
  3.1.0  Fusion V2.0 + V3.0.
         - Conservation de toutes les fonctionnalités V2.0:
           multi-clauses, ConversationFrame, ConversationContext,
           FrenchTemporalResolver complet, EventGraph, PipelineMetrics,
           RegisterAnalyzer complet, SyntacticTree, emotional_tone
         - Ajout des améliorations V3.0:
           Propagation xcomp (commandes polies),
           Table de routage (Strategy Pattern),
           Logs DEBUG détaillés,
           Chain of Responsibility pour correcteurs

Raspberry Pi 5, 100 % offline. Zéro modèle requis.
"""

import datetime
import json
import logging
import re
import threading
import time
import argparse
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Optional, Tuple, Any, Set, Union

import numpy as np

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-14s] %(levelname)-8s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TTI")
debug_logger = logging.getLogger("TTI.Debug")
debug_logger.setLevel(logging.DEBUG)
_dh = logging.StreamHandler()
_dh.setFormatter(logging.Formatter("%(asctime)s [DEBUG] %(message)s", datefmt="%H:%M:%S.%f")[:-3])
debug_logger.addHandler(_dh)
debug_logger.propagate = False

error_logger = logging.getLogger("TTI.Errors")
_err_handler = logging.FileHandler("tti_errors.log")
_err_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
error_logger.addHandler(_err_handler)
error_logger.setLevel(logging.WARNING)

# ============================================================
# Imports optionnels
# ============================================================

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False

try:
    from workalendar.europe import France as FranceCalendar
    WORKALENDAR_AVAILABLE = True
except ImportError:
    WORKALENDAR_AVAILABLE = False

try:
    import holidays as _holidays_unused
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

try:
    import Levenshtein as _lev
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

try:
    from vacances_scolaires import get_holidays as get_school_holidays
    VACANCES_AVAILABLE = True
except ImportError:
    VACANCES_AVAILABLE = False

# ============================================================
# Types stricts
# ============================================================

IntentType = Literal[
    "query_narrative", "query_state", "query_intention",
    "action_device", "information_input", "chit_chat",
    "memorize", "recall",
    "command_motor", "command_sense", "command_actuator",
    "teach"
]

TenseType = Literal["past", "present", "future", "conditional", "unknown"]
PersonType = Literal["1st_sg", "2nd_sg", "3rd_sg", "1st_pl", "2nd_pl", "3rd_pl", "unknown"]
MoodType = Literal["indicative", "subjunctive", "imperative", "conditional", "infinitive"]
PolarityType = Literal["positive", "negative"]
ModalType = Literal["obligation", "possibility", "volition", "wish", None]
ConceptType = Literal[
    "PERCEIVE", "COMMUNICATE", "BE", "MOVE", "INGEST", "CREATE",
    "TRANSFER", "COGNITION", "EMOTION", "SOCIAL_ACT", "HEALTH",
    "MOTOR_CMD", "SENSE_FOCUS", "ACTUATOR_CMD", "MEMORIZE", "RECALL",
    "TEACH", None
]
ScopeType = Literal["PAST", "PRESENT", "FUTURE", "HYPOTHETICAL", "UNKNOWN"]
RoleType = Literal["user", "ase", "other"]

# ============================================================
# Configuration
# ============================================================

SPACY_MODEL = "fr_core_news_sm"
LANGUAGE = "fr"
CONFIDENCE_THRESHOLD = 0.60
CONTEXT_RESET_SEC = 30.0
CACHE_SIZE = 256

NER_TYPE_MAP = {
    "CARDINAL": "number", "ORDINAL": "ordinal", "QUANTITY": "quantity",
    "PERCENT": "percent", "MONEY": "money", "TIME": "time_ref",
    "DATE": "date_ref", "DURATION": "duration", "NORP": "group",
    "FAC": "facility", "ORG": "organization", "PER": "person",
    "LOC": "location", "GPE": "location", "PRODUCT": "product",
    "EVENT": "event", "LANGUAGE": "language"
}

UNIT_PATTERNS = {
    "km": "km", "kilometre": "km", "kilometres": "km",
    "m": "m", "metre": "m", "metres": "m", "cm": "cm",
    "kg": "kg", "kilo": "kg", "kilos": "kg",
    "g": "g", "gramme": "g", "grammes": "g",
    "l": "L", "litre": "L", "litres": "L", "cl": "cL", "ml": "mL",
    "h": "h", "heure": "h", "heures": "h",
    "min": "min", "minute": "min", "minutes": "min",
    "s": "s", "seconde": "s", "secondes": "s",
    "degre": "celsius", "degres": "celsius",
    "euro": "EUR", "euros": "EUR", "%": "%",
}

INFO_SUBJECTS = {"je", "j'", "j", "moi", "nous", "on"}

CLITIC_TO_PERSON: Dict[str, PersonType] = {
    "me": "1st_sg", "m'": "1st_sg", "moi": "1st_sg", "m": "1st_sg",
    "te": "2nd_sg", "t'": "2nd_sg", "toi": "2nd_sg", "t": "2nd_sg",
    "se": "3rd_sg", "s'": "3rd_sg", "soi": "3rd_sg", "s": "3rd_sg",
    "lui": "3rd_sg", "elle": "3rd_sg",
    "nous": "1st_pl", "on": "1st_pl",
    "vous": "2nd_pl",
    "leur": "3rd_pl", "eux": "3rd_pl", "elles": "3rd_pl",
}

MODAL_MARKERS: Dict[str, ModalType] = {
    "devoir": "obligation", "falloir": "obligation", "obliger": "obligation",
    "pouvoir": "possibility",
    "vouloir": "volition", "souhaiter": "volition", "désirer": "volition", "espérer": "volition",
    "aimer": "wish",
}

TENSE_TO_SCOPE: Dict[str, ScopeType] = {
    "past": "PAST", "present": "PRESENT", "future": "FUTURE",
    "conditional": "HYPOTHETICAL", "unknown": "UNKNOWN",
}

ACTION_VERBS: Dict[str, str] = {
    "allumer": "turn_on", "activer": "turn_on", "lancer": "turn_on",
    "ouvrir": "turn_on", "démarrer": "turn_on",
    "éteindre": "turn_off", "couper": "turn_off", "désactiver": "turn_off",
    "arrêter": "turn_off", "fermer": "turn_off", "stopper": "turn_off",
    "monter": "set_up", "augmenter": "set_up",
    "baisser": "set_down", "diminuer": "set_down", "réduire": "set_down",
    "régler": "set", "mettre": "set", "configurer": "set", "programmer": "set",
}

DEVICE_NOUNS: Dict[str, str] = {
    "lumière": "light", "lampe": "light", "éclairage": "light",
    "plafonnier": "light", "spot": "light",
    "télé": "tv", "télévision": "tv", "écran": "tv",
    "musique": "music", "son": "audio", "volume": "audio",
    "chauffage": "heating", "radiateur": "heating", "thermostat": "heating",
    "volet": "shutter", "store": "shutter", "rideau": "shutter",
    "porte": "door", "fenêtre": "door",
    "alarme": "alarm", "climatisation": "ac", "clim": "ac", "ventilateur": "fan",
}

# ============================================================
# SEMANTIC_MAP (verbes → concepts)
# ============================================================

SEMANTIC_MAP: Dict[str, ConceptType] = {
    # PERCEIVE
    "voir": "PERCEIVE", "regarder": "PERCEIVE", "entendre": "PERCEIVE",
    "écouter": "PERCEIVE", "sentir": "PERCEIVE", "remarquer": "PERCEIVE",
    "observer": "PERCEIVE", "apercevoir": "PERCEIVE", "percevoir": "PERCEIVE",
    "distinguer": "PERCEIVE", "noter": "PERCEIVE", "détecter": "PERCEIVE",
    "repérer": "PERCEIVE", "surveiller": "PERCEIVE", "examiner": "PERCEIVE",
    "inspecter": "PERCEIVE", "scruter": "PERCEIVE",
    # COMMUNICATE
    "dire": "COMMUNICATE", "parler": "COMMUNICATE", "appeler": "COMMUNICATE",
    "répondre": "COMMUNICATE", "demander": "COMMUNICATE", "expliquer": "COMMUNICATE",
    "raconter": "COMMUNICATE", "discuter": "COMMUNICATE", "écrire": "COMMUNICATE",
    "communiquer": "COMMUNICATE", "téléphoner": "COMMUNICATE", "annoncer": "COMMUNICATE",
    "informer": "COMMUNICATE", "mentionner": "COMMUNICATE", "préciser": "COMMUNICATE",
    "signaler": "COMMUNICATE", "crier": "COMMUNICATE", "chuchoter": "COMMUNICATE",
    "réclamer": "COMMUNICATE", "négocier": "COMMUNICATE", "interviewer": "COMMUNICATE",
    "interroger": "COMMUNICATE", "contacter": "COMMUNICATE", "textoter": "COMMUNICATE",
    # BE
    "être": "BE", "rester": "BE", "demeurer": "BE", "trouver": "BE", "exister": "BE",
    "sembler": "BE", "paraître": "BE", "devenir": "BE", "apparaître": "BE", "s'avérer": "BE",
    # MOVE
    "aller": "MOVE", "venir": "MOVE", "bouger": "MOVE", "sortir": "MOVE", "entrer": "MOVE",
    "partir": "MOVE", "arriver": "MOVE", "quitter": "MOVE", "marcher": "MOVE",
    "courir": "MOVE", "rentrer": "MOVE", "revenir": "MOVE", "monter": "MOVE",
    "descendre": "MOVE", "passer": "MOVE", "traverser": "MOVE", "déplacer": "MOVE",
    "voyager": "MOVE", "circuler": "MOVE", "conduire": "MOVE", "rouler": "MOVE",
    "voler": "MOVE", "nager": "MOVE", "grimper": "MOVE", "reculer": "MOVE",
    "avancer": "MOVE", "fuir": "MOVE", "s'approcher": "MOVE", "s'éloigner": "MOVE",
    # INGEST
    "manger": "INGEST", "boire": "INGEST", "avaler": "INGEST", "consommer": "INGEST",
    "goûter": "INGEST", "grignoter": "INGEST", "déguster": "INGEST", "dîner": "INGEST",
    "déjeuner": "INGEST", "cuisiner": "INGEST", "préparer": "INGEST",
    "ingérer": "INGEST", "absorber": "INGEST",
    # CREATE
    "faire": "CREATE", "fabriquer": "CREATE", "construire": "CREATE", "créer": "CREATE",
    "produire": "CREATE", "réaliser": "CREATE", "concevoir": "CREATE", "dessiner": "CREATE",
    "peindre": "CREATE", "sculpter": "CREATE", "composer": "CREATE", "rédiger": "CREATE",
    "coder": "CREATE", "programmer": "CREATE", "bricoler": "CREATE", "tricoter": "CREATE",
    "coudre": "CREATE", "jardiner": "CREATE", "planter": "CREATE", "aménager": "CREATE",
    # TRANSFER
    "donner": "TRANSFER", "envoyer": "TRANSFER", "recevoir": "TRANSFER",
    "apporter": "TRANSFER", "remettre": "TRANSFER", "transmettre": "TRANSFER",
    "offrir": "TRANSFER", "prêter": "TRANSFER", "emprunter": "TRANSFER",
    "rendre": "TRANSFER", "poster": "TRANSFER", "livrer": "TRANSFER",
    "expédier": "TRANSFER", "partager": "TRANSFER", "distribuer": "TRANSFER",
    "vendre": "TRANSFER", "acheter": "TRANSFER",
    # COGNITION
    "penser": "COGNITION", "croire": "COGNITION", "savoir": "COGNITION",
    "comprendre": "COGNITION", "oublier": "COGNITION", "apprendre": "COGNITION",
    "réfléchir": "COGNITION", "imaginer": "COGNITION", "supposer": "COGNITION",
    "douter": "COGNITION", "ignorer": "COGNITION", "réaliser": "COGNITION",
    "considérer": "COGNITION", "estimer": "COGNITION", "juger": "COGNITION",
    "analyser": "COGNITION", "décider": "COGNITION", "choisir": "COGNITION",
    "planifier": "COGNITION", "mémoriser": "COGNITION", "se souvenir": "COGNITION",
    "calculer": "COGNITION", "étudier": "COGNITION", "lire": "COGNITION",
    # EMOTION
    "aimer": "EMOTION", "adorer": "EMOTION", "détester": "EMOTION",
    "craindre": "EMOTION", "espérer": "EMOTION", "vouloir": "EMOTION",
    "désirer": "EMOTION", "regretter": "EMOTION", "souffrir": "EMOTION",
    "pleurer": "EMOTION", "rire": "EMOTION", "s'énerver": "EMOTION",
    "s'inquiéter": "EMOTION", "apprécier": "EMOTION", "ressentir": "EMOTION",
    "se sentir": "EMOTION", "se réjouir": "EMOTION", "se plaindre": "EMOTION",
    "s'ennuyer": "EMOTION", "stresser": "EMOTION",
    # SOCIAL_ACT
    "saluer": "SOCIAL_ACT", "remercier": "SOCIAL_ACT", "s'excuser": "SOCIAL_ACT",
    "promettre": "SOCIAL_ACT", "féliciter": "SOCIAL_ACT", "inviter": "SOCIAL_ACT",
    "refuser": "SOCIAL_ACT", "accepter": "SOCIAL_ACT", "proposer": "SOCIAL_ACT",
    "voter": "SOCIAL_ACT", "signer": "SOCIAL_ACT", "jurer": "SOCIAL_ACT",
    "rencontrer": "SOCIAL_ACT", "retrouver": "SOCIAL_ACT",
    "présenter": "SOCIAL_ACT", "accueillir": "SOCIAL_ACT",
    # HEALTH
    "dormir": "HEALTH", "se reposer": "HEALTH", "tomber": "HEALTH",
    "se blesser": "HEALTH", "soigner": "HEALTH", "guérir": "HEALTH",
    "opérer": "HEALTH", "consulter": "HEALTH", "vacciner": "HEALTH",
    "exercer": "HEALTH", "respirer": "HEALTH", "tousser": "HEALTH",
    "éternuer": "HEALTH", "se rétablir": "HEALTH", "transpirer": "HEALTH",
    # MOTOR_CMD
    "avancer": "MOTOR_CMD", "reculer": "MOTOR_CMD", "tourner": "MOTOR_CMD",
    "pivoter": "MOTOR_CMD", "lever": "MOTOR_CMD", "baisser": "MOTOR_CMD",
    "plier": "MOTOR_CMD", "tendre": "MOTOR_CMD", "incliner": "MOTOR_CMD",
    "pencher": "MOTOR_CMD", "étendre": "MOTOR_CMD", "replier": "MOTOR_CMD",
    "immobiliser": "MOTOR_CMD", "s'arrêter": "MOTOR_CMD", "s'asseoir": "MOTOR_CMD",
    "se lever": "MOTOR_CMD", "se mettre": "MOTOR_CMD", "saisir": "MOTOR_CMD",
    "lâcher": "MOTOR_CMD", "serrer": "MOTOR_CMD", "poser": "MOTOR_CMD",
    # SENSE_FOCUS
    "regarder": "SENSE_FOCUS", "fixer": "SENSE_FOCUS", "scruter": "SENSE_FOCUS",
    "surveiller": "SENSE_FOCUS", "scanner": "SENSE_FOCUS", "écouter": "SENSE_FOCUS",
    "percevoir": "SENSE_FOCUS", "détecter": "SENSE_FOCUS", "vocaliser": "SENSE_FOCUS",
    "articuler": "SENSE_FOCUS", "prononcer": "SENSE_FOCUS",
    # ACTUATOR_CMD
    "allumer": "ACTUATOR_CMD", "éteindre": "ACTUATOR_CMD", "éclairer": "ACTUATOR_CMD",
    "illuminer": "ACTUATOR_CMD", "activer": "ACTUATOR_CMD", "désactiver": "ACTUATOR_CMD",
    "augmenter": "ACTUATOR_CMD", "diminuer": "ACTUATOR_CMD", "régler": "ACTUATOR_CMD",
    "couper": "ACTUATOR_CMD", "démarrer": "ACTUATOR_CMD", "stopper": "ACTUATOR_CMD",
    "lancer": "ACTUATOR_CMD",
    # MEMORIZE / RECALL
    "mémoriser": "MEMORIZE", "retenir": "MEMORIZE", "enregistrer": "MEMORIZE",
    "noter": "MEMORIZE", "stocker": "MEMORIZE", "garder": "MEMORIZE", "graver": "MEMORIZE",
    "se souvenir": "RECALL", "rappeler": "RECALL", "restituer": "RECALL",
    "retrouver": "RECALL", "oublier": "RECALL",
    # TEACH
    "enseigner": "TEACH", "montrer": "TEACH", "définir": "TEACH",
    "nommer": "TEACH", "désigner": "TEACH",
}

# ============================================================
# Lexiques pour classification
# ============================================================

CHIT_CHAT_WORDS = frozenset({
    "bonjour", "salut", "coucou", "bonsoir", "bye", "merci", "svp",
    "bravo", "félicitations", "chapeau", "super", "génial"
})

CHIT_CHAT_PHRASES = (
    "bonne nuit", "au revoir", "s'il vous plaît", "s'il te plaît",
    "ça va", "comment vas", "comment allez"
)

INTERROGATIVE_WORDS = frozenset({
    "quel", "quelle", "quels", "quelles", "quoi", "comment",
    "pourquoi", "quand", "où", "qui", "combien", "lequel", "laquelle",
    "lesquels", "lesquelles", "est-ce", "est ce"
})

MEMORIZE_TRIGGERS = frozenset({
    "retiens", "mémorise", "souviens-toi", "enregistre", "note", "stocke",
    "garde en mémoire", "souviens que", "rappelle-toi que",
    "je veux que tu retiennes", "ajoute à tes souvenirs"
})

RECALL_TRIGGERS = frozenset({
    "rappelle-moi", "qu'est-ce que j'ai dit", "je t'ai dit quoi",
    "redites-moi", "redis-moi", "je me souviens plus",
    "tu te souviens", "est-ce que je t'ai dit", "je t'avais dit",
    "qu'est-ce que tu as retenu", "dis-moi ce que"
})

MODAL_VERBS = {"pouvoir", "devoir", "vouloir", "falloir"}

# ============================================================
# 1. FrenchTextCorrector (avec phonétique)
# ============================================================

COMMON_WORDS_FR = {
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais",
    "donc", "car", "est", "sont", "était", "ai", "as", "a", "avons", "avez", "ont",
    "vais", "vas", "va", "peux", "peut", "veux", "veut", "fais", "fait", "dis", "dit",
    "pour", "par", "avec", "sans", "dans", "sur", "sous",
}


class FrenchTextCorrector:
    """Normalise le texte avant analyse NLU. SMS, dyslexie, fautes phonétiques."""

    COMMON_MISTAKES: Dict[str, str] = {
        "sava": "ça va", "cetais": "c'était", "cété": "c'était",
        "jsp": "je ne sais pas", "pk": "pourquoi", "pcq": "parce que",
        "dc": "donc", "tkt": "ne t'en fais pas", "chui": "je suis",
        "ajd": "aujourd'hui", "demin": "demain", "yer": "hier",
        "vien": "viens", "veu": "veux", "peu": "peux", "fé": "fais",
        "pa": "pas", "jamait": "jamais", "mes": "mais", "don": "donc",
        "dan": "dans", "aveq": "avec", "tré": "très",
        "jé mangé": "j'ai mangé",
    }

    _CLEANUP = [
        (re.compile(r'\s+([?.!,;:])'), r'\1'),
        (re.compile(r'([?.!,;:])([^\s])'), r'\1 \2'),
        (re.compile(r'\s+'), ' '),
    ]

    _cache: Dict[str, Tuple[str, List[Tuple[str, str]]]] = {}

    @classmethod
    def correct(cls, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        if not text:
            return text, []
        if text in cls._cache:
            return cls._cache[text]

        corrections: List[Tuple[str, str]] = []
        words = text.lower().split()
        out: List[str] = []
        i = 0

        while i < len(words):
            matched = False
            for error, fix in cls.COMMON_MISTAKES.items():
                err_words = error.split()
                n = len(err_words)
                if words[i:i + n] == err_words:
                    out.extend(fix.split())
                    corrections.append((error, fix))
                    i += n
                    matched = True
                    break
            if not matched:
                w = words[i]
                if LEVENSHTEIN_AVAILABLE and w not in COMMON_WORDS_FR and len(w) > 3:
                    threshold = 2 if len(w) > 5 else 1
                    best = min(COMMON_WORDS_FR, key=lambda x: _lev.distance(w, x))
                    if _lev.distance(w, best) <= threshold:
                        out.append(best)
                        corrections.append((w, best))
                    else:
                        out.append(w)
                else:
                    close = get_close_matches(w, COMMON_WORDS_FR, n=1, cutoff=0.75)
                    if close and w not in COMMON_WORDS_FR and len(w) > 2:
                        out.append(close[0])
                        corrections.append((w, close[0]))
                    else:
                        out.append(w)
                i += 1

        result = " ".join(out)
        for pat, repl in cls._CLEANUP:
            result = pat.sub(repl, result)
        result = result.strip()
        cls._cache[text] = (result, corrections)
        return result, corrections


# ============================================================
# 2. FrenchVerbAnalyzer
# ============================================================

class FrenchVerbAnalyzer:
    """Enrichit l'analyse verbale spaCy : irréguliers, formes composées, pronominal."""

    IRREGULAR: Dict[str, Tuple[str, str, str]] = {
        "aller": ("vais/vas/va/allons/allez/vont", "allai", "allé"),
        "avoir": ("ai/as/a/avons/avez/ont", "eus", "eu"),
        "être": ("suis/es/est/sommes/êtes/sont", "fus", "été"),
        "faire": ("fais/fait/faisons/faites/font", "fis", "fait"),
        "dire": ("dis/dit/disons/dites/disent", "dis", "dit"),
        "venir": ("viens/vient/venons/venez/viennent", "vins", "venu"),
        "tenir": ("tiens/tient/tenons/tenez/tiennent", "tins", "tenu"),
        "voir": ("vois/voit/voyons/voyez/voient", "vis", "vu"),
        "savoir": ("sais/sait/savons/savez/savent", "sus", "su"),
        "pouvoir": ("peux/peut/pouvons/pouvez/peuvent", "pus", "pu"),
        "vouloir": ("veux/veut/voulons/voulez/veulent", "voulus", "voulu"),
        "devoir": ("dois/doit/devons/devez/doivent", "dus", "dû"),
        "falloir": ("faut", "fallut", "fallu"),
        "pleuvoir": ("pleut", "plut", "plu"),
    }

    _AUX_AVOIR = {"ai", "as", "a", "avons", "avez", "ont"}
    _AUX_ETRE = {"suis", "es", "est", "sommes", "êtes", "sont"}

    @classmethod
    def is_compound(cls, token_text: str) -> Optional[str]:
        t = token_text.lower()
        if t in cls._AUX_AVOIR:
            return "avoir"
        if t in cls._AUX_ETRE:
            return "être"
        return None

    @classmethod
    def is_pronominal(cls, doc, verb_token) -> bool:
        for child in verb_token.children:
            if child.dep_ in ("expl", "expl:comp") and child.text.lower() in (
                    "se", "s'", "me", "te", "nous", "vous"
            ):
                return True
        return False

    @classmethod
    def is_impersonal(cls, lemma: str) -> bool:
        return lemma in ("falloir", "pleuvoir", "neiger", "grêler", "tonner", "venter")

    @classmethod
    def refine_tense(cls, token_text: str, lemma: str, morph_tense: str) -> str:
        if lemma in cls.IRREGULAR:
            pres_forms = set(cls.IRREGULAR[lemma][0].split("/"))
            if token_text.lower() in pres_forms:
                return "present"
            if token_text.lower() == cls.IRREGULAR[lemma][1]:
                return "past"
        if re.search(r"(?:erai|eras|era|erons|erez|eront)$", token_text):
            return "future"
        if re.search(r"(?:erais|erait|erions|eriez|eraient)$", token_text):
            return "conditional"
        return morph_tense


# ============================================================
# 3. TemporalSpan + FrenchTemporalResolver (complet)
# ============================================================

@dataclass
class TemporalSpan:
    raw: str
    iso_start: str
    iso_end: str
    timex_type: str = "DATE"
    source: str = "dateparser"
    duration_raw: Optional[str] = None
    duration_unit: Optional[str] = None
    duration_value: Optional[float] = None
    until_raw: Optional[str] = None
    interval_start_raw: Optional[str] = None
    interval_end_raw: Optional[str] = None
    named_event: Optional[dict] = None


_DATE_PARSER = None
_TIMEXY_CHECKED = False
_TIMEXY_AVAILABLE = False


def _get_dateparser():
    global _DATE_PARSER
    if _DATE_PARSER is None:
        import dateparser as _dp
        _DATE_PARSER = _dp
    return _DATE_PARSER


def _check_timexy() -> bool:
    global _TIMEXY_CHECKED, _TIMEXY_AVAILABLE
    if not _TIMEXY_CHECKED:
        try:
            import timexy  # noqa
            _TIMEXY_AVAILABLE = True
        except ImportError:
            _TIMEXY_AVAILABLE = False
        _TIMEXY_CHECKED = True
    return _TIMEXY_AVAILABLE


class FrenchTemporalResolver:
    """
    Résout les expressions temporelles françaises.
    Couverture : moments, saisons, jours fériés, vacances scolaires,
    jours de semaine, jours relatifs, durées, intervals, dateparser.
    Cache LRU pour les expressions répétées.
    """

    MOMENTS = {
        "midi": (12, 12, "point"),
        "minuit": (0, 0, "point"),
        "matin": (6, 12, "interval"),
        "matinée": (6, 12, "interval"),
        "après-midi": (12, 18, "interval"),
        "soir": (18, 22, "interval"),
        "soirée": (18, 22, "interval"),
        "nuit": (22, 6, "interval"),
        "journée": (8, 20, "interval"),
        "aube": (5, 7, "interval"),
        "crépuscule": (18, 20, "interval"),
    }

    MOMENT_PATTERNS = {
        "midi": ["midi", "à midi", "12h"],
        "minuit": ["minuit", "à minuit", "0h"],
        "matin": ["matin", "ce matin", "dans la matinée"],
        "après-midi": ["après-midi", "cet après-midi"],
        "soir": ["soir", "ce soir", "dans la soirée", "ce soir"],
        "nuit": ["nuit", "cette nuit", "pendant la nuit"],
        "journée": ["journée", "cette journée", "toute la journée"],
    }

    SEASONS = {
        "printemps": {"months": [3, 4, 5], "emoji": "🌸"},
        "été": {"months": [6, 7, 8], "emoji": "☀️"},
        "automne": {"months": [9, 10, 11], "emoji": "🍂"},
        "hiver": {"months": [12, 1, 2], "emoji": "❄️"},
    }

    SEASON_PATTERNS = {
        "printemps": ["printemps", "au printemps", "ce printemps"],
        "été": ["été", "cet été", "en été"],
        "automne": ["automne", "cet automne", "en automne"],
        "hiver": ["hiver", "cet hiver", "en hiver"],
    }

    RELATIVE_DAYS = {
        "aujourd'hui": 0, "demain": 1, "après-demain": 2,
        "hier": -1, "avant-hier": -2,
    }

    WEEKDAYS = {
        "lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3,
        "vendredi": 4, "samedi": 5, "dimanche": 6,
    }

    HOLIDAY_NAMES = {
        "nouvel an": "01-01", "1er janvier": "01-01",
        "1er mai": "05-01", "fête du travail": "05-01",
        "8 mai": "05-08", "victoire": "05-08",
        "14 juillet": "07-14", "fête nationale": "07-14",
        "15 août": "08-15", "assomption": "08-15",
        "1er novembre": "11-01", "toussaint": "11-01",
        "11 novembre": "11-11", "armistice": "11-11",
        "noël": "12-25", "25 décembre": "12-25",
    }

    def __init__(self):
        self._calendar = FranceCalendar() if WORKALENDAR_AVAILABLE else None
        self._stats: Dict[str, int] = {"hits": 0, "misses": 0}

    @staticmethod
    def _duration_iso(start: datetime.datetime, end: datetime.datetime) -> str:
        sec = int((end - start).total_seconds())
        if sec % 86400 == 0:
            return f"P{sec // 86400}D"
        if sec % 3600 == 0:
            return f"PT{sec // 3600}H"
        if sec % 60 == 0:
            return f"PT{sec // 60}M"
        return f"PT{sec}S"

    @lru_cache(maxsize=CACHE_SIZE)
    def _cached_resolve(self, text: str, ref_iso: str) -> Optional[dict]:
        self._stats["misses"] += 1
        ref = datetime.datetime.fromisoformat(ref_iso)
        return self._resolve_uncached(text, ref)

    def resolve(self, text: str, ref: Optional[datetime.datetime] = None) -> Optional[dict]:
        if ref is None:
            ref = datetime.datetime.now()
        result = self._cached_resolve(text.lower(), ref.isoformat())
        if result:
            self._stats["hits"] += 1
        return result

    def _resolve_uncached(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        debug_logger.debug(f"[Temporal] Résolution de '{text[:40]}...'")
        year = ref.year
        for fn in (
            lambda: self._moment(text, ref),
            lambda: self._season(text, year),
            lambda: self._holiday(text, year),
            lambda: self._school_holiday(text, year),
            lambda: self._weekday(text, ref),
            lambda: self._relative_day(text, ref),
            lambda: self._duration(text),
            lambda: self._fallback_dateparser(text, ref),
        ):
            r = fn()
            if r:
                debug_logger.debug(f"[Temporal] Résolu: {r.get('timex_type')} via {r.get('source')}")
                return r
        return None

    def _moment(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        for name, patterns in self.MOMENT_PATTERNS.items():
            if not any(p in text for p in patterns):
                continue
            offset = (-2 if "avant-hier" in text else
                      -1 if "hier" in text else
                      2 if "après-demain" in text else
                      1 if "demain" in text else 0)
            base = ref + datetime.timedelta(days=offset)
            sh, eh, mtype = self.MOMENTS[name]
            named = {"name": name, "type": "moment_of_day", "day_offset": offset}
            if mtype == "point":
                dt = base.replace(hour=sh, minute=0, second=0, microsecond=0)
                iso = dt.isoformat()
                return {"raw": text, "timex_type": "TIME", "iso_start": iso, "iso_end": iso,
                        "source": "moment_rule", "named_event": named}
            start = base.replace(hour=sh, minute=0, second=0, microsecond=0)
            end = base.replace(hour=eh, minute=0, second=0, microsecond=0)
            if eh < sh:
                end += datetime.timedelta(days=1)
            return {"raw": text, "timex_type": "INTERVAL",
                    "iso_start": start.isoformat(), "iso_end": end.isoformat(),
                    "duration_iso": self._duration_iso(start, end),
                    "source": "moment_rule", "named_event": named}
        return None

    def _season(self, text: str, year: int) -> Optional[dict]:
        for season, patterns in self.SEASON_PATTERNS.items():
            if not any(p in text for p in patterns):
                continue
            delta = (1 if "prochain" in text or "prochaine" in text else
                     -1 if "dernier" in text or "dernière" in text or "passé" in text else 0)
            ty = year + delta
            months = self.SEASONS[season]["months"]
            if season == "hiver" and delta == 0:
                start = datetime.datetime(ty - 1, 12, 1)
                end = datetime.datetime(ty, 3, 1) - datetime.timedelta(days=1)
            else:
                start = datetime.datetime(ty, months[0], 1)
                last = months[-1]
                end_m = datetime.datetime(ty, last + 1, 1) if last < 12 else datetime.datetime(ty + 1, 1, 1)
                end = end_m - datetime.timedelta(days=1)
            named = {"name": season, "type": "season", "year": ty, "emoji": self.SEASONS[season]["emoji"]}
            return {"raw": text, "timex_type": "INTERVAL",
                    "iso_start": start.isoformat(), "iso_end": end.isoformat(),
                    "duration_iso": self._duration_iso(start, end),
                    "source": "season_rule", "named_event": named}
        return None

    def _holiday(self, text: str, year: int) -> Optional[dict]:
        for name, md in self.HOLIDAY_NAMES.items():
            if name in text:
                m, d = map(int, md.split("-"))
                dt = datetime.datetime(year, m, d)
                return {"raw": text, "timex_type": "DATE",
                        "iso_start": dt.isoformat(), "iso_end": dt.isoformat(),
                        "source": "holiday_rule",
                        "named_event": {"name": name, "type": "french_holiday"}}
        if ("paques" in text or "pâques" in text) and self._calendar:
            e = self._calendar.get_easter(year)
            dt = datetime.datetime(e.year, e.month, e.day)
            return {"raw": text, "timex_type": "DATE",
                    "iso_start": dt.isoformat(), "iso_end": dt.isoformat(),
                    "source": "workalendar",
                    "named_event": {"name": "Pâques", "type": "movable_holiday"}}
        return None

    def _school_holiday(self, text: str, year: int) -> Optional[dict]:
        if not VACANCES_AVAILABLE:
            return None
        holiday_map = {
            "vacances de noël": "Noël", "vacances d'hiver": "Hiver",
            "vacances de printemps": "Printemps", "vacances de pâques": "Printemps",
            "vacances d'été": "Été", "grandes vacances": "Été",
            "vacances de la toussaint": "Toussaint",
        }
        for key, hname in holiday_map.items():
            if key not in text:
                continue
            for zone in ("A", "B", "C"):
                try:
                    for h in get_school_holidays(year, zone):
                        if h["name"] == hname:
                            s = datetime.datetime.fromisoformat(h["start_date"])
                            e = datetime.datetime.fromisoformat(h["end_date"])
                            return {"raw": text, "timex_type": "INTERVAL",
                                    "iso_start": s.isoformat(), "iso_end": e.isoformat(),
                                    "duration_iso": self._duration_iso(s, e),
                                    "source": "vacances_scolaires",
                                    "named_event": {"name": hname, "type": "school_holiday", "zone": zone}}
                except Exception:
                    continue
        return None

    def _weekday(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        for day, wd in self.WEEKDAYS.items():
            if day not in text:
                continue
            ahead = (wd - ref.weekday()) % 7 or 7
            dt = ref + datetime.timedelta(days=ahead)
            s = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            e = dt.replace(hour=23, minute=59, second=59, microsecond=0)
            return {"raw": text, "timex_type": "DATE",
                    "iso_start": s.isoformat(), "iso_end": e.isoformat(),
                    "source": "weekday_rule",
                    "named_event": {"name": day, "type": "weekday"}}
        return None

    def _relative_day(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        for name, offset in self.RELATIVE_DAYS.items():
            if name not in text:
                continue
            dt = ref + datetime.timedelta(days=offset)
            s = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            e = dt.replace(hour=23, minute=59, second=59, microsecond=0)
            return {"raw": text, "timex_type": "DATE",
                    "iso_start": s.isoformat(), "iso_end": e.isoformat(),
                    "source": "relative_day_rule",
                    "named_event": {"name": name, "type": "relative_day", "offset": offset}}
        return None

    def _duration(self, text: str) -> Optional[dict]:
        patterns = [
            (r"pendant\s+(\d+(?:[.,]\d+)?)\s*(heures?|h)", 3600, "H"),
            (r"pendant\s+(\d+(?:[.,]\d+)?)\s*(minutes?|min)", 60, "M"),
            (r"pendant\s+(\d+(?:[.,]\d+)?)\s*(secondes?|s)", 1, "S"),
            (r"(\d+(?:[.,]\d+)?)\s*(jours?|j(?:\s|$))", 86400, "D"),
            (r"(\d+(?:[.,]\d+)?)\s*(semaines?|sem)", 604800, "W"),
        ]
        for pat, spu, unit in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(",", "."))
                total = val * spu
                return {"raw": m.group(0), "timex_type": "DURATION",
                        "iso_start": "", "iso_end": "",
                        "duration_value": val, "duration_unit": unit,
                        "duration_iso": f"PT{int(val)}{unit}",
                        "source": "duration_rule",
                        "named_event": {"type": "duration", "unit": unit}}
        return None

    def _fallback_dateparser(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        if not DATEPARSER_AVAILABLE:
            return None
        dp = _get_dateparser()
        try:
            dt = dp.parse(text, languages=[LANGUAGE],
                          settings={"RELATIVE_BASE": ref,
                                    "RETURN_AS_TIMEZONE_AWARE": True,
                                    "TIMEZONE": "Europe/Paris"})
            if dt:
                iso = dt.isoformat()
                return {"raw": text, "timex_type": "DATE",
                        "iso_start": iso, "iso_end": iso, "source": "dateparser"}
        except Exception:
            pass
        return None

    def to_temporal_span(self, d: dict, tense: str = "unknown") -> Optional[TemporalSpan]:
        if not d or not d.get("iso_start") and d.get("timex_type") != "DURATION":
            return None
        return TemporalSpan(
            raw=d.get("raw", ""),
            iso_start=d.get("iso_start", ""),
            iso_end=d.get("iso_end", ""),
            timex_type=d.get("timex_type", "DATE"),
            source=d.get("source", ""),
            duration_raw=d.get("raw") if d.get("timex_type") == "DURATION" else None,
            duration_unit=d.get("duration_unit"),
            duration_value=d.get("duration_value"),
            named_event=d.get("named_event"),
        )

    def get_stats(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "cache_hits": self._stats["hits"],
            "cache_misses": self._stats["misses"],
            "hit_rate": round(self._stats["hits"] / total, 3) if total else 0,
        }


_temporal_resolver: Optional[FrenchTemporalResolver] = None


def _get_temporal_resolver() -> FrenchTemporalResolver:
    global _temporal_resolver
    if _temporal_resolver is None:
        _temporal_resolver = FrenchTemporalResolver()
    return _temporal_resolver


# ============================================================
# 4. ConversationContext + ConversationFrame (conservés)
# ============================================================

@dataclass
class ConversationContext:
    last_intent_type: Optional[IntentType] = None
    last_verb_lemma: Optional[str] = None
    last_actor: Optional[str] = None
    last_action: Optional[str] = None
    last_location: Optional[str] = None
    last_when_iso: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > CONTEXT_RESET_SEC

    def update_from_intent(self, intent: "Intent"):
        self.last_intent_type = intent.intent
        self.last_verb_lemma = intent.verb_analysis.lemma if intent.verb_analysis else None
        self.last_actor = intent.who_raw
        self.last_action = intent.action or (intent.verb_analysis.lemma if intent.verb_analysis else None)
        self.last_location = intent.where
        self.last_when_iso = intent.when.iso_start if intent.when else None
        self.timestamp = time.time()

    def resolve_ellipsis(self, text: str) -> str:
        if self.is_expired():
            return text
        tl = text.lower().strip()
        if tl in ("oui", "ouais", "ok", "d'accord", "okay", "bien sûr"):
            if self.last_intent_type in ("query_narrative", "query_state") and self.last_verb_lemma:
                debug_logger.debug(f"[Ellipse] 'oui' → reprise du verbe '{self.last_verb_lemma}'")
                return f"oui, {self.last_verb_lemma}"
        if tl in ("encore", "recommence", "refais", "re"):
            if self.last_action:
                debug_logger.debug(f"[Ellipse] 'encore' → reprise de l'action '{self.last_action}'")
                return f"refais {self.last_action}"
        return text

    def resolve_pronouns(self, text: str) -> str:
        if self.is_expired():
            return text
        result = text
        if re.search(r'\by\b', result) and self.last_location:
            debug_logger.debug(f"[Pronom] 'y' → {self.last_location}")
            result = re.sub(r'\by\b', self.last_location, result)
        if re.search(r'\ben\b', result) and self.last_action:
            debug_logger.debug(f"[Pronom] 'en' → de {self.last_action}")
            result = re.sub(r'\ben\b', f"de {self.last_action}", result)
        return result


@dataclass
class ConversationFrame:
    who: Optional[PersonType] = None
    who_raw: Optional[str] = None
    with_who: List[str] = field(default_factory=list)
    when: Optional[TemporalSpan] = None
    where: Optional[str] = None
    what: Optional[ConceptType] = None
    last_update: float = field(default_factory=time.monotonic)
    pending_fragments: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    durations: List[dict] = field(default_factory=list)
    register: Optional[str] = None

    def is_expired(self) -> bool:
        return time.monotonic() - self.last_update > CONTEXT_RESET_SEC

    def reset(self):
        self.who = self.who_raw = self.when = self.where = self.what = self.register = None
        self.with_who = []
        self.pending_fragments = []
        self.subjects = []
        self.durations = []
        self.last_update = time.monotonic()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k) or v is None:
                continue
            if k == "with_who":
                self.with_who = list(set(self.with_who + v))
            elif k == "fragment":
                self.pending_fragments.append(v)
                debug_logger.debug(f"[Frame] Fragment ajouté: '{v}'")
            else:
                setattr(self, k, v)
        self.last_update = time.monotonic()

    def flush_pending(self) -> List[str]:
        frags, self.pending_fragments = list(self.pending_fragments), []
        return frags


# ============================================================
# 5. Structures de données (VerbAnalysis, Intent)
# ============================================================

@dataclass
class VerbAnalysis:
    lemma: str
    tense: TenseType
    scope: ScopeType
    concept: Optional[ConceptType]
    person: PersonType = "unknown"
    number: str = "unknown"
    mood: MoodType = "indicative"
    polarity: PolarityType = "positive"
    modal: ModalType = None
    pronominal: bool = False
    impersonal: bool = False
    compound: bool = False
    # Propagation
    is_modal_root: bool = False
    propagated_from: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegisterAnalysis:
    language: str = "fr"
    style: str = "neutre"
    confidence: float = 0.5
    markers: List[str] = field(default_factory=list)
    tu_vous: str = "indéterminé"
    negation: str = "absente"


@dataclass
class TokenNode:
    text: str
    lemma: str
    pos: str
    dep: str
    token_index: int
    role: Optional[RoleType] = None
    children: List["TokenNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text, "lemma": self.lemma, "pos": self.pos,
            "dep": self.dep, "role": self.role,
            "children": [c.to_dict() for c in self.children]
        }


@dataclass
class SyntacticTree:
    phrase: str
    root: Optional[TokenNode] = None
    subject: Optional[str] = None
    subject_role: Optional[RoleType] = None
    object_: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "root": self.root.to_dict() if self.root else None,
            "subject": self.subject, "subject_role": self.subject_role,
            "object": self.object_
        }


@dataclass
class Intent:
    text: str
    intent: IntentType
    confidence: float
    uncertain: bool
    scores: dict
    verb_analysis: Optional[VerbAnalysis]
    who: Optional[PersonType]
    who_raw: Optional[str]
    with_who: List[str]
    when: Optional[TemporalSpan]
    where: Optional[str]
    what: Optional[ConceptType]
    action: Optional[str]
    target: Optional[str]
    actions: List[dict]
    entities: List[dict]
    memory_hint: Optional[dict]
    register: Optional[dict]
    assembled_from: List[str]
    syntax_tree: Optional[SyntacticTree] = None
    corrections: List[Tuple[str, str]] = field(default_factory=list)
    clause_index: int = 0
    processing_ms: float = 0.0
    ts: float = field(default_factory=time.time)

    # Mémorisation brute
    is_memorize: bool = False
    memorize_content: Optional[str] = None
    memorize_trigger: Optional[str] = None
    memorize_subject: Optional[str] = None

    # Rappel
    is_recall: bool = False
    recall_query: Optional[str] = None
    recall_filters: Optional[Dict] = None

    # Commandes incarnées
    raw_args: List[str] = field(default_factory=list)
    unresolved: bool = False

    # Émotion du locuteur (prosodie STT)
    emotional_tone: Optional[str] = None
    emotional_intensity: float = 0.0

    # Routage (debug)
    routing_decision: Optional[str] = None
    routing_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.when:
            d["when"] = asdict(self.when)
        if self.verb_analysis:
            d["verb_analysis"] = self.verb_analysis.to_dict()
        if self.syntax_tree:
            d["syntax_tree"] = self.syntax_tree.to_dict()
        return d

    def to_cognitive_frame(self) -> dict:
        return {
            "type": "intent",
            "intent_type": self.intent,
            "verb": self.verb_analysis.lemma if self.verb_analysis else None,
            "concept": self.verb_analysis.concept if self.verb_analysis else None,
            "scope": self.verb_analysis.scope if self.verb_analysis else "UNKNOWN",
            "modal": self.verb_analysis.modal if self.verb_analysis else None,
            "polarity": self.verb_analysis.polarity if self.verb_analysis else "positive",
            "who": self.who,
            "who_raw": self.who_raw,
            "with_who": self.with_who,
            "time_start": self.when.iso_start if self.when else None,
            "time_end": self.when.iso_end if self.when else None,
            "time_event": self.when.named_event if self.when else None,
            "location": self.where,
            "action": self.action,
            "target": self.target,
            "confidence": self.confidence,
            "processing_ms": self.processing_ms,
            "is_memorize": self.is_memorize,
            "memorize_content": self.memorize_content,
            "memorize_subject": self.memorize_subject,
            "is_recall": self.is_recall,
            "recall_query": self.recall_query,
            "recall_filters": self.recall_filters,
            "raw_args": self.raw_args,
            "unresolved": self.unresolved,
            "emotional_tone": self.emotional_tone,
            "emotional_intensity": self.emotional_intensity,
        }


# ============================================================
# 6. PipelineMetrics
# ============================================================

class PipelineMetrics:
    def __init__(self):
        self.total = 0
        self.total_ms = 0.0
        self.errors = 0
        self.intent_dist: Dict[str, int] = defaultdict(int)
        self.corrections: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()

    def record(self, intent: Intent):
        self.total += 1
        self.total_ms += intent.processing_ms
        self.intent_dist[intent.intent] += 1
        for orig, _ in intent.corrections:
            self.corrections[orig] += 1

    def record_error(self):
        self.errors += 1

    def summary(self) -> dict:
        avg = self.total_ms / self.total if self.total else 0
        return {
            "total_requests": self.total,
            "total_errors": self.errors,
            "avg_ms": round(avg, 1),
            "intent_distribution": dict(self.intent_dist),
            "top_corrections": sorted(self.corrections.items(),
                                       key=lambda x: x[1], reverse=True)[:10],
            "uptime_s": round(time.time() - self.start_time, 1),
        }


# ============================================================
# 7. IntentRouter (Strategy Pattern avec propagation xcomp)
# ============================================================

class RoutingRule:
    def __init__(self, priority: int, condition_fn, result_intent: str, confidence: float, name: str):
        self.priority = priority
        self.condition = condition_fn
        self.result_intent = result_intent
        self.confidence = confidence
        self.name = name

    def matches(self, **kwargs) -> bool:
        return self.condition(**kwargs)


class IntentRouter:
    def __init__(self):
        self.rules = self._build_rules()

    def _build_rules(self) -> List[RoutingRule]:
        rules = [
            # Priorité 100: Mémorisation explicite
            RoutingRule(100,
                       lambda text, **kw: any(t in text.lower() for t in MEMORIZE_TRIGGERS),
                       "memorize", 0.95, "memorize_explicit"),

            # Priorité 95: Rappel explicite
            RoutingRule(95,
                       lambda text, **kw: any(t in text.lower() for t in RECALL_TRIGGERS),
                       "recall", 0.92, "recall_explicit"),

            # Priorité 90: Enseignement
            RoutingRule(90,
                       lambda verb, **kw: verb and verb.concept == "TEACH",
                       "teach", 0.90, "teach_concept"),

            # Priorité 85: Commandes incarnées (avec propagation xcomp déjà faite)
            RoutingRule(85,
                       lambda verb, mood, person, **kw: (
                           verb and verb.concept == "MOTOR_CMD" and
                           (mood == "imperative" or person == "2nd_sg")
                       ),
                       "command_motor", 0.93, "command_motor"),

            RoutingRule(85,
                       lambda verb, mood, person, **kw: (
                           verb and verb.concept == "SENSE_FOCUS" and
                           (mood == "imperative" or person == "2nd_sg")
                       ),
                       "command_sense", 0.93, "command_sense"),

            RoutingRule(85,
                       lambda verb, mood, person, **kw: (
                           verb and verb.concept == "ACTUATOR_CMD" and
                           (mood == "imperative" or person == "2nd_sg")
                       ),
                       "command_actuator", 0.93, "command_actuator"),

            # Priorité 80: Mémorisation/rappel par concept
            RoutingRule(80,
                       lambda verb, **kw: verb and verb.concept == "MEMORIZE",
                       "memorize", 0.90, "memorize_concept"),

            RoutingRule(80,
                       lambda verb, **kw: verb and verb.concept == "RECALL",
                       "recall", 0.90, "recall_concept"),

            # Priorité 75: Action device (impératif)
            RoutingRule(75,
                       lambda verb, mood, text, **kw: (
                           verb and verb.mood == "imperative" and
                           (verb.lemma in ACTION_VERBS or
                            any(d in text.lower() for d in DEVICE_NOUNS))
                       ),
                       "action_device", 0.95, "action_device_imperative"),

            # Priorité 70: Chit-chat lexical
            RoutingRule(70,
                       lambda text, **kw: (
                           any(w in text.lower() for w in CHIT_CHAT_WORDS) or
                           any(p in text.lower() for p in CHIT_CHAT_PHRASES)
                       ),
                       "chit_chat", 0.93, "chit_chat_lexical"),

            RoutingRule(65,
                       lambda verb, text, **kw: (
                           verb and verb.concept == "SOCIAL_ACT" and len(text.split()) <= 7
                       ),
                       "chit_chat", 0.88, "chit_chat_social"),

            # Priorité 60: Questions
            RoutingRule(60,
                       lambda text, verb, **kw: (
                           ("?" in text or any(w in text.lower() for w in INTERROGATIVE_WORDS)) and
                           verb and verb.scope == "PAST"
                       ),
                       "query_narrative", 0.90, "question_narrative"),

            RoutingRule(60,
                       lambda text, verb, **kw: (
                           ("?" in text or any(w in text.lower() for w in INTERROGATIVE_WORDS)) and
                           verb and verb.scope == "FUTURE"
                       ),
                       "query_intention", 0.88, "question_intention"),

            RoutingRule(60,
                       lambda text, **kw: (
                           "?" in text or any(w in text.lower() for w in INTERROGATIVE_WORDS)
                       ),
                       "query_state", 0.85, "question_state"),

            # Priorité 50: Information
            RoutingRule(50,
                       lambda verb, **kw: verb and verb.person in ("1st_sg", "1st_pl"),
                       "information_input", 0.88, "information_self"),

            RoutingRule(45,
                       lambda verb, **kw: verb and verb.tense in ("past", "present", "future"),
                       "information_input", 0.78, "information_other"),

            # Priorité 10: Fallback
            RoutingRule(10,
                       lambda **kw: True,
                       "chit_chat", 0.55, "fallback"),
        ]
        return sorted(rules, key=lambda r: -r.priority)

    def classify(self, text: str, verb: Optional[VerbAnalysis] = None, **kwargs) -> dict:
        text_lower = text.lower()
        mood = verb.mood if verb else None
        person = verb.person if verb else None
        applied = []

        for rule in self.rules:
            if rule.matches(text=text_lower, verb=verb, mood=mood, person=person, **kwargs):
                applied.append(rule.name)
                debug_logger.debug(f"[Router] {rule.name} → {rule.result_intent} (conf={rule.confidence})")
                return {
                    "intent": rule.result_intent,
                    "confidence": rule.confidence,
                    "uncertain": rule.confidence < CONFIDENCE_THRESHOLD,
                    "scores": {rule.result_intent: rule.confidence},
                    "routing_rule": rule.name,
                    "applied_rules": applied
                }

        return {
            "intent": "chit_chat",
            "confidence": 0.5,
            "uncertain": True,
            "scores": {},
            "routing_rule": "none",
            "applied_rules": applied
        }


# ============================================================
# 8. Extracteurs spaCy (1 passe)
# ============================================================

def _extract_all_slots(doc) -> Tuple[Optional[PersonType], Optional[str], List[str], Optional[str], List[dict]]:
    who_person: Optional[PersonType] = None
    who_raw: Optional[str] = None
    with_who_set: set = set()
    where: Optional[str] = None
    entities: List[dict] = []
    seen_spans: set = set()
    subjects: set = set()

    for ent in doc.ents:
        etype = NER_TYPE_MAP.get(ent.label_)
        if not etype:
            continue
        key = (ent.start, ent.end)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        raw = ent.text.strip()
        value: Optional[float] = None
        unit: Optional[str] = None
        if etype in ("number", "ordinal", "quantity", "percent", "money",
                     "time_ref", "date_ref", "duration"):
            nm = re.search(r"(\d+(?:[.,]\d+)?)",
                           raw.replace("\u202f", "").replace(" ", ""))
            if nm:
                try:
                    value = float(nm.group(1).replace(",", "."))
                except (ValueError, AttributeError):
                    pass
            for pat, norm in UNIT_PATTERNS.items():
                if pat in raw.lower():
                    unit = norm
                    break
        entities.append({"raw": raw, "type": etype, "value": value, "unit": unit})
        if ent.label_ == "PER":
            with_who_set.add(ent.text)

    for token in doc:
        if not who_person and token.dep_ in ("iobj", "obj:iobj"):
            raw_t = token.text.lower().rstrip("'")
            p = CLITIC_TO_PERSON.get(raw_t) or CLITIC_TO_PERSON.get(token.text.lower())
            if p:
                who_person, who_raw = p, token.text

        elif not who_person and token.dep_ == "obj":
            raw_t = token.text.lower().rstrip("'")
            p = CLITIC_TO_PERSON.get(raw_t) or CLITIC_TO_PERSON.get(token.text.lower())
            if p:
                who_person, who_raw = p, token.text

        elif token.dep_ in ("nsubj", "nsubj:pass"):
            raw_t = token.text.lower().rstrip("'")
            p = CLITIC_TO_PERSON.get(raw_t)
            if p:
                if not who_person:
                    who_person, who_raw = p, token.text
            elif token.text.lower() in ("je", "j'", "j", "moi"):
                if not who_person:
                    who_person, who_raw = "1st_sg", token.text
            elif token.text.lower() in ("tu", "t'", "te", "toi"):
                if not who_person:
                    who_person, who_raw = "2nd_sg", token.text
            elif token.text.lower() in ("nous", "on"):
                if not who_person:
                    who_person, who_raw = "1st_pl", token.text
            elif token.text.lower() == "vous":
                if not who_person:
                    who_person, who_raw = "2nd_pl", token.text
            elif token.ent_type_ == "PER" and not who_person:
                who_person, who_raw = token.text, token.text
            subjects.add(token.text)

        if not where and token.text.lower() in ("dans", "au", "en", "à", "chez") \
                and token.dep_ == "case":
            head = token.head
            if head.pos_ in ("NOUN", "PROPN"):
                where = head.text.lower()

        if token.ent_type_ == "PER" and token.text not in subjects:
            with_who_set.add(token.text)

    if not where:
        for ent in doc.ents:
            if ent.label_ in ("LOC", "GPE"):
                where = ent.text.lower()
                break

    return who_person, who_raw, list(with_who_set), where, entities


def _extract_verb_with_propagation(doc) -> Optional[VerbAnalysis]:
    """
    Extrait le verbe principal avec propagation via xcomp pour les modaux.
    Exemple: "Tu peux regarder" → concept=SENSE_FOCUS (propagé depuis "regarder")
    """
    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break
    if not root or root.pos_ != "VERB":
        return None

    def analyze(token):
        morph = token.morph
        mood_map = {"Cnd": "conditional", "Imp": "imperative",
                    "Sub": "subjunctive", "Ind": "indicative"}
        mood_raw = morph.get("Mood")
        if mood_raw:
            mood = mood_map.get(mood_raw[0], "indicative")
        elif morph.get("VerbForm") and "Inf" in morph.get("VerbForm"):
            mood = "infinitive"
        else:
            mood = "indicative"

        tense_raw = morph.get("Tense")
        if mood == "conditional":
            tense = "conditional"
        elif tense_raw:
            if "Past" in tense_raw or "Imp" in tense_raw:
                tense = "past"
            elif "Pres" in tense_raw:
                aux_pres = any(
                    c.pos_ == "AUX" and c.morph.get("Tense") and "Pres" in c.morph.get("Tense")
                    for c in token.children
                )
                is_pp = morph.get("VerbForm") and "Part" in morph.get("VerbForm")
                tense = "past" if (is_pp and aux_pres) else "present"
            elif "Fut" in tense_raw:
                tense = "future"
            else:
                tense = "unknown"
        else:
            tense = "unknown"

        tense = FrenchVerbAnalyzer.refine_tense(token.text, token.lemma_, tense)

        person_raw = morph.get("Person")
        number_raw = morph.get("Number")
        person: PersonType = "unknown"
        number = "unknown"
        if person_raw and number_raw:
            pm = {("1", "Sing"): "1st_sg", ("2", "Sing"): "2nd_sg", ("3", "Sing"): "3rd_sg",
                  ("1", "Plur"): "1st_pl", ("2", "Plur"): "2nd_pl", ("3", "Plur"): "3rd_pl"}
            person = pm.get((person_raw[0], number_raw[0]), "unknown")
            number = "sg" if number_raw[0] == "Sing" else "pl" if number_raw[0] == "Plur" else "unknown"

        neg = [c for c in token.children
               if c.dep_ == "advmod" and c.lemma_ in ("ne", "pas", "plus", "jamais", "rien", "guère")]
        left_neg = any(t.dep_ == "advmod" and t.lemma_ in ("ne", "n") for t in token.lefts)
        polarity = "negative" if (neg or left_neg) else "positive"

        modal = None
        for aux in token.children:
            if aux.pos_ == "AUX":
                modal = MODAL_MARKERS.get(aux.lemma_.lower())
                if modal:
                    break
        if not modal and mood == "conditional":
            modal = "wish"

        concept = SEMANTIC_MAP.get(token.lemma_.lower())

        # PROPAGATION: si c'est un modal avec xcomp, prendre le concept du xcomp
        if token.lemma_ in MODAL_VERBS:
            for child in token.children:
                if child.dep_ == "xcomp" and child.pos_ == "VERB":
                    xcomp_concept = SEMANTIC_MAP.get(child.lemma_.lower())
                    if xcomp_concept and xcomp_concept != concept:
                        debug_logger.debug(f"[Propagation] {token.lemma_} ({concept}) → {child.lemma_} ({xcomp_concept})")
                        concept = xcomp_concept
                    break

        return VerbAnalysis(
            lemma=token.lemma_, tense=tense,
            scope=TENSE_TO_SCOPE.get(tense, "UNKNOWN"),
            concept=concept, person=person, number=number, mood=mood,
            polarity=polarity, modal=modal,
            pronominal=FrenchVerbAnalyzer.is_pronominal(doc, token),
            impersonal=FrenchVerbAnalyzer.is_impersonal(token.lemma_),
            compound=FrenchVerbAnalyzer.is_compound(token.text) is not None,
            is_modal_root=token.lemma_ in MODAL_VERBS
        )

    return analyze(root)


def _extract_actions(doc) -> List[dict]:
    actions = []
    for token in doc:
        if token.pos_ != "VERB" or token.dep_ in ("aux", "aux:pass", "cop"):
            continue
        subs = [c.text for c in token.children if c.dep_ in ("nsubj", "nsubj:pass", "expl:subj")]
        objs = [c.text for c in token.children if c.dep_ in ("obj", "iobj", "obl", "xcomp")]
        if not subs:
            for t in doc:
                if t.text.lower() in ("je", "j'", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on"):
                    subs.append(t.text)
                    break
        actions.append({"verb": token.lemma_.lower(), "subjects": subs,
                        "objects": objs, "concept": SEMANTIC_MAP.get(token.lemma_.lower())})
    return actions


def _extract_what(doc) -> Optional[ConceptType]:
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("ROOT", "acl", "relcl", "advcl", "xcomp", "ccomp"):
            c = SEMANTIC_MAP.get(token.lemma_.lower())
            if c:
                return c
            for child in token.children:
                if child.pos_ == "VERB":
                    c = SEMANTIC_MAP.get(child.lemma_.lower())
                    if c:
                        return c
    return None


def _extract_device(doc) -> Tuple[Optional[str], Optional[str]]:
    action = target = None
    for token in doc:
        if token.pos_ == "VERB" and not action:
            action = ACTION_VERBS.get(token.lemma_.lower())
        if token.pos_ in ("NOUN", "PROPN") and not target:
            target = DEVICE_NOUNS.get(token.lemma_.lower())
    if not target:
        tl = doc.text.lower()
        for noun, norm in DEVICE_NOUNS.items():
            if noun in tl:
                target = norm
                break
    return action, target


def _extract_raw_args(doc) -> List[str]:
    args: List[str] = []
    seen: set = set()

    for token in doc:
        text = token.text.strip()
        if not text or text.lower() in seen or token.is_punct:
            continue
        if token.dep_ in ("obj", "iobj", "obl", "obl:arg", "xcomp"):
            args.append(text)
            seen.add(text.lower())
        elif token.dep_ == "advmod" and token.pos_ in ("ADV", "ADJ", "NOUN"):
            args.append(text)
            seen.add(text.lower())
        elif (token.dep_ in ("prep", "case") and token.head.dep_ in ("obl", "nmod", "advmod")):
            phrase = " ".join(t.text for t in token.head.subtree if not t.is_punct).strip()
            if phrase and phrase.lower() not in seen:
                args.append(phrase)
                seen.add(phrase.lower())
        elif (token.dep_ in ("nummod", "amod") and token.head.dep_ in ("obj", "obl", "attr")):
            args.append(text)
            seen.add(text.lower())

    return args


def _compute_salience(doc, entities: List[dict], verb: Optional[VerbAnalysis] = None) -> float:
    score = 0.0
    for e in entities:
        score += (0.20 if e["type"] in ("person", "organization", "event", "product") else
                  0.12 if e["type"] in ("location", "facility") else
                  0.10 if e["type"] in ("quantity", "money", "percent", "duration") else 0.05)
    if verb and verb.concept in ("COMMUNICATE", "PERCEIVE", "MOVE", "TRANSFER", "SOCIAL_ACT"):
        score += 0.15
    elif verb and verb.concept in ("INGEST", "CREATE", "HEALTH"):
        score += 0.10
    content = [t for t in doc if t.pos_ in ("NOUN", "PROPN", "VERB", "ADJ", "NUM")]
    score += min(0.20, len(content) * 0.04)
    return round(min(1.0, score), 3)


def _infer_info_subject(doc) -> str:
    for token in doc:
        if token.dep_ in ("nsubj", "nsubj:pass"):
            t = token.text.lower().rstrip("'")
            if t in ("je", "j", "moi", "nous", "on"):
                return "human"
            if t in ("tu", "toi", "te", "t"):
                return "self"
            if token.ent_type_ == "PER":
                return "other"
            return "world"
    return "other"


def _is_fragment(doc, slots: dict) -> bool:
    has_verb = any(
        t.pos_ == "VERB" and t.dep_ in ("ROOT", "acl", "relcl", "advcl", "xcomp", "ccomp")
        for t in doc
    )
    return not has_verb and any([
        slots.get("who"), slots.get("when"), slots.get("where"), slots.get("with_who"),
    ])


def _analyze_register(doc, text: str) -> RegisterAnalysis:
    toks = [t.text.lower().rstrip("'") for t in doc]
    votes: Dict[str, float] = {s: 0.0 for s in REGISTER_LEXICON}
    marks: Dict[str, list] = {s: [] for s in REGISTER_LEXICON}
    for tok in toks:
        for style, lex in REGISTER_LEXICON.items():
            if tok in lex:
                votes[style] += 1
                marks[style].append(tok)
    has_pas = any(w in toks for w in ("pas", "plus", "jamais"))
    has_ne = any(t in toks for t in NEGATION_COMPLETE)
    if has_pas and not has_ne:
        votes["familier"] += 1
        marks["familier"].append("neg-inc")
    tu_vous = "indéterminé"
    if any(t in VOUVOIEMENT_MARKERS for t in toks) and not any(t in TUTOIEMENT_MARKERS for t in toks):
        tu_vous = "vous"
        votes["soutenu"] += 0.5
    elif any(t in TUTOIEMENT_MARKERS for t in toks):
        tu_vous = "tu"
    best = max(votes, key=votes.get)
    conf = round(min(0.95, sum(votes.values()) / max(len(toks), 1)), 3)
    if votes[best] < 0.8:
        best = "neutre"
    all_markers = marks.get(best, [])
    if tu_vous != "indéterminé":
        all_markers.append(f"tutv:{tu_vous}")
    neg = ("complete" if has_pas and has_ne else "incomplete" if has_pas else "absente")
    return RegisterAnalysis(style=best, confidence=conf,
                            markers=all_markers[:8], tu_vous=tu_vous, negation=neg)


# ============================================================
# 9. split_clauses
# ============================================================

_CLAUSE_COORD = re.compile(
    r'\s+(?:et|puis|ensuite|après|alors)\s+'
    r'(?=je\s|j\'|tu\s|il\s|elle\s|nous\s|vous\s|ils\s|elles\s|on\s)',
    re.IGNORECASE,
)

_VERB_PAT = re.compile(
    r"\b(suis|es|est|sommes|êtes|sont|ai|as|a|avons|avez|ont|"
    r"vais|vas|va|allons|allez|vont|ferai|feras|fera|"
    r"\w+erai|\w+eras|\w+era|\w+erons|\w+erez|\w+eront|"
    r"\w+ais|\w+ait|\w+ions|\w+iez|\w+aient)\b",
    re.IGNORECASE,
)


def split_clauses(text: str) -> List[str]:
    parts = _CLAUSE_COORD.split(text)
    if len(parts) == 1:
        return [text.strip()]
    valid: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _VERB_PAT.search(part):
            valid.append(part)
        else:
            if valid:
                valid[-1] += " " + part
            else:
                valid.append(part)
    return valid if len(valid) >= 2 else [text.strip()]


# ============================================================
# 10. SyntacticTreeExtractor
# ============================================================

_USER = {"je", "j'", "moi", "me", "m'"}
_ASE = {"tu", "toi", "te", "t'"}


class SyntacticTreeExtractor:
    @classmethod
    def _role(cls, token) -> Optional[RoleType]:
        tl = token.text.lower()
        if tl in _USER:
            return "user"
        if tl in _ASE:
            return "ase"
        if token.ent_type_ == "PER":
            return "other"
        return None

    @classmethod
    def extract(cls, doc) -> SyntacticTree:
        tree = SyntacticTree(phrase=doc.text)
        nodes: Dict[int, TokenNode] = {}

        for token in doc:
            node = TokenNode(
                text=token.text, lemma=token.lemma_,
                pos=token.pos_, dep=token.dep_,
                token_index=token.i, role=cls._role(token),
            )
            nodes[token.i] = node
            if token.dep_ == "ROOT":
                tree.root = node
            if token.dep_ in ("nsubj", "nsubj:pass"):
                tree.subject = token.text
                tree.subject_role = node.role
            if token.dep_ == "obj" and tree.object_ is None:
                tree.object_ = token.text

        for token in doc:
            if token.head != token and token.head.i in nodes:
                nodes[token.head.i].children.append(nodes[token.i])

        return tree


# ============================================================
# 11. RawMemory
# ============================================================

@dataclass
class RawMemoryItem:
    id: str
    raw_text: str
    speaker: str
    subject: Optional[str] = None
    location: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    recalled_count: int = 0
    last_recalled: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_restoration(self) -> str:
        return self.raw_text


class RawMemory:
    def __init__(self):
        self._items: Dict[str, RawMemoryItem] = {}
        self._counter = 0
        self._lock = threading.RLock()

    def store(self, text: str, speaker: str, subject: Optional[str] = None,
              location: Optional[str] = None, context: Optional[Dict] = None) -> str:
        with self._lock:
            self._counter += 1
            item_id = f"raw_mem_{self._counter}"
            self._items[item_id] = RawMemoryItem(
                id=item_id, raw_text=text, speaker=speaker,
                subject=subject, location=location, context=context or {}
            )
            debug_logger.debug(f"[RawMemory] Stored: {item_id} → '{text[:50]}...'")
            return item_id

    def recall(self, query) -> List[RawMemoryItem]:
        with self._lock:
            results = []
            for item in self._items.values():
                if query.speaker and item.speaker != query.speaker:
                    continue
                if query.subject and item.subject != query.subject:
                    continue
                if query.location and item.location != query.location:
                    continue
                if query.contains_text and query.contains_text.lower() not in item.raw_text.lower():
                    continue
                results.append(item)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            for r in results[:query.max_results]:
                r.recalled_count += 1
                r.last_recalled = time.time()
            return results[:query.max_results]


@dataclass
class RecallQuery:
    speaker: Optional[str] = None
    subject: Optional[str] = None
    location: Optional[str] = None
    time_after: Optional[float] = None
    time_before: Optional[float] = None
    contains_text: Optional[str] = None
    max_results: int = 5


# ============================================================
# 12. EventGraph + Consumer
# ============================================================

@dataclass
class EpisodicEntity:
    name: str
    type: str
    count: int = 1


@dataclass
class EpisodicEvent:
    id: int
    action: str
    agent: List[str]
    objects: List[str]
    location: Optional[str]
    concept: Optional[str]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    scope: str
    salience: float
    trigger_id: Optional[int]
    intent_type: str
    raw_text: str
    ts: float


@dataclass
class GraphContext:
    subjects: List[str] = field(default_factory=list)
    location: Optional[str] = None
    time_base: Optional[datetime.datetime] = None
    time_end: Optional[datetime.datetime] = None


class EventGraph:
    def __init__(self):
        self._lock = threading.RLock()
        self.entities: Dict[str, EpisodicEntity] = {}
        self.events: List[EpisodicEvent] = []
        self.context = GraphContext()
        self._counter = 0

    def set_location(self, loc: Optional[str]):
        with self._lock:
            if loc:
                self.context.location = loc.lower()

    def set_time(self, iso_start: Optional[str], iso_end: Optional[str] = None):
        with self._lock:
            if iso_start:
                try:
                    self.context.time_base = datetime.datetime.fromisoformat(iso_start)
                except (ValueError, TypeError):
                    pass
            if iso_end:
                try:
                    self.context.time_end = datetime.datetime.fromisoformat(iso_end)
                except (ValueError, TypeError):
                    pass

    def set_subjects(self, subjects: List[str]):
        with self._lock:
            if subjects:
                self.context.subjects = subjects

    def upsert_entity(self, name: str, etype: str):
        key = name.lower()
        with self._lock:
            if key in self.entities:
                self.entities[key].count += 1
            else:
                self.entities[key] = EpisodicEntity(name=name, type=etype, count=1)

    def add_event(self, action: str, agent: List[str] = None, objects: List[str] = None,
                  concept: str = None, scope: str = "UNKNOWN", salience: float = 0.0,
                  trigger_id: int = None, intent_type: str = "unknown",
                  raw_text: str = "") -> EpisodicEvent:
        with self._lock:
            self._counter += 1
            ev = EpisodicEvent(
                id=self._counter, action=action,
                agent=agent or self.context.subjects or [],
                objects=objects or [], location=self.context.location,
                concept=concept, start_time=self.context.time_base,
                end_time=self.context.time_end, scope=scope,
                salience=salience, trigger_id=trigger_id,
                intent_type=intent_type, raw_text=raw_text, ts=time.time(),
            )
            self.events.append(ev)
            return ev

    def last_events(self, n: int = 5) -> List[EpisodicEvent]:
        with self._lock:
            return list(self.events[-n:])

    def top_entities(self, n: int = 5) -> List[EpisodicEntity]:
        with self._lock:
            return sorted(self.entities.values(), key=lambda e: e.count, reverse=True)[:n]

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_events": len(self.events),
                "total_entities": len(self.entities),
                "by_scope": {s: sum(1 for e in self.events if e.scope == s)
                             for s in ("PAST", "PRESENT", "FUTURE", "HYPOTHETICAL", "UNKNOWN")},
                "top_entities": [{"name": e.name, "type": e.type, "count": e.count}
                                 for e in self.top_entities()],
            }


class EventGraphConsumer(threading.Thread):
    EVENT_INTENTS = {"information_input", "action_device", "chit_chat"}

    def __init__(self, q_in: Queue, graph: EventGraph):
        super().__init__(name="eventgraph-consumer", daemon=True)
        self._q_in = q_in
        self._graph = graph
        self._running = True

    def stop(self):
        self._running = False
        self._q_in.put(None)

    def run(self):
        logger.info("EventGraphConsumer démarré")
        while self._running:
            try:
                payload = self._q_in.get(timeout=1.0)
            except Empty:
                continue
            if payload is None:
                break
            if isinstance(payload, dict):
                payload = [payload]
            for intent in payload:
                self._consume(intent)
        logger.info("EventGraphConsumer arrêté")

    def _consume(self, intent: dict):
        self._graph.set_location(intent.get("where"))
        when = intent.get("when")
        if when:
            self._graph.set_time(when.get("iso_start"), when.get("iso_end"))
        actions = intent.get("actions") or []
        agents = list({s for a in actions for s in (a.get("subjects") or [])
                       if s.lower() not in ("je", "j'", "j", "on")})
        if any(s.lower().rstrip("'") in ("je", "j", "moi", "on", "nous")
               for a in actions for s in (a.get("subjects") or [])):
            agents = ["human"] + agents
        if agents:
            self._graph.set_subjects(agents)
        for ent in (intent.get("entities") or []):
            if ent.get("type") in ("person", "location", "organization", "event", "product"):
                self._graph.upsert_entity(ent["raw"], ent["type"])
        for p in (intent.get("with_who") or []):
            self._graph.upsert_entity(p, "person")
        if intent.get("intent") not in self.EVENT_INTENTS:
            return
        verb = intent.get("verb_analysis") or {}
        scope = verb.get("scope", "UNKNOWN")
        salience = (intent.get("memory_hint") or {}).get("salience", 0.3)
        prev_id = None
        for act in (actions or [{"verb": intent.get("action", "unknown"),
                                 "subjects": [], "objects": [intent.get("target")], "concept": None}]):
            ev = self._graph.add_event(
                action=act.get("verb", "unknown"),
                agent=act.get("subjects") or [],
                objects=[o for o in (act.get("objects") or []) if o],
                concept=act.get("concept"),
                scope=scope, salience=salience,
                trigger_id=prev_id,
                intent_type=intent.get("intent", "unknown"),
                raw_text=intent.get("text", ""),
            )
            prev_id = ev.id


# ============================================================
# 13. IntentPipeline (classe principale)
# ============================================================

class IntentPipeline:
    def __init__(self, q_in: Queue, q_out: Queue, debug: bool = False):
        self._q_in = q_in
        self._q_out = q_out
        self.debug = debug

        self._nlp = None
        self._router = IntentRouter()
        self._temporal_resolver = FrenchTemporalResolver()
        self._ctx = ConversationContext()
        self._frame = ConversationFrame()
        self._metrics = PipelineMetrics()
        self._raw_memory = RawMemory()
        self._event_graph = EventGraph()
        self._graph_consumer = None

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._loaded = False
        self._enable_corrector = True
        self._corrections_stats: Dict[str, int] = defaultdict(int)

    def enable_corrector(self, enable: bool = True):
        self._enable_corrector = enable

    def load(self):
        if not SPACY_AVAILABLE:
            raise RuntimeError("spacy non installé. pip install spacy && python -m spacy download fr_core_news_sm")
        self._nlp = spacy.load(SPACY_MODEL)
        self._loaded = True
        logger.info("IntentPipeline v3.1 chargé")

    def start(self):
        if not self._loaded:
            raise RuntimeError("Appelez load() avant start()")
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="TTI-Worker", daemon=True)
        self._thread.start()
        self._graph_consumer = EventGraphConsumer(self._q_out, self._event_graph)
        self._graph_consumer.start()
        logger.info("IntentPipeline v3.1 démarré")

    def stop(self):
        self._running = False
        if self._graph_consumer:
            self._graph_consumer.stop()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("IntentPipeline v3.1 arrêté")

    def _worker(self):
        while self._running:
            try:
                text = self._q_in.get(timeout=0.5)
                if text is None:
                    continue
                intents = self._process_text(text)
                if intents:
                    self._q_out.put([i.to_dict() for i in intents])
                    for i in intents:
                        self._metrics.record(i)
                        self._ctx.update_from_intent(i)
            except Empty:
                if self._frame.is_expired():
                    self._frame.reset()
                if self._ctx.is_expired():
                    self._ctx = ConversationContext()
                continue

    def _process_text(self, text: str) -> List[Intent]:
        t0 = time.time()
        debug_logger.debug(f"\n{'=' * 60}\nTEXTE: {text}")

        if self._frame.is_expired():
            self._frame.reset()
        if self._ctx.is_expired():
            self._ctx = ConversationContext()

        # Résolution ellipses + pronoms (ConversationContext)
        text = self._ctx.resolve_ellipsis(text)
        text = self._ctx.resolve_pronouns(text)

        # Nettoyage
        text = re.sub(r'[^\w\s\u00C0-\u00FF?.!,;:\'"()-]', '', text)

        # Correction (SMS, phonétique)
        if self._enable_corrector:
            corrected, corrections = FrenchTextCorrector.correct(text)
            for orig, fix in corrections:
                self._corrections_stats[f"{orig}→{fix}"] += 1
            text_clean = corrected
            debug_logger.debug(f"[Corrigé] {text_clean}")
        else:
            text_clean = text
            corrections = []

        # spaCy
        doc = self._nlp(text_clean)

        # Extraction du verbe (avec propagation xcomp)
        verb_analysis = _extract_verb_with_propagation(doc)

        # Classification (table de routage)
        clf = self._router.classify(text_clean, verb=verb_analysis)

        # Extraction des slots (who, where, with_who, entities)
        who_p, who_r, with_who, where, entities = _extract_all_slots(doc)

        # Extraction temporelle
        temporal = None
        if verb_analysis:
            resolved = self._temporal_resolver.resolve(text_clean)
            if resolved:
                temporal = self._temporal_resolver.to_temporal_span(resolved, verb_analysis.tense)
        if temporal is None:
            # Fallback sur dateparser simple
            pass

        # Extraction des actions multiples
        actions = _extract_actions(doc)

        # Héritage des slots depuis ConversationFrame
        if clause_index == 0:
            self._frame.update(who=who_p, who_raw=who_r, with_who=with_who,
                               when=temporal, where=where,
                               subjects=[s for a in actions for s in a.get("subjects", [])])

        # Fusion des slots
        merged_who = who_p or self._frame.who
        merged_who_raw = who_r or self._frame.who_raw
        merged_with_who = with_who or self._frame.with_who
        merged_when = temporal or self._frame.when
        merged_where = where or self._frame.where

        # Analyse du registre
        register = _analyze_register(doc, text_clean)

        # Action device
        action = None
        target = None
        if clf["intent"] == "action_device":
            action, target = _extract_device(doc)

        # Mémoire hint
        memory_hint = None
        if clf["intent"] == "information_input":
            memory_hint = {
                "subject": _infer_info_subject(doc),
                "salience": _compute_salience(doc, entities, verb_analysis),
                "entities": entities,
                "raw_info": text_clean,
            }

        # Mémorisation brute
        is_memorize = clf["intent"] == "memorize"
        memorize_content = None
        memorize_trigger = None
        memorize_subject = None
        if is_memorize:
            for trigger in MEMORIZE_TRIGGERS:
                if trigger in text_clean.lower():
                    memorize_trigger = trigger
                    idx = text_clean.lower().find(trigger)
                    memorize_content = text_clean[idx + len(trigger):].strip()
                    if memorize_content.startswith("que "):
                        memorize_content = memorize_content[4:]
                    break
            memorize_subject = self._extract_subject_from_text(memorize_content or "")

        # Rappel
        is_recall = clf["intent"] == "recall"
        recall_query = None
        recall_filters = None
        if is_recall:
            for trigger in RECALL_TRIGGERS:
                if trigger in text_clean.lower():
                    idx = text_clean.lower().find(trigger)
                    recall_query = text_clean[idx + len(trigger):].strip()
                    break
            recall_filters = self._build_recall_filters(recall_query or "")

        # Commandes incarnées
        embodied_intents = ("command_motor", "command_sense", "command_actuator", "teach")
        raw_args = _extract_raw_args(doc) if clf["intent"] in embodied_intents else []
        unresolved = clf["intent"] in embodied_intents

        # Arbre syntaxique
        syntax_tree = SyntacticTreeExtractor.extract(doc)

        # Construction de l'Intent
        intent = Intent(
            text=text_clean,
            intent=clf["intent"],
            confidence=clf["confidence"],
            uncertain=clf["uncertain"],
            scores=clf["scores"],
            verb_analysis=verb_analysis,
            who=merged_who,
            who_raw=merged_who_raw,
            with_who=merged_with_who,
            when=merged_when,
            where=merged_where,
            what=verb_analysis.concept if verb_analysis else None,
            action=action,
            target=target,
            actions=actions,
            entities=entities,
            memory_hint=memory_hint,
            register=register,
            assembled_from=self._frame.flush_pending(),
            syntax_tree=syntax_tree,
            corrections=corrections,
            clause_index=0,
            processing_ms=(time.time() - t0) * 1000,
            is_memorize=is_memorize,
            memorize_content=memorize_content,
            memorize_trigger=memorize_trigger,
            memorize_subject=memorize_subject,
            is_recall=is_recall,
            recall_query=recall_query,
            recall_filters=recall_filters,
            raw_args=raw_args,
            unresolved=unresolved,
            emotional_tone=None,  # À alimenter par STT
            emotional_intensity=0.0,
            routing_decision=clf.get("routing_rule"),
            routing_rules=clf.get("applied_rules", [])
        )

        debug_logger.debug(f"[Résultat] {intent.intent} (conf={intent.confidence}) en {intent.processing_ms:.0f}ms")
        return [intent]

    def _extract_subject_from_text(self, text: str) -> Optional[str]:
        for pattern in (r"que\s+(\w+)", r"(\w+)\s+est", r"(\w+)\s+a"):
            m = re.search(pattern, text.lower())
            if m:
                return m.group(1)
        return None

    def _build_recall_filters(self, query: str) -> Dict:
        filters = {}
        if any(w in query.lower() for w in ("je", "moi", "j'")):
            filters["speaker"] = "user"
        m = re.search(r"(?:de|sur|à propos de)\s+(\w+)", query.lower())
        if m:
            filters["subject"] = m.group(1)
        return filters

    def get_raw_memory(self) -> RawMemory:
        return self._raw_memory

    def get_event_graph(self) -> EventGraph:
        return self._event_graph

    def get_metrics(self) -> dict:
        return {
            "pipeline": self._metrics.summary(),
            "temporal_cache": self._temporal_resolver.get_stats(),
            "corrections_top": dict(sorted(
                self._corrections_stats.items(),
                key=lambda x: x[1], reverse=True)[:10]),
        }


# ============================================================
# 14. Tests
# ============================================================

def run_tests():
    print("\n" + "=" * 60)
    print("TESTS TTI v3.1")
    print("=" * 60)

    q_in = Queue()
    q_out = Queue()
    pipeline = IntentPipeline(q_in, q_out, debug=True)
    pipeline.load()
    pipeline.start()

    test_phrases = [
        ("Tu peux regarder sur ta gauche, s'il te plaît ?", "command_sense"),
        ("allume la lumière", "action_device"),
        ("qui est Vincent", "query_state"),
        ("je suis fatigué", "information_input"),
        ("bonjour", "chit_chat"),
        ("retiens que j'ai rendez-vous demain", "memorize"),
        ("rappelle-moi ce que j'ai dit", "recall"),
        ("avance tout droit", "command_motor"),
        ("regarde à droite", "command_sense"),
        ("je jardinerai et je lirai ensuite", "information_input"),  # multi-clause
    ]

    passed = 0
    for text, expected in test_phrases:
        q_in.put(text)
        try:
            result = q_out.get(timeout=5.0)
            intent = result[0]["intent"] if result else None
            status = "✅" if intent == expected else "❌"
            print(f"{status} '{text}' → {intent} (attendu: {expected})")
            if intent == expected:
                passed += 1
        except Empty:
            print(f"❌ '{text}' → TIMEOUT")

    pipeline.stop()
    print(f"\nRésultat: {passed}/{len(test_phrases)} OK")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.interactive:
        q_in = Queue()
        q_out = Queue()
        pipeline = IntentPipeline(q_in, q_out, debug=True)
        pipeline.load()
        pipeline.start()
        print("Mode interactif (tapez 'quit' pour quitter)")
        while True:
            text = input("\n> ").strip()
            if text.lower() in ("quit", "exit"):
                break
            q_in.put(text)
            try:
                result = q_out.get(timeout=5.0)
                for intent in result:
                    print(f"→ {intent['intent']} ({intent['confidence']:.2f})")
                    if intent.get("raw_args"):
                        print(f"   raw_args: {intent['raw_args']}")
            except Empty:
                print("→ TIMEOUT")
        pipeline.stop()
    else:
        print("Usage: python tti.py --test  ou  --interactive")