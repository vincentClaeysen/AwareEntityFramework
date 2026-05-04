#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intent_pipeline_v4.py — Pipeline NLU français → Intent(s) structuré(s)
=======================================================================
Fusion v3 + v6. Raspberry Pi 5, 100 % offline.

De v3 (conservé intégralement) :
  - SEMANTIC_MAP 224 verbes → 11 concepts
  - FrenchVerbAnalyzer : irréguliers, formes composées, pronominal, impersonnel
  - VerbAnalysis complet : person/number/mood/polarity/modal/pronominal/impersonal/compound
  - _extract_all_slots : une seule passe spaCy
  - split_clauses : multi-intent ("je jardinerai ET je lirai" → 2 intents)
  - ConversationFrame : héritage de slots inter-phrases + fragments
  - TIMEX3 standard + PREFER_DATES_FROM conditionné au tense
  - _extract_interval "entre X et Y"
  - EventGraph + EventGraphConsumer
  - run_tests() + run_graph_tests()
  - IntentType 6 classes métier
  - Entraînement CamemBERT avec early stopping

De v6 (intégré) :
  - FrenchTemporalResolver : moments/saisons/jours fériés/vacances scolaires/
                              jours de semaine/relatifs/durées + cache LRU
                              → remplace _extract_temporal (plus riche)
  - ConversationContext : résolution ellipses + pronoms anaphoriques
                          → couche au-dessus de ConversationFrame
  - HybridIntentClassifier : ML-first + fallback règles (démarrage sans modèle)
  - SyntacticTreeExtractor + Participant.role (user/ase/other)
  - Intent.to_cognitive_frame() pour CognitionCore
  - PipelineMetrics : avg_ms, distribution intents, taux d'erreur
  - batch_process : traitement parallèle ThreadPoolExecutor
  - Mode debug : affichage step-by-step
  - Error logger séparé (pipeline_errors.log)
  - Imports optionnels avec fallbacks (workalendar, holidays, vacances_scolaires…)

Architecture :
    FrenchTextCorrector      → normalisation SMS/dyslexie
    ConversationContext       → ellipses + pronoms (couche haute)
    ConversationFrame         → héritage slots inter-phrases (couche basse)
    CamemBERT (lazy)         → intent_type + confidence
    HybridIntentClassifier   → ML-first + fallback règles
    spaCy (1 passe)          → who/with_who/where/entities + arbre syntaxique
    FrenchVerbAnalyzer       → irréguliers + formes composées
    _extract_verb            → tense+scope+concept+mood+person+polarity+modal
    FrenchTemporalResolver   → résolution temporelle complète + cache LRU
    RegisterAnalyzer         → registre lexical (5 styles)
    split_clauses            → multi-intent
    PipelineMetrics          → métriques temps réel
    EventGraph               → mémoire épisodique
    EventGraphConsumer       → thread consommateur

API :
    q_in  = Queue()
    q_out = Queue()      ← listes d'Intent dict

    pipeline = IntentPipeline(q_in, q_out)
    pipeline.load()
    pipeline.start()
    q_in.put("demain matin pendant 2h")
    q_in.put("je jardinerai et je lirai ensuite")
    # → q_out.get() == [Intent(jardiner), Intent(lire)]  ← multi-intent

    # Batch
    results = pipeline.batch_process(["phrase1", "phrase2"])

    # Métriques
    print(pipeline.get_metrics())
"""

import datetime
import json
import logging
import re
import threading
import time
import argparse
import unicodedata
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

# ============================================================
# Imports optionnels avec fallbacks
# ============================================================

try:
    import torch
    from transformers import CamembertTokenizer, CamembertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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
    import holidays as holidays_lib
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
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-14s] %(levelname)-8s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger      = logging.getLogger("IntentPipeline")
error_logger = logging.getLogger("IntentPipeline.Errors")
_err_handler = logging.FileHandler("pipeline_errors.log")
_err_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
error_logger.addHandler(_err_handler)
error_logger.setLevel(logging.WARNING)

# ============================================================
# Types stricts
# ============================================================

IntentType   = Literal["query_narrative","query_state","query_intention",
                        "action_device","information_input","chit_chat"]
TenseType    = Literal["past","present","future","conditional","unknown"]
PersonType   = Literal["1st_sg","2nd_sg","3rd_sg","1st_pl","2nd_pl","3rd_pl","unknown"]
MoodType     = Literal["indicative","subjunctive","imperative","conditional","infinitive"]
PolarityType = Literal["positive","negative"]
ModalType    = Literal["obligation","possibility","volition","wish",None]
ConceptType  = Literal["PERCEIVE","COMMUNICATE","BE","MOVE","INGEST","CREATE",
                        "TRANSFER","COGNITION","EMOTION","SOCIAL_ACT","HEALTH",None]
ScopeType    = Literal["PAST","PRESENT","FUTURE","HYPOTHETICAL","UNKNOWN"]
RoleType     = Literal["user","ase","other"]

# ============================================================
# Configuration
# ============================================================

MODEL_NAME              = "camembert-base"
MODEL_DIR               = Path("./model_camembert")
DATA_FILE               = Path("./training_data.json")
SPACY_MODEL             = "fr_core_news_sm"
LANGUAGE                = "fr"
MAX_LENGTH              = 64
BATCH_SIZE              = 8
EPOCHS                  = 10
LEARNING_RATE           = 2e-5
WARMUP_RATIO            = 0.1
TEST_SPLIT              = 0.15
CONFIDENCE_THRESHOLD    = 0.60
CONTEXT_RESET_SEC       = 30.0
EARLY_STOPPING_PATIENCE = 3
CACHE_SIZE              = 256
MAX_WORKERS             = 2          # threads batch (RPi5 : 4 cœurs)
INPUT_QUEUE_SIZE        = 50
OUTPUT_QUEUE_SIZE       = 50

INTENTS: List[IntentType] = [
    "query_narrative","query_state","query_intention",
    "action_device","information_input","chit_chat",
]

NER_TYPE_MAP = {
    "CARDINAL":"number","ORDINAL":"ordinal","QUANTITY":"quantity",
    "PERCENT":"percent","MONEY":"money","TIME":"time_ref","DATE":"date_ref",
    "DURATION":"duration","NORP":"group","FAC":"facility","ORG":"organization",
    "PER":"person","LOC":"location","GPE":"location","PRODUCT":"product",
    "EVENT":"event","LANGUAGE":"language","MISC":"misc",
}

UNIT_PATTERNS = {
    "km":"km","kilometre":"km","kilometres":"km",
    "m":"m","metre":"m","metres":"m","cm":"cm",
    "kg":"kg","kilo":"kg","kilos":"kg",
    "g":"g","gramme":"g","grammes":"g",
    "l":"L","litre":"L","litres":"L","cl":"cL","ml":"mL",
    "h":"h","heure":"h","heures":"h",
    "min":"min","minute":"min","minutes":"min",
    "s":"s","seconde":"s","secondes":"s",
    "degre":"celsius","degres":"celsius",
    "euro":"EUR","euros":"EUR","%":"%",
}

INFO_SUBJECTS = {"je","j'","j","moi","nous","on"}

CLITIC_TO_PERSON: Dict[str, PersonType] = {
    "me":"1st_sg","m'":"1st_sg","moi":"1st_sg","m":"1st_sg",
    "te":"2nd_sg","t'":"2nd_sg","toi":"2nd_sg","t":"2nd_sg",
    "se":"3rd_sg","s'":"3rd_sg","soi":"3rd_sg","s":"3rd_sg",
    "lui":"3rd_sg","elle":"3rd_sg",
    "nous":"1st_pl","on":"1st_pl",
    "vous":"2nd_pl",
    "leur":"3rd_pl","eux":"3rd_pl","elles":"3rd_pl",
}

MODAL_MARKERS: Dict[str, ModalType] = {
    "devoir":"obligation","falloir":"obligation","obliger":"obligation",
    "pouvoir":"possibility",
    "vouloir":"volition","souhaiter":"volition","désirer":"volition","espérer":"volition",
    "aimer":"wish",
}

TENSE_TO_SCOPE: Dict[str, ScopeType] = {
    "past":"PAST","present":"PRESENT","future":"FUTURE",
    "conditional":"HYPOTHETICAL","unknown":"UNKNOWN",
}

ACTION_VERBS: Dict[str, str] = {
    "allumer":"turn_on","activer":"turn_on","lancer":"turn_on",
    "ouvrir":"turn_on","démarrer":"turn_on",
    "éteindre":"turn_off","couper":"turn_off","désactiver":"turn_off",
    "arrêter":"turn_off","fermer":"turn_off","stopper":"turn_off",
    "monter":"set_up","augmenter":"set_up",
    "baisser":"set_down","diminuer":"set_down","réduire":"set_down",
    "régler":"set","mettre":"set","configurer":"set","programmer":"set",
}

DEVICE_NOUNS: Dict[str, str] = {
    "lumière":"light","lampe":"light","éclairage":"light",
    "plafonnier":"light","spot":"light",
    "télé":"tv","télévision":"tv","écran":"tv",
    "musique":"music","son":"audio","volume":"audio",
    "chauffage":"heating","radiateur":"heating","thermostat":"heating",
    "volet":"shutter","store":"shutter","rideau":"shutter",
    "porte":"door","fenêtre":"door",
    "alarme":"alarm","climatisation":"ac","clim":"ac","ventilateur":"fan",
}

# ============================================================
# SEMANTIC_MAP — 224 verbes, 11 concepts (v3)
# ============================================================

SEMANTIC_MAP: Dict[str, ConceptType] = {
    # PERCEIVE
    "voir":"PERCEIVE","regarder":"PERCEIVE","entendre":"PERCEIVE",
    "écouter":"PERCEIVE","sentir":"PERCEIVE","remarquer":"PERCEIVE",
    "observer":"PERCEIVE","apercevoir":"PERCEIVE","percevoir":"PERCEIVE",
    "distinguer":"PERCEIVE","noter":"PERCEIVE","détecter":"PERCEIVE",
    "repérer":"PERCEIVE","surveiller":"PERCEIVE","examiner":"PERCEIVE",
    "inspecter":"PERCEIVE","scruter":"PERCEIVE",
    # COMMUNICATE
    "dire":"COMMUNICATE","parler":"COMMUNICATE","appeler":"COMMUNICATE",
    "répondre":"COMMUNICATE","demander":"COMMUNICATE","expliquer":"COMMUNICATE",
    "raconter":"COMMUNICATE","discuter":"COMMUNICATE","écrire":"COMMUNICATE",
    "communiquer":"COMMUNICATE","téléphoner":"COMMUNICATE","annoncer":"COMMUNICATE",
    "informer":"COMMUNICATE","mentionner":"COMMUNICATE","préciser":"COMMUNICATE",
    "signaler":"COMMUNICATE","crier":"COMMUNICATE","chuchoter":"COMMUNICATE",
    "réclamer":"COMMUNICATE","négocier":"COMMUNICATE","interviewer":"COMMUNICATE",
    "interroger":"COMMUNICATE","contacter":"COMMUNICATE","textoter":"COMMUNICATE",
    # BE
    "être":"BE","rester":"BE","demeurer":"BE","trouver":"BE","exister":"BE",
    "sembler":"BE","paraître":"BE","devenir":"BE","apparaître":"BE","s'avérer":"BE",
    # MOVE
    "aller":"MOVE","venir":"MOVE","bouger":"MOVE","sortir":"MOVE","entrer":"MOVE",
    "partir":"MOVE","arriver":"MOVE","quitter":"MOVE","marcher":"MOVE",
    "courir":"MOVE","rentrer":"MOVE","revenir":"MOVE","monter":"MOVE",
    "descendre":"MOVE","passer":"MOVE","traverser":"MOVE","déplacer":"MOVE",
    "voyager":"MOVE","circuler":"MOVE","conduire":"MOVE","rouler":"MOVE",
    "voler":"MOVE","nager":"MOVE","grimper":"MOVE","reculer":"MOVE",
    "avancer":"MOVE","fuir":"MOVE","s'approcher":"MOVE","s'éloigner":"MOVE",
    # INGEST
    "manger":"INGEST","boire":"INGEST","avaler":"INGEST","consommer":"INGEST",
    "goûter":"INGEST","grignoter":"INGEST","déguster":"INGEST","dîner":"INGEST",
    "déjeuner":"INGEST","cuisiner":"INGEST","préparer":"INGEST",
    "ingérer":"INGEST","absorber":"INGEST",
    # CREATE
    "faire":"CREATE","fabriquer":"CREATE","construire":"CREATE","créer":"CREATE",
    "produire":"CREATE","réaliser":"CREATE","concevoir":"CREATE","dessiner":"CREATE",
    "peindre":"CREATE","sculpter":"CREATE","composer":"CREATE","rédiger":"CREATE",
    "coder":"CREATE","programmer":"CREATE","bricoler":"CREATE","tricoter":"CREATE",
    "coudre":"CREATE","jardiner":"CREATE","planter":"CREATE","aménager":"CREATE",
    # TRANSFER
    "donner":"TRANSFER","envoyer":"TRANSFER","recevoir":"TRANSFER",
    "apporter":"TRANSFER","remettre":"TRANSFER","transmettre":"TRANSFER",
    "offrir":"TRANSFER","prêter":"TRANSFER","emprunter":"TRANSFER",
    "rendre":"TRANSFER","poster":"TRANSFER","livrer":"TRANSFER",
    "expédier":"TRANSFER","partager":"TRANSFER","distribuer":"TRANSFER",
    "vendre":"TRANSFER","acheter":"TRANSFER",
    # COGNITION
    "penser":"COGNITION","croire":"COGNITION","savoir":"COGNITION",
    "comprendre":"COGNITION","oublier":"COGNITION","apprendre":"COGNITION",
    "réfléchir":"COGNITION","imaginer":"COGNITION","supposer":"COGNITION",
    "douter":"COGNITION","ignorer":"COGNITION","réaliser":"COGNITION",
    "considérer":"COGNITION","estimer":"COGNITION","juger":"COGNITION",
    "analyser":"COGNITION","décider":"COGNITION","choisir":"COGNITION",
    "planifier":"COGNITION","mémoriser":"COGNITION","se souvenir":"COGNITION",
    "calculer":"COGNITION","étudier":"COGNITION","lire":"COGNITION",
    # EMOTION
    "aimer":"EMOTION","adorer":"EMOTION","détester":"EMOTION",
    "craindre":"EMOTION","espérer":"EMOTION","vouloir":"EMOTION",
    "désirer":"EMOTION","regretter":"EMOTION","souffrir":"EMOTION",
    "pleurer":"EMOTION","rire":"EMOTION","s'énerver":"EMOTION",
    "s'inquiéter":"EMOTION","apprécier":"EMOTION","ressentir":"EMOTION",
    "se sentir":"EMOTION","se réjouir":"EMOTION","se plaindre":"EMOTION",
    "s'ennuyer":"EMOTION","stresser":"EMOTION",
    # SOCIAL_ACT
    "saluer":"SOCIAL_ACT","remercier":"SOCIAL_ACT","s'excuser":"SOCIAL_ACT",
    "promettre":"SOCIAL_ACT","féliciter":"SOCIAL_ACT","inviter":"SOCIAL_ACT",
    "refuser":"SOCIAL_ACT","accepter":"SOCIAL_ACT","proposer":"SOCIAL_ACT",
    "voter":"SOCIAL_ACT","signer":"SOCIAL_ACT","jurer":"SOCIAL_ACT",
    "rencontrer":"SOCIAL_ACT","retrouver":"SOCIAL_ACT",
    "présenter":"SOCIAL_ACT","accueillir":"SOCIAL_ACT",
    # HEALTH
    "dormir":"HEALTH","se reposer":"HEALTH","tomber":"HEALTH",
    "se blesser":"HEALTH","soigner":"HEALTH","guérir":"HEALTH",
    "opérer":"HEALTH","consulter":"HEALTH","vacciner":"HEALTH",
    "exercer":"HEALTH","respirer":"HEALTH","tousser":"HEALTH",
    "éternuer":"HEALTH","se rétablir":"HEALTH","transpirer":"HEALTH",
}

# Registre lexical
REGISTER_LEXICON = {
    "familier": {
        "ouais","nan","chais","jsuis","j'suis","ya","y'a","wesh","bah","ben",
        "quoi","hein","machin","truc","bidule","trop","grave","vachement",
        "carrément","voilà","genre","sympa","chouette","cool","super","nul",
        "relou","chelou","kiffer","kiffe","ouf","zarbi","nickel","tranquille",
        "ok","oki","mouais","pfff","lol","mdr","ptdr","faut","t'inquiète",
    },
    "soutenu": {
        "néanmoins","cependant","toutefois","ainsi","également","certes",
        "quoique","nonobstant","afin","lequel","laquelle","auquel","duquel",
        "dont","ledit","ladite","susmentionné","permettez","veuillez","daigner",
        "souhaiteriez","pourriez","daignez","sauriez","conviendrait","s'avère",
        "appréhender","concevoir","envisager","considérer","évoquer",
    },
    "affectif": {
        "oh","ah","eh","hé","hélas","malheureusement","heureusement",
        "magnifique","horrible","terrible","merveilleux","fantastique",
        "incroyable","extraordinaire","adorable","détestable",
        "tellement","vraiment","absolument","totalement",
    },
    "technique": {
        "paramètre","configuration","instance","module","composant",
        "interface","protocole","algorithme","fonction","variable",
        "système","process","thread","queue","buffer","pipeline",
        "latence","throughput","overhead","benchmark","optimiser",
        "déployer","intégrer","implémenter","requête","payload",
    },
}

NEGATION_COMPLETE   = {"ne","n'"}
TUTOIEMENT_MARKERS  = {"tu","toi","t'","te","ton","ta","tes"}
VOUVOIEMENT_MARKERS = {"vous","votre","vos"}

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

_DATE_PARSER      = None
_TIMEXY_CHECKED   = False
_TIMEXY_AVAILABLE = False

COMMON_WORDS_FR = {
    "je","tu","il","elle","on","nous","vous","ils","elles",
    "le","la","les","un","une","des","du","de","et","ou","mais",
    "donc","car","est","sont","était","ai","as","a","avons","avez","ont",
    "vais","vas","va","peux","peut","veux","veut","fais","fait","dis","dit",
    "pour","par","avec","sans","dans","sur","sous",
}


# ============================================================
# SECTION 1 — FrenchTextCorrector (v3)
# ============================================================

class FrenchTextCorrector:
    """Normalise le texte avant analyse NLU. SMS, dyslexie, fautes phonétiques."""

    COMMON_MISTAKES: Dict[str, str] = {
        "sava":"ça va","cetais":"c'était","cété":"c'était",
        "jsp":"je sais pas","pk":"pourquoi","pcq":"parce que",
        "dc":"donc","tkt":"ne t'en fais pas","chui":"je suis",
        "ajd":"aujourd'hui","demin":"demain","yer":"hier",
        "vien":"viens","veu":"veux","peu":"peux","fé":"fais",
        "pa":"pas","jamait":"jamais","mes":"mais","don":"donc",
        "dan":"dans","aveq":"avec","tré":"très",
        "jé mangé":"j'ai mangé",
    }

    _CLEANUP = [
        (re.compile(r'\s+([?.!,;:])'), r'\1'),
        (re.compile(r'([?.!,;:])([^\s])'), r'\1 \2'),
        (re.compile(r'\s+'), ' '),
    ]

    _cache: Dict[str, Tuple[str, List[Tuple[str,str]]]] = {}

    @classmethod
    def correct(cls, text: str) -> Tuple[str, List[Tuple[str,str]]]:
        if not text:
            return text, []
        if text in cls._cache:
            return cls._cache[text]

        corrections: List[Tuple[str,str]] = []
        words = text.lower().split()
        out: List[str] = []
        i = 0

        while i < len(words):
            matched = False
            for error, fix in cls.COMMON_MISTAKES.items():
                err_words = error.split()
                n = len(err_words)
                if words[i:i+n] == err_words:
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
# SECTION 2 — FrenchVerbAnalyzer (v3)
# ============================================================

class FrenchVerbAnalyzer:
    """Enrichit l'analyse verbale spaCy : irréguliers, formes composées, pronominal."""

    IRREGULAR: Dict[str, Tuple[str, str, str]] = {
        "aller":    ("vais/vas/va/allons/allez/vont",     "allai",  "allé"),
        "avoir":    ("ai/as/a/avons/avez/ont",            "eus",    "eu"),
        "être":     ("suis/es/est/sommes/êtes/sont",      "fus",    "été"),
        "faire":    ("fais/fait/faisons/faites/font",     "fis",    "fait"),
        "dire":     ("dis/dit/disons/dites/disent",       "dis",    "dit"),
        "venir":    ("viens/vient/venons/venez/viennent", "vins",   "venu"),
        "tenir":    ("tiens/tient/tenons/tenez/tiennent", "tins",   "tenu"),
        "voir":     ("vois/voit/voyons/voyez/voient",     "vis",    "vu"),
        "savoir":   ("sais/sait/savons/savez/savent",     "sus",    "su"),
        "pouvoir":  ("peux/peut/pouvons/pouvez/peuvent",  "pus",    "pu"),
        "vouloir":  ("veux/veut/voulons/voulez/veulent",  "voulus", "voulu"),
        "devoir":   ("dois/doit/devons/devez/doivent",    "dus",    "dû"),
        "falloir":  ("faut",                              "fallut", "fallu"),
        "pleuvoir": ("pleut",                             "plut",   "plu"),
    }

    _AUX_AVOIR = {"ai","as","a","avons","avez","ont"}
    _AUX_ETRE  = {"suis","es","est","sommes","êtes","sont"}

    @classmethod
    def is_compound(cls, token_text: str) -> Optional[str]:
        t = token_text.lower()
        if t in cls._AUX_AVOIR: return "avoir"
        if t in cls._AUX_ETRE:  return "être"
        return None

    @classmethod
    def is_pronominal(cls, doc, verb_token) -> bool:
        for child in verb_token.children:
            if child.dep_ in ("expl","expl:comp") and child.text.lower() in (
                "se","s'","me","te","nous","vous"
            ):
                return True
        return False

    @classmethod
    def is_impersonal(cls, lemma: str) -> bool:
        return lemma in ("falloir","pleuvoir","neiger","grêler","tonner","venter")

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
# SECTION 3 — TemporalSpan (TIMEX3)
# ============================================================

@dataclass
class TemporalSpan:
    """Expression temporelle normalisée TIMEX3 : DATE|TIME|DURATION|SET|INTERVAL."""
    raw:                str
    iso_start:          str
    iso_end:            str
    timex_type:         str   = "DATE"
    source:             str   = "dateparser"
    duration_raw:       Optional[str]   = None
    duration_unit:      Optional[str]   = None
    duration_value:     Optional[float] = None
    until_raw:          Optional[str]   = None
    interval_start_raw: Optional[str]   = None
    interval_end_raw:   Optional[str]   = None
    named_event:        Optional[dict]  = None   # v6 : saison, férié, vacances…


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


# ============================================================
# SECTION 4 — FrenchTemporalResolver (v6, LRU cache)
# ============================================================

class FrenchTemporalResolver:
    """
    Résout les expressions temporelles françaises.
    Couverture : moments, saisons, jours fériés, vacances scolaires,
    jours de semaine, jours relatifs, durées, intervals, dateparser.
    Cache LRU pour les expressions répétées.
    """

    MOMENTS = {
        "midi":        (12, 12, "point"),
        "minuit":      (0,  0,  "point"),
        "matin":       (6,  12, "interval"),
        "matinée":     (6,  12, "interval"),
        "après-midi":  (12, 18, "interval"),
        "soir":        (18, 22, "interval"),
        "soirée":      (18, 22, "interval"),
        "nuit":        (22, 6,  "interval"),
        "journée":     (8,  20, "interval"),
        "aube":        (5,  7,  "interval"),
        "crépuscule":  (18, 20, "interval"),
    }

    MOMENT_PATTERNS = {
        "midi":       ["midi","à midi","12h"],
        "minuit":     ["minuit","à minuit","0h"],
        "matin":      ["matin","ce matin","dans la matinée"],
        "après-midi": ["après-midi","cet après-midi"],
        "soir":       ["soir","ce soir","dans la soirée","ce soir"],
        "nuit":       ["nuit","cette nuit","pendant la nuit"],
        "journée":    ["journée","cette journée","toute la journée"],
    }

    SEASONS = {
        "printemps": {"months":[3,4,5],  "emoji":"🌸"},
        "été":       {"months":[6,7,8],  "emoji":"☀️"},
        "automne":   {"months":[9,10,11],"emoji":"🍂"},
        "hiver":     {"months":[12,1,2], "emoji":"❄️"},
    }

    SEASON_PATTERNS = {
        "printemps": ["printemps","au printemps","ce printemps"],
        "été":       ["été","cet été","en été"],
        "automne":   ["automne","cet automne","en automne"],
        "hiver":     ["hiver","cet hiver","en hiver"],
    }

    RELATIVE_DAYS = {
        "aujourd'hui":0,"demain":1,"après-demain":2,
        "hier":-1,"avant-hier":-2,
    }

    WEEKDAYS = {
        "lundi":0,"mardi":1,"mercredi":2,"jeudi":3,
        "vendredi":4,"samedi":5,"dimanche":6,
    }

    HOLIDAY_NAMES = {
        "nouvel an":"01-01","1er janvier":"01-01",
        "1er mai":"05-01","fête du travail":"05-01",
        "8 mai":"05-08","victoire":"05-08",
        "14 juillet":"07-14","fête nationale":"07-14",
        "15 août":"08-15","assomption":"08-15",
        "1er novembre":"11-01","toussaint":"11-01",
        "11 novembre":"11-11","armistice":"11-11",
        "noël":"12-25","25 décembre":"12-25",
    }

    def __init__(self):
        self._calendar = FranceCalendar() if WORKALENDAR_AVAILABLE else None
        self._stats: Dict[str,int] = {"hits":0,"misses":0}

    @staticmethod
    def _duration_iso(start: datetime.datetime, end: datetime.datetime) -> str:
        sec = int((end - start).total_seconds())
        if sec % 86400 == 0: return f"P{sec//86400}D"
        if sec % 3600  == 0: return f"PT{sec//3600}H"
        if sec % 60    == 0: return f"PT{sec//60}M"
        return f"PT{sec}S"

    @lru_cache(maxsize=CACHE_SIZE)
    def _cached_resolve(self, text: str, ref_iso: str) -> Optional[dict]:
        """Résolution avec cache LRU — clé = (text, ref_iso)."""
        self._stats["misses"] += 1
        ref = datetime.datetime.fromisoformat(ref_iso)
        return self._resolve_uncached(text, ref)

    def resolve(self, text: str, ref: Optional[datetime.datetime] = None) -> Optional[dict]:
        """Résout une expression temporelle et retourne un dict compatible TemporalSpan."""
        if ref is None:
            ref = datetime.datetime.now()
        result = self._cached_resolve(text.lower(), ref.isoformat())
        if result:
            self._stats["hits"] += 1
        return result

    def _resolve_uncached(self, text: str, ref: datetime.datetime) -> Optional[dict]:
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
                return r
        return None

    def _moment(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        for name, patterns in self.MOMENT_PATTERNS.items():
            if not any(p in text for p in patterns):
                continue
            offset = (-2 if "avant-hier" in text else
                      -1 if "hier"        in text else
                       2 if "après-demain" in text else
                       1 if "demain"      in text else 0)
            base = ref + datetime.timedelta(days=offset)
            sh, eh, mtype = self.MOMENTS[name]
            named = {"name":name,"type":"moment_of_day","day_offset":offset}
            if mtype == "point":
                dt  = base.replace(hour=sh, minute=0, second=0, microsecond=0)
                iso = dt.isoformat()
                return {"raw":text,"timex_type":"TIME","iso_start":iso,"iso_end":iso,
                        "source":"moment_rule","named_event":named}
            start = base.replace(hour=sh, minute=0, second=0, microsecond=0)
            end   = base.replace(hour=eh, minute=0, second=0, microsecond=0)
            if eh < sh: end += datetime.timedelta(days=1)
            return {"raw":text,"timex_type":"INTERVAL",
                    "iso_start":start.isoformat(),"iso_end":end.isoformat(),
                    "duration_iso":self._duration_iso(start,end),
                    "source":"moment_rule","named_event":named}
        return None

    def _season(self, text: str, year: int) -> Optional[dict]:
        for season, patterns in self.SEASON_PATTERNS.items():
            if not any(p in text for p in patterns):
                continue
            delta = (1 if "prochain" in text or "prochaine" in text else
                    -1 if "dernier"  in text or "dernière"  in text or "passé" in text else 0)
            ty = year + delta
            months = self.SEASONS[season]["months"]
            if season == "hiver" and delta == 0:
                start = datetime.datetime(ty-1, 12, 1)
                end   = datetime.datetime(ty, 3, 1) - datetime.timedelta(days=1)
            else:
                start = datetime.datetime(ty, months[0], 1)
                last  = months[-1]
                end_m = datetime.datetime(ty, last+1, 1) if last < 12 else datetime.datetime(ty+1,1,1)
                end   = end_m - datetime.timedelta(days=1)
            named = {"name":season,"type":"season","year":ty,"emoji":self.SEASONS[season]["emoji"]}
            return {"raw":text,"timex_type":"INTERVAL",
                    "iso_start":start.isoformat(),"iso_end":end.isoformat(),
                    "duration_iso":self._duration_iso(start,end),
                    "source":"season_rule","named_event":named}
        return None

    def _holiday(self, text: str, year: int) -> Optional[dict]:
        for name, md in self.HOLIDAY_NAMES.items():
            if name in text:
                m, d = map(int, md.split("-"))
                dt = datetime.datetime(year, m, d)
                return {"raw":text,"timex_type":"DATE",
                        "iso_start":dt.isoformat(),"iso_end":dt.isoformat(),
                        "source":"holiday_rule",
                        "named_event":{"name":name,"type":"french_holiday"}}
        if ("paques" in text or "pâques" in text) and self._calendar:
            e = self._calendar.get_easter(year)
            dt = datetime.datetime(e.year, e.month, e.day)
            return {"raw":text,"timex_type":"DATE",
                    "iso_start":dt.isoformat(),"iso_end":dt.isoformat(),
                    "source":"workalendar",
                    "named_event":{"name":"Pâques","type":"movable_holiday"}}
        return None

    def _school_holiday(self, text: str, year: int) -> Optional[dict]:
        if not VACANCES_AVAILABLE:
            return None
        holiday_map = {
            "vacances de noël":"Noël","vacances d'hiver":"Hiver",
            "vacances de printemps":"Printemps","vacances de pâques":"Printemps",
            "vacances d'été":"Été","grandes vacances":"Été",
            "vacances de la toussaint":"Toussaint",
        }
        for key, hname in holiday_map.items():
            if key not in text:
                continue
            for zone in ("A","B","C"):
                try:
                    for h in get_school_holidays(year, zone):
                        if h["name"] == hname:
                            s = datetime.datetime.fromisoformat(h["start_date"])
                            e = datetime.datetime.fromisoformat(h["end_date"])
                            return {"raw":text,"timex_type":"INTERVAL",
                                    "iso_start":s.isoformat(),"iso_end":e.isoformat(),
                                    "duration_iso":self._duration_iso(s,e),
                                    "source":"vacances_scolaires",
                                    "named_event":{"name":hname,"type":"school_holiday","zone":zone}}
                except Exception:
                    continue
        return None

    def _weekday(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        for day, wd in self.WEEKDAYS.items():
            if day not in text:
                continue
            ahead = (wd - ref.weekday()) % 7 or 7
            dt    = ref + datetime.timedelta(days=ahead)
            s = dt.replace(hour=0,  minute=0, second=0, microsecond=0)
            e = dt.replace(hour=23, minute=59,second=59, microsecond=0)
            return {"raw":text,"timex_type":"DATE",
                    "iso_start":s.isoformat(),"iso_end":e.isoformat(),
                    "source":"weekday_rule",
                    "named_event":{"name":day,"type":"weekday"}}
        return None

    def _relative_day(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        for name, offset in self.RELATIVE_DAYS.items():
            if name not in text:
                continue
            dt = ref + datetime.timedelta(days=offset)
            s  = dt.replace(hour=0,  minute=0, second=0, microsecond=0)
            e  = dt.replace(hour=23, minute=59,second=59, microsecond=0)
            return {"raw":text,"timex_type":"DATE",
                    "iso_start":s.isoformat(),"iso_end":e.isoformat(),
                    "source":"relative_day_rule",
                    "named_event":{"name":name,"type":"relative_day","offset":offset}}
        return None

    def _duration(self, text: str) -> Optional[dict]:
        patterns = [
            (r"pendant\s+(\d+(?:[.,]\d+)?)\s*(heures?|h)",    3600,   "H"),
            (r"pendant\s+(\d+(?:[.,]\d+)?)\s*(minutes?|min)", 60,     "M"),
            (r"pendant\s+(\d+(?:[.,]\d+)?)\s*(secondes?|s)",  1,      "S"),
            (r"(\d+(?:[.,]\d+)?)\s*(jours?|j(?:\s|$))",       86400,  "D"),
            (r"(\d+(?:[.,]\d+)?)\s*(semaines?|sem)",          604800, "W"),
        ]
        for pat, spu, unit in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = float(m.group(1).replace(",","."))
                total = val * spu
                return {"raw":m.group(0),"timex_type":"DURATION",
                        "iso_start":"","iso_end":"",
                        "duration_value":val,"duration_unit":unit,
                        "duration_iso":f"PT{int(val)}{unit}",
                        "source":"duration_rule",
                        "named_event":{"type":"duration","unit":unit}}
        return None

    def _fallback_dateparser(self, text: str, ref: datetime.datetime) -> Optional[dict]:
        if not DATEPARSER_AVAILABLE:
            return None
        dp = _get_dateparser()
        try:
            dt = dp.parse(text, languages=[LANGUAGE],
                          settings={"RELATIVE_BASE":ref,
                                    "RETURN_AS_TIMEZONE_AWARE":True,
                                    "TIMEZONE":"Europe/Paris"})
            if dt:
                iso = dt.isoformat()
                return {"raw":text,"timex_type":"DATE",
                        "iso_start":iso,"iso_end":iso,"source":"dateparser"}
        except Exception:
            pass
        return None

    def to_temporal_span(self, d: dict, tense: str = "unknown") -> Optional[TemporalSpan]:
        """Convertit un dict résolu en TemporalSpan compatible v3."""
        if not d or not d.get("iso_start") and d.get("timex_type") != "DURATION":
            return None
        return TemporalSpan(
            raw=d.get("raw",""),
            iso_start=d.get("iso_start",""),
            iso_end=d.get("iso_end",""),
            timex_type=d.get("timex_type","DATE"),
            source=d.get("source",""),
            duration_raw=d.get("raw") if d.get("timex_type")=="DURATION" else None,
            duration_unit=d.get("duration_unit"),
            duration_value=d.get("duration_value"),
            named_event=d.get("named_event"),
        )

    def get_stats(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "cache_hits":   self._stats["hits"],
            "cache_misses": self._stats["misses"],
            "hit_rate":     round(self._stats["hits"]/total, 3) if total else 0,
        }


# Singleton partagé
_temporal_resolver: Optional[FrenchTemporalResolver] = None

def _get_temporal_resolver() -> FrenchTemporalResolver:
    global _temporal_resolver
    if _temporal_resolver is None:
        _temporal_resolver = FrenchTemporalResolver()
    return _temporal_resolver


# ============================================================
# SECTION 5 — ConversationContext (v6) + ConversationFrame (v3)
# ============================================================

@dataclass
class ConversationContext:
    """
    Couche haute : résolution des ellipses et pronoms anaphoriques.
    Distinct de ConversationFrame qui hérite les slots inter-phrases.
    """
    last_intent_type: Optional[IntentType] = None
    last_verb_lemma:  Optional[str]        = None
    last_actor:       Optional[str]        = None
    last_action:      Optional[str]        = None
    last_location:    Optional[str]        = None
    last_when_iso:    Optional[str]        = None
    timestamp:        float                = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > CONTEXT_RESET_SEC

    def update_from_intent(self, intent: "Intent"):
        self.last_intent_type = intent.intent
        self.last_verb_lemma  = intent.verb.lemma if intent.verb else None
        self.last_actor       = intent.who_raw
        self.last_action      = intent.action or (intent.verb.lemma if intent.verb else None)
        self.last_location    = intent.where
        self.last_when_iso    = intent.when.iso_start if intent.when else None
        self.timestamp        = time.time()

    def resolve_ellipsis(self, text: str) -> str:
        if self.is_expired():
            return text
        tl = text.lower().strip()
        if tl in ("oui","ouais","ok","d'accord","okay","bien sûr"):
            if self.last_intent_type == "query_narrative" and self.last_verb_lemma:
                return f"oui, {self.last_verb_lemma}"
        if tl in ("encore","recommence","refais","re"):
            if self.last_action:
                return f"refais {self.last_action}"
        return text

    def resolve_pronouns(self, text: str) -> str:
        if self.is_expired():
            return text
        result = text
        if re.search(r'\by\b', result) and self.last_location:
            result = re.sub(r'\by\b', self.last_location, result)
        if re.search(r'\ben\b', result) and self.last_action:
            result = re.sub(r'\ben\b', f"de {self.last_action}", result)
        return result


@dataclass
class ConversationFrame:
    """Couche basse : héritage de slots (who/when/where) entre fragments."""
    who:               Optional[PersonType]   = None
    who_raw:           Optional[str]          = None
    with_who:          List[str]              = field(default_factory=list)
    when:              Optional[TemporalSpan] = None
    where:             Optional[str]          = None
    what:              Optional[ConceptType]  = None
    last_update:       float                  = field(default_factory=time.monotonic)
    pending_fragments: List[str]              = field(default_factory=list)
    subjects:          List[str]              = field(default_factory=list)
    durations:         List[dict]             = field(default_factory=list)
    register:          Optional[str]          = None

    def is_expired(self) -> bool:
        return time.monotonic() - self.last_update > CONTEXT_RESET_SEC

    def reset(self):
        self.who = self.who_raw = self.when = self.where = self.what = self.register = None
        self.with_who = self.pending_fragments = self.subjects = self.durations = []  # type: ignore
        self.last_update = time.monotonic()

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k) or v is None:
                continue
            if k == "with_who":
                self.with_who = list(set(self.with_who + v))
            elif k == "fragment":
                self.pending_fragments.append(v)
            else:
                setattr(self, k, v)
        self.last_update = time.monotonic()

    def flush_pending(self) -> List[str]:
        frags, self.pending_fragments = list(self.pending_fragments), []
        return frags


# ============================================================
# SECTION 6 — Structures de données
# ============================================================

@dataclass
class VerbAnalysis:
    lemma:      str
    tense:      TenseType
    scope:      ScopeType
    concept:    Optional[ConceptType]
    person:     PersonType   = "unknown"
    number:     str          = "unknown"
    mood:       MoodType     = "indicative"
    polarity:   PolarityType = "positive"
    modal:      ModalType    = None
    pronominal: bool         = False
    impersonal: bool         = False
    compound:   bool         = False


@dataclass
class RegisterAnalysis:
    language:   str       = "fr"
    style:      str       = "neutre"
    confidence: float     = 0.5
    markers:    List[str] = field(default_factory=list)
    tu_vous:    str       = "indéterminé"
    negation:   str       = "absente"


@dataclass
class TokenNode:
    """Nœud de l'arbre syntaxique (v6)."""
    text:        str
    lemma:       str
    pos:         str
    dep:         str
    token_index: int
    role:        Optional[RoleType] = None   # user | ase | other
    children:    List["TokenNode"]  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text":self.text,"lemma":self.lemma,"pos":self.pos,
                "dep":self.dep,"role":self.role,
                "children":[c.to_dict() for c in self.children]}


@dataclass
class SyntacticTree:
    """Arbre syntaxique complet (v6)."""
    phrase:      str
    root:        Optional[TokenNode] = None
    subject:     Optional[str]       = None
    subject_role: Optional[RoleType] = None
    object_:     Optional[str]       = None

    def to_dict(self) -> dict:
        return {"phrase":self.phrase,
                "root":self.root.to_dict() if self.root else None,
                "subject":self.subject,"subject_role":self.subject_role,
                "object":self.object_}


@dataclass
class Intent:
    text:           str
    intent:         IntentType
    confidence:     float
    uncertain:      bool
    scores:         dict
    verb:           Optional[VerbAnalysis]
    who:            Optional[PersonType]
    who_raw:        Optional[str]
    with_who:       List[str]
    when:           Optional[TemporalSpan]
    where:          Optional[str]
    what:           Optional[ConceptType]
    action:         Optional[str]
    target:         Optional[str]
    actions:        List[dict]
    entities:       List[dict]
    memory_hint:    Optional[dict]
    register:       Optional[dict]
    assembled_from: List[str]
    syntax_tree:    Optional[SyntacticTree] = None
    corrections:    List[Tuple[str,str]]    = field(default_factory=list)
    clause_index:   int                     = 0
    processing_ms:  float                   = 0.0
    ts:             float                   = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_cognitive_frame(self) -> dict:
        """Export pour CognitionCore (v6)."""
        return {
            "type":         "intent",
            "intent_type":  self.intent,
            "verb":         self.verb.lemma if self.verb else None,
            "concept":      self.verb.concept if self.verb else None,
            "scope":        self.verb.scope if self.verb else "UNKNOWN",
            "modal":        self.verb.modal if self.verb else None,
            "polarity":     self.verb.polarity if self.verb else "positive",
            "who":          self.who,
            "who_raw":      self.who_raw,
            "with_who":     self.with_who,
            "time_start":   self.when.iso_start if self.when else None,
            "time_end":     self.when.iso_end   if self.when else None,
            "time_event":   self.when.named_event if self.when else None,
            "location":     self.where,
            "action":       self.action,
            "target":       self.target,
            "confidence":   self.confidence,
            "processing_ms":self.processing_ms,
        }


# ============================================================
# SECTION 7 — PipelineMetrics (v6)
# ============================================================

class PipelineMetrics:
    def __init__(self):
        self.total        = 0
        self.total_ms     = 0.0
        self.errors       = 0
        self.intent_dist: Dict[str,int]   = defaultdict(int)
        self.corrections:  Dict[str,int]  = defaultdict(int)
        self.start_time   = time.time()

    def record(self, intent: Intent):
        self.total     += 1
        self.total_ms  += intent.processing_ms
        self.intent_dist[intent.intent] += 1
        for orig, _ in intent.corrections:
            self.corrections[orig] += 1

    def record_error(self):
        self.errors += 1

    def summary(self) -> dict:
        avg = self.total_ms / self.total if self.total else 0
        return {
            "total_requests":    self.total,
            "total_errors":      self.errors,
            "avg_ms":            round(avg,1),
            "intent_distribution": dict(self.intent_dist),
            "top_corrections":   sorted(self.corrections.items(),
                                        key=lambda x:x[1], reverse=True)[:10],
            "uptime_s":          round(time.time()-self.start_time,1),
        }


# ============================================================
# SECTION 8 — HybridIntentClassifier (v6)
# ============================================================

class HybridIntentClassifier:
    """
    ML-first (CamemBERT finetuné) + fallback règles.
    Fonctionne sans modèle (règles seules) pour le démarrage à froid.
    """

    RULES_ACTION   = {"allume","éteins","ouvre","ferme","active","désactive",
                      "monte","baisse","règle","mets","configure","programme"}
    RULES_QUESTION = {"?","quel","quelle","quoi","comment","pourquoi",
                      "quand","où","qui","combien","lequel","laquelle"}
    RULES_EMOTION  = {"aime","adore","déteste","triste","heureux","fatigué",
                      "content","énervé","peur","angoisse","ravi","déprimé"}
    RULES_SOCIAL   = {"bonjour","salut","coucou","au revoir","merci",
                      "bravo","félicitations","bonne nuit","bonsoir"}

    def __init__(self, model_dir: Path, id2label: Optional[Dict[int,str]] = None):
        self._model_dir = model_dir
        self._id2label  = id2label
        self._tokenizer = None
        self._model     = None
        self._loaded    = False

    def load(self):
        if self._loaded or not TRANSFORMERS_AVAILABLE:
            self._loaded = True
            return
        if not self._model_dir.exists():
            logger.warning("Modèle absent — mode règles uniquement.")
            self._loaded = True
            return
        try:
            self._tokenizer = CamembertTokenizer.from_pretrained(str(self._model_dir))
            self._model     = CamembertForSequenceClassification.from_pretrained(
                str(self._model_dir))
            self._model.eval()
            lm = self._model_dir / "label_map.json"
            if lm.exists() and self._id2label is None:
                with open(lm) as f:
                    d = json.load(f)
                self._id2label = {int(k):v for k,v in d["id2label"].items()}
            self._loaded = True
            logger.info("CamemBERT chargé (hybride).")
        except Exception as e:
            logger.warning(f"Chargement échoué ({e}) — mode règles.")
            self._loaded = True

    def classify(self, text: str) -> dict:
        tl = text.lower()
        # ML d'abord si disponible
        if self._model and self._tokenizer and self._id2label:
            r = self._ml(text)
            if r["confidence"] >= 0.70:
                return r
        return self._rules(tl)

    def _ml(self, text: str) -> dict:
        try:
            inputs = self._tokenizer(
                text, truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(
                    self._model(**inputs).logits, dim=1)[0].numpy()
            pid  = int(probs.argmax())
            conf = float(probs[pid])
            return {
                "intent":    self._id2label[pid],
                "confidence":round(conf,4),
                "uncertain": conf < CONFIDENCE_THRESHOLD,
                "scores":    {self._id2label[i]:round(float(p),4)
                              for i,p in enumerate(probs)},
            }
        except Exception as e:
            logger.debug(f"ML classify error: {e}")
            return {"intent":"information_input","confidence":0.5,
                    "uncertain":True,"scores":{}}

    @classmethod
    def _rules(cls, text: str) -> dict:
        words = set(text.split())
        if words & cls.RULES_ACTION:
            return {"intent":"action_device","confidence":0.85,
                    "uncertain":False,"scores":{}}
        if "?" in text or words & cls.RULES_QUESTION:
            # Distinguer query_narrative vs query_state vs query_intention
            if any(w in text for w in ("fais","fait","allé","étais","as fait","as été")):
                intent = "query_narrative"
            elif any(w in text for w in ("vas","feras","iras","sera","irez")):
                intent = "query_intention"
            else:
                intent = "query_state"
            return {"intent":intent,"confidence":0.88,"uncertain":False,"scores":{}}
        if words & cls.RULES_SOCIAL:
            return {"intent":"chit_chat","confidence":0.95,
                    "uncertain":False,"scores":{}}
        if words & cls.RULES_EMOTION:
            return {"intent":"information_input","confidence":0.75,
                    "uncertain":False,"scores":{}}
        return {"intent":"information_input","confidence":0.65,
                "uncertain":True,"scores":{}}


# ============================================================
# SECTION 9 — SyntacticTreeExtractor (v6)
# ============================================================

class SyntacticTreeExtractor:
    """Extrait l'arbre syntaxique complet avec rôles user/ase/other."""

    _USER = {"je","j'","moi","me","m'"}
    _ASE  = {"tu","toi","te","t'"}

    @classmethod
    def _role(cls, token) -> Optional[RoleType]:
        tl = token.text.lower()
        if tl in cls._USER: return "user"
        if tl in cls._ASE:  return "ase"
        if token.ent_type_ == "PER": return "other"
        return None

    @classmethod
    def extract(cls, doc) -> SyntacticTree:
        tree  = SyntacticTree(phrase=doc.text)
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
            if token.dep_ in ("nsubj","nsubj:pass"):
                tree.subject      = token.text
                tree.subject_role = node.role
            if token.dep_ == "obj" and tree.object_ is None:
                tree.object_ = token.text

        for token in doc:
            if token.head != token and token.head.i in nodes:
                nodes[token.head.i].children.append(nodes[token.i])

        return tree


# ============================================================
# SECTION 10 — Extracteurs spaCy (1 passe, v3)
# ============================================================

def _extract_all_slots(
    doc,
) -> Tuple[Optional[PersonType], Optional[str], List[str], Optional[str], List[dict]]:
    who_person: Optional[PersonType] = None
    who_raw:    Optional[str]        = None
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
        unit:  Optional[str]   = None
        if etype in ("number","ordinal","quantity","percent","money",
                     "time_ref","date_ref","duration"):
            nm = re.search(r"(\d+(?:[.,]\d+)?)",
                           raw.replace("\u202f","").replace(" ",""))
            if nm:
                try: value = float(nm.group(1).replace(",","."))
                except: pass
            for pat, norm in UNIT_PATTERNS.items():
                if pat in raw.lower():
                    unit = norm; break
        entities.append({"raw":raw,"type":etype,"value":value,"unit":unit})
        if ent.label_ == "PER":
            with_who_set.add(ent.text)

    for token in doc:
        if not who_person and token.dep_ in ("iobj","obj:iobj"):
            raw_t = token.text.lower().rstrip("'")
            p = CLITIC_TO_PERSON.get(raw_t) or CLITIC_TO_PERSON.get(token.text.lower())
            if p:
                who_person, who_raw = p, token.text

        elif not who_person and token.dep_ == "obj":
            raw_t = token.text.lower().rstrip("'")
            p = CLITIC_TO_PERSON.get(raw_t) or CLITIC_TO_PERSON.get(token.text.lower())
            if p:
                who_person, who_raw = p, token.text

        elif token.dep_ in ("nsubj","nsubj:pass"):
            raw_t = token.text.lower().rstrip("'")
            p = CLITIC_TO_PERSON.get(raw_t)
            if p:
                if not who_person: who_person, who_raw = p, token.text
            elif token.text.lower() in ("je","j'","j","moi"):
                if not who_person: who_person, who_raw = "1st_sg", token.text
            elif token.text.lower() in ("tu","t'","te","toi"):
                if not who_person: who_person, who_raw = "2nd_sg", token.text
            elif token.text.lower() in ("nous","on"):
                if not who_person: who_person, who_raw = "1st_pl", token.text
            elif token.text.lower() == "vous":
                if not who_person: who_person, who_raw = "2nd_pl", token.text
            elif token.ent_type_ == "PER" and not who_person:
                who_person, who_raw = token.text, token.text
            subjects.add(token.text)

        if not where and token.text.lower() in ("dans","au","en","à","chez") \
                and token.dep_ == "case":
            head = token.head
            if head.pos_ in ("NOUN","PROPN"):
                where = head.text.lower()

        if token.ent_type_ == "PER" and token.text not in subjects:
            with_who_set.add(token.text)

    if not where:
        for ent in doc.ents:
            if ent.label_ in ("LOC","GPE"):
                where = ent.text.lower(); break

    return who_person, who_raw, list(with_who_set), where, entities


def _extract_verb(doc) -> Optional[VerbAnalysis]:
    for token in doc:
        if token.pos_ != "VERB":
            continue
        if token.dep_ not in ("ROOT","acl","relcl","advcl","xcomp","ccomp"):
            continue
        morph = token.morph
        mood_map = {"Cnd":"conditional","Imp":"imperative",
                    "Sub":"subjunctive","Ind":"indicative"}
        mood_raw = morph.get("Mood")
        mood: MoodType = "indicative"
        if mood_raw:
            mood = mood_map.get(mood_raw[0],"indicative")
        elif morph.get("VerbForm") and "Inf" in morph.get("VerbForm"):
            mood = "infinitive"

        tense_raw = morph.get("Tense")
        if mood == "conditional":
            raw_tense: TenseType = "conditional"
        elif tense_raw:
            if "Past" in tense_raw or "Imp" in tense_raw:
                raw_tense = "past"
            elif "Pres" in tense_raw:
                vf = morph.get("VerbForm")
                is_pp = vf and "Part" in vf
                aux_pres = any(
                    c.pos_=="AUX" and c.morph.get("Tense") and "Pres" in c.morph.get("Tense")
                    for c in token.children
                )
                raw_tense = "past" if (is_pp and aux_pres) else "present"
            elif "Fut" in tense_raw:
                raw_tense = "future"
            else:
                raw_tense = "unknown"
        else:
            raw_tense = "unknown"

        raw_tense = FrenchVerbAnalyzer.refine_tense(token.text, token.lemma_, raw_tense)

        person_raw = morph.get("Person")
        number_raw = morph.get("Number")
        person: PersonType = "unknown"
        number = "unknown"
        if person_raw and number_raw:
            pm = {("1","Sing"):"1st_sg",("2","Sing"):"2nd_sg",("3","Sing"):"3rd_sg",
                  ("1","Plur"):"1st_pl",("2","Plur"):"2nd_pl",("3","Plur"):"3rd_pl"}
            person = pm.get((person_raw[0],number_raw[0]),"unknown")
            number = "sg" if number_raw[0]=="Sing" else "pl" if number_raw[0]=="Plur" else "unknown"

        neg = [c for c in token.children
               if c.dep_=="advmod" and c.lemma_ in ("ne","pas","plus","jamais","rien","guère")]
        left_neg = any(t.dep_=="advmod" and t.lemma_ in ("ne","n") for t in token.lefts)
        polarity: PolarityType = "negative" if (neg or left_neg) else "positive"

        modal: ModalType = None
        for aux in token.children:
            if aux.pos_ == "AUX":
                modal = MODAL_MARKERS.get(aux.lemma_.lower())
                if modal: break
        if not modal and mood == "conditional":
            modal = "wish"

        concept = SEMANTIC_MAP.get(token.lemma_.lower())
        if not concept:
            for child in token.children:
                if child.pos_ == "VERB":
                    concept = SEMANTIC_MAP.get(child.lemma_.lower())
                    if concept: break

        return VerbAnalysis(
            lemma=token.lemma_, tense=raw_tense,
            scope=TENSE_TO_SCOPE.get(raw_tense,"UNKNOWN"),
            concept=concept, person=person, number=number, mood=mood,
            polarity=polarity, modal=modal,
            pronominal=FrenchVerbAnalyzer.is_pronominal(doc, token),
            impersonal=FrenchVerbAnalyzer.is_impersonal(token.lemma_),
            compound=FrenchVerbAnalyzer.is_compound(token.text) is not None,
        )
    return None


def _extract_actions(doc) -> List[dict]:
    actions = []
    for token in doc:
        if token.pos_ != "VERB" or token.dep_ in ("aux","aux:pass","cop"):
            continue
        subs = [c.text for c in token.children if c.dep_ in ("nsubj","nsubj:pass","expl:subj")]
        objs = [c.text for c in token.children if c.dep_ in ("obj","iobj","obl","xcomp")]
        if not subs:
            for t in doc:
                if t.text.lower() in ("je","j'","tu","il","elle","nous","vous","ils","elles","on"):
                    subs.append(t.text); break
        actions.append({"verb":token.lemma_.lower(),"subjects":subs,
                        "objects":objs,"concept":SEMANTIC_MAP.get(token.lemma_.lower())})
    return actions


def _extract_what(doc) -> Optional[ConceptType]:
    for token in doc:
        if token.pos_=="VERB" and token.dep_ in ("ROOT","acl","relcl","advcl","xcomp","ccomp"):
            c = SEMANTIC_MAP.get(token.lemma_.lower())
            if c: return c
            for child in token.children:
                if child.pos_=="VERB":
                    c = SEMANTIC_MAP.get(child.lemma_.lower())
                    if c: return c
    return None


def _extract_device(doc) -> Tuple[Optional[str], Optional[str]]:
    action = target = None
    for token in doc:
        if token.pos_=="VERB" and not action:
            action = ACTION_VERBS.get(token.lemma_.lower())
        if token.pos_ in ("NOUN","PROPN") and not target:
            target = DEVICE_NOUNS.get(token.lemma_.lower())
    if not target:
        tl = doc.text.lower()
        for noun, norm in DEVICE_NOUNS.items():
            if noun in tl: target = norm; break
    return action, target


def _compute_salience(doc, entities: List[dict], verb: Optional[VerbAnalysis]=None) -> float:
    score = 0.0
    for e in entities:
        score += (0.20 if e["type"] in ("person","organization","event","product") else
                  0.12 if e["type"] in ("location","facility") else
                  0.10 if e["type"] in ("quantity","money","percent","duration") else 0.05)
    if verb and verb.concept in ("COMMUNICATE","PERCEIVE","MOVE","TRANSFER","SOCIAL_ACT"):
        score += 0.15
    elif verb and verb.concept in ("INGEST","CREATE","HEALTH"):
        score += 0.10
    content = [t for t in doc if t.pos_ in ("NOUN","PROPN","VERB","ADJ","NUM")]
    score += min(0.20, len(content)*0.04)
    return round(min(1.0, score), 3)


def _infer_info_subject(doc) -> str:
    for token in doc:
        if token.dep_ in ("nsubj","nsubj:pass"):
            t = token.text.lower().rstrip("'")
            if t in ("je","j","moi","nous","on"): return "human"
            if t in ("tu","toi","te","t"):        return "self"
            if token.ent_type_ == "PER":          return "other"
            return "world"
    return "other"


def _is_information_input(doc, verb) -> bool:
    if doc.text.strip().endswith("?"): return False
    starters = {"est-ce","qu'","que","quoi","comment","quand","ou","qui","quel","quelle"}
    first = {t.text.lower().rstrip("'") for t in list(doc)[:3]}
    if first & starters: return False
    for token in doc:
        if token.dep_ in ("nsubj","nsubj:pass"):
            if token.text.lower().rstrip("'") in INFO_SUBJECTS or token.ent_type_=="PER":
                return True
    return False


def _is_fragment(doc, slots: dict) -> bool:
    has_verb = any(
        t.pos_=="VERB" and t.dep_ in ("ROOT","acl","relcl","advcl","xcomp","ccomp")
        for t in doc
    )
    return not has_verb and any([
        slots.get("who"), slots.get("when"), slots.get("where"), slots.get("with_who"),
    ])


def _analyze_register(doc, text: str) -> RegisterAnalysis:
    toks  = [t.text.lower().rstrip("'") for t in doc]
    votes: Dict[str,float] = {s:0.0 for s in REGISTER_LEXICON}
    marks: Dict[str,list]  = {s:[]   for s in REGISTER_LEXICON}
    for tok in toks:
        for style, lex in REGISTER_LEXICON.items():
            if tok in lex:
                votes[style] += 1; marks[style].append(tok)
    has_pas = any(w in toks for w in ("pas","plus","jamais"))
    has_ne  = any(t in toks for t in NEGATION_COMPLETE)
    if has_pas and not has_ne:
        votes["familier"] += 1; marks["familier"].append("neg-inc")
    tu_vous = "indéterminé"
    if any(t in VOUVOIEMENT_MARKERS for t in toks) and not any(t in TUTOIEMENT_MARKERS for t in toks):
        tu_vous = "vous"; votes["soutenu"] += 0.5
    elif any(t in TUTOIEMENT_MARKERS for t in toks):
        tu_vous = "tu"
    best = max(votes, key=votes.get)
    conf = round(min(0.95, sum(votes.values())/max(len(toks),1)), 3)
    if votes[best] < 0.8: best = "neutre"
    all_markers = marks.get(best,[])
    if tu_vous != "indéterminé": all_markers.append(f"tutv:{tu_vous}")
    neg = ("complete" if has_pas and has_ne else "incomplete" if has_pas else "absente")
    return RegisterAnalysis(style=best, confidence=conf,
                            markers=all_markers[:8], tu_vous=tu_vous, negation=neg)


# ============================================================
# SECTION 11 — split_clauses (v3)
# ============================================================

def split_clauses(text: str) -> List[str]:
    parts = _CLAUSE_COORD.split(text)
    if len(parts) == 1:
        return [text.strip()]
    valid: List[str] = []
    for part in parts:
        part = part.strip()
        if not part: continue
        if _VERB_PAT.search(part):
            valid.append(part)
        else:
            if valid: valid[-1] += " " + part
            else:     valid.append(part)
    return valid if len(valid) >= 2 else [text.strip()]


# ============================================================
# SECTION 12 — Extraction temporelle (v4 : FrenchTemporalResolver)
# ============================================================

def _extract_temporal_v4(
    doc,
    tense: str = "unknown",
    resolver: Optional[FrenchTemporalResolver] = None,
) -> Optional[TemporalSpan]:
    """
    Extraction temporelle v4.
    Utilise FrenchTemporalResolver en priorité (cache LRU, couverture étendue).
    Fallback : TIMEX3 via timexy + spaCy NER + dateparser search_dates.
    """
    resolver = resolver or _get_temporal_resolver()
    prefer   = "future" if tense in ("future","conditional") else "past"
    dp       = _get_dateparser() if DATEPARSER_AVAILABLE else None

    # 1. Intervalle "entre X et Y" (géométrie temporelle)
    interval = _extract_interval_v4(doc.text, prefer)
    if interval:
        return interval

    # 2. FrenchTemporalResolver (moments, saisons, fériés, vacances, semaine, relatifs, durées)
    ref = datetime.datetime.now()
    resolved = resolver.resolve(doc.text, ref)
    if resolved and resolved.get("source","none") not in ("none",""):
        span = resolver.to_temporal_span(resolved, tense)
        if span:
            return span

    # 3. timexy
    if _check_timexy() and dp:
        TIMEX3 = {"DATE":"DATE","TIME":"TIME","DURATION":"DURATION","SET":"SET"}
        cfg = {"RETURN_AS_TIMEZONE_AWARE":True,"TIMEZONE":"Europe/Paris","PREFER_DATES_FROM":prefer}
        for ent in doc.ents:
            ttype = getattr(getattr(ent,"_",None),"timex_type",None)
            if ttype:
                tv = getattr(getattr(ent,"_",None),"timex_value",None)
                dt = None
                if tv:
                    try: dt = datetime.datetime.fromisoformat(tv)
                    except: pass
                if not dt:
                    dt = dp.parse(ent.text, languages=[LANGUAGE], settings=cfg)
                if dt:
                    iso = dt.isoformat()
                    return TemporalSpan(raw=ent.text, iso_start=iso, iso_end=iso,
                                        timex_type=TIMEX3.get(ttype,"DATE"),
                                        source="timexy")

    # 4. spaCy NER DATE/TIME
    if dp:
        cfg = {"RETURN_AS_TIMEZONE_AWARE":True,"TIMEZONE":"Europe/Paris","PREFER_DATES_FROM":prefer}
        for ent in doc.ents:
            if ent.label_ in ("DATE","TIME"):
                dt = dp.parse(ent.text, languages=[LANGUAGE], settings=cfg)
                if dt:
                    iso = dt.isoformat()
                    return TemporalSpan(raw=ent.text, iso_start=iso, iso_end=iso,
                                        timex_type="TIME" if ent.label_=="TIME" else "DATE",
                                        source="spacy_ner")

        # 5. search_dates fallback
        try:
            from dateparser.search import search_dates
            found = search_dates(doc.text, languages=[LANGUAGE], settings=cfg)
            if found:
                raw_expr, dt = found[0]
                if dt:
                    iso = dt.isoformat()
                    return TemporalSpan(raw=raw_expr, iso_start=iso, iso_end=iso,
                                        timex_type="DATE", source="dateparser")
        except Exception:
            pass

    return None


def _extract_interval_v4(text: str, prefer: str = "past") -> Optional[TemporalSpan]:
    m = re.search(
        r"entre\s+(\d{1,2}h\d{0,2}|\d{1,2}:\d{2}|[\w\s]+?)"
        r"\s+et\s+(\d{1,2}h\d{0,2}|\d{1,2}:\d{2}|[\w\s]+?)(?=\s|$|,|\.|;)",
        text, re.IGNORECASE,
    )
    if not m or not DATEPARSER_AVAILABLE:
        return None
    dp  = _get_dateparser()
    cfg = {"RETURN_AS_TIMEZONE_AWARE":True,"TIMEZONE":"Europe/Paris","PREFER_DATES_FROM":prefer}
    dt1 = dp.parse(m.group(1).strip(), languages=[LANGUAGE], settings=cfg)
    dt2 = dp.parse(m.group(2).strip(), languages=[LANGUAGE], settings=cfg)
    if dt1 and dt2:
        return TemporalSpan(
            raw=m.group(0), iso_start=dt1.isoformat(), iso_end=dt2.isoformat(),
            timex_type="INTERVAL", source="regex_interval",
            interval_start_raw=m.group(1).strip(), interval_end_raw=m.group(2).strip(),
        )
    return None


def _parse_durations(text: str) -> List[dict]:
    results = []
    for m in re.finditer(
        r"pendant\s+(\d+(?:[.,]\d+)?)\s*"
        r"(heures?|h|minutes?|min|secondes?|s|jours?|semaines?|mois)",
        text, re.IGNORECASE
    ):
        val   = float(m.group(1).replace(",","."))
        u_raw = m.group(2).lower()
        unit  = ("h"    if u_raw.startswith("h")   else
                 "min"  if u_raw.startswith("min")  else
                 "s"    if u_raw.startswith("s")    else
                 "day"  if u_raw.startswith("j")    else
                 "week" if u_raw.startswith("sem")  else "month")
        results.append({"type":"duration","value":val,"unit":unit,"raw":m.group(0)})
    if DATEPARSER_AVAILABLE:
        dp = _get_dateparser()
        for m in re.finditer(
            r"jusqu['\u2019]?\s*[aà]\s+(\d{1,2}h\d{0,2}|\d{1,2}:\d{2})",
            text, re.IGNORECASE
        ):
            dt = dp.parse(m.group(1), languages=[LANGUAGE],
                          settings={"RETURN_AS_TIMEZONE_AWARE":True,"TIMEZONE":"Europe/Paris"})
            if dt:
                results.append({"type":"until","iso":dt.isoformat(),"raw":m.group(0)})
    return results


def _merge_time_duration(span: TemporalSpan, durations: List[dict]) -> TemporalSpan:
    if not durations or span.timex_type == "INTERVAL":
        return span
    try:
        start_dt = datetime.datetime.fromisoformat(span.iso_start)
    except (ValueError, TypeError):
        return span
    end_dt = start_dt
    dur_raw = dur_unit = until_raw = None
    dur_val: Optional[float] = None
    for d in durations:
        if d["type"] == "until":
            try:
                end_dt    = datetime.datetime.fromisoformat(d["iso"])
                until_raw = d["raw"]
            except (ValueError, KeyError):
                pass
        elif d["type"] == "duration":
            v, u = d.get("value",0), d.get("unit","h")
            delta = (datetime.timedelta(hours=v)   if u=="h"    else
                     datetime.timedelta(minutes=v) if u=="min"  else
                     datetime.timedelta(seconds=v) if u=="s"    else
                     datetime.timedelta(days=v)    if u=="day"  else
                     datetime.timedelta(weeks=v)   if u=="week" else
                     datetime.timedelta(days=v*30))
            end_dt   = start_dt + delta
            dur_raw  = d["raw"]
            dur_unit = u
            dur_val  = v
    from dataclasses import replace
    return replace(span, iso_end=end_dt.isoformat(),
                   duration_raw=dur_raw, duration_unit=dur_unit,
                   duration_value=dur_val, until_raw=until_raw)


# ============================================================
# SECTION 13 — Entraînement CamemBERT (v3)
# ============================================================

def download_model():
    from transformers import CamembertTokenizer, CamembertForSequenceClassification
    logger.info(f"Téléchargement de '{MODEL_NAME}'…")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CamembertTokenizer.from_pretrained(MODEL_NAME).save_pretrained(MODEL_DIR/"tokenizer")
    CamembertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(INTENTS), ignore_mismatched_sizes=True
    ).save_pretrained(MODEL_DIR/"base")
    logger.info(f"Sauvegardé → {MODEL_DIR}")


def train(data_file: Path = DATA_FILE):
    import torch
    from torch.utils.data import DataLoader, random_split
    from torch.optim import AdamW
    from transformers import (CamembertTokenizer, CamembertForSequenceClassification,
                              get_linear_schedule_with_warmup)
    from sklearn.metrics import classification_report

    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    label2id = {intent:i for i,intent in enumerate(INTENTS)}
    id2label = {i:intent for intent,i in label2id.items()}
    texts  = [d["text"]   for d in data]
    labels = [label2id[d["intent"]] for d in data]

    tok_path = MODEL_DIR/"tokenizer"
    if not tok_path.exists():
        raise FileNotFoundError("Tokenizer absent — lance --download d'abord.")

    tokenizer = CamembertTokenizer.from_pretrained(str(tok_path))

    class _DS:
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, padding=True,
                                 max_length=MAX_LENGTH, return_tensors="pt")
            self.labels = labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, i):
            item = {k:v[i] for k,v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    ds = _DS(texts, labels)
    val_n = int(len(ds)*TEST_SPLIT)
    train_ds, val_ds = random_split(ds,[len(ds)-val_n,val_n],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    device = torch.device("cpu")
    model  = CamembertForSequenceClassification.from_pretrained(
        str(MODEL_DIR/"base"), num_labels=len(INTENTS),
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader)*EPOCHS*WARMUP_RATIO),
        num_training_steps=len(train_loader)*EPOCHS)

    best_acc = 0.0; no_improve = 0
    fdir = MODEL_DIR/"finetuned"

    for epoch in range(1, EPOCHS+1):
        model.train(); total_loss = 0.0; t0 = time.time()
        for batch in train_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            loss  = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
        model.eval(); preds, true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k:v.to(device) for k,v in batch.items()}
                preds.extend(torch.argmax(model(**batch).logits,dim=1).cpu().numpy())
                true.extend(batch["labels"].cpu().numpy())
        acc = np.mean(np.array(preds)==np.array(true))
        logger.info(f"Epoch {epoch}/{EPOCHS} | loss={total_loss/len(train_loader):.4f} "
                    f"| val_acc={acc:.3f} | {time.time()-t0:.1f}s")
        if acc > best_acc:
            best_acc = acc; no_improve = 0
            model.save_pretrained(str(fdir))
            tokenizer.save_pretrained(str(fdir))
            with open(fdir/"label_map.json","w") as f:
                json.dump({"id2label":id2label,"label2id":label2id},f)
            logger.info(f"  ✓ Meilleur modèle (val_acc={acc:.3f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping à l'époque {epoch}.")
                break

    model_best = CamembertForSequenceClassification.from_pretrained(str(fdir)).to(device)
    model_best.eval(); preds, true = [], []
    with torch.no_grad():
        for batch in DataLoader(val_ds, batch_size=BATCH_SIZE):
            batch = {k:v.to(device) for k,v in batch.items()}
            preds.extend(torch.argmax(model_best(**batch).logits,dim=1).cpu().numpy())
            true.extend(batch["labels"].cpu().numpy())
    print(classification_report(true, preds, target_names=INTENTS, digits=3))


# ============================================================
# SECTION 14 — IntentPipeline v4
# ============================================================

class IntentPipeline:
    """
    Texte → liste d'Intent(s).
    Cumule v3 (sémantique, multi-intent, EventGraph) +
    v6 (temporal riche, ConversationContext, métriques, batch, debug).
    """

    def __init__(
        self,
        q_in:      Queue,
        q_out:     Queue,
        model_dir: Path  = MODEL_DIR / "finetuned",
        debug:     bool  = False,
    ):
        self._q_in           = q_in
        self._q_out          = q_out
        self._model_dir      = model_dir
        self._debug          = debug
        self._nlp            = None
        self._classifier:    Optional[HybridIntentClassifier] = None
        self._resolver       = FrenchTemporalResolver()
        self._ctx            = ConversationContext()   # v6 : ellipses + pronoms
        self._frame          = ConversationFrame()     # v3 : slots inter-phrases
        self._metrics        = PipelineMetrics()
        self._executor       = ThreadPoolExecutor(max_workers=MAX_WORKERS,
                                                  thread_name_prefix="intent-batch")
        self._thread:        Optional[threading.Thread] = None
        self._running        = False
        self._loaded         = False
        self._lock           = threading.Lock()
        self._enable_corrector = True
        self._corrections_stats: Dict[str,int] = defaultdict(int)

    def enable_corrector(self, enable: bool = True):
        self._enable_corrector = enable

    # ── Chargement ────────────────────────────────────────────

    def load(self):
        self._load_spacy()
        self._load_classifier()
        self._loaded = True
        logger.info("IntentPipeline v4 prêt.")

    def _load_spacy(self):
        if not SPACY_AVAILABLE:
            raise ImportError("spacy requis : pip install spacy fr_core_news_sm")
        t0 = time.time()
        self._nlp = spacy.load(SPACY_MODEL)
        if _check_timexy():
            try:
                cfg = {"per":True,"dur":True,"set":True,"num":True,"lang":"fr"}
                if "timexy" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("timexy", config=cfg, last=True)
                logger.info(f"spaCy + timexy chargés en {time.time()-t0:.2f}s")
            except Exception:
                logger.info(f"spaCy chargé en {time.time()-t0:.2f}s (timexy erreur)")
        else:
            logger.info(f"spaCy chargé en {time.time()-t0:.2f}s")

    def _load_classifier(self):
        self._classifier = HybridIntentClassifier(self._model_dir)
        self._classifier.load()

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self):
        if not self._loaded:
            raise RuntimeError("Appelle load() avant start().")
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker, name="intent-pipeline-v4", daemon=True)
        self._thread.start()
        logger.info("IntentPipeline v4 démarré.")

    def stop(self):
        self._running = False
        self._q_in.put(None)
        self._executor.shutdown(wait=False)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("IntentPipeline v4 arrêté.")

    # ── Worker ────────────────────────────────────────────────

    def _worker(self):
        while self._running:
            try:
                text = self._q_in.get(timeout=1.0)
            except Empty:
                if self._frame.is_expired(): self._frame.reset()
                if self._ctx.is_expired():   self._ctx = ConversationContext()
                continue
            if text is None:
                break
            intents = self._process(text.strip())
            if intents:
                with self._lock:
                    self._q_out.put([i.to_dict() for i in intents])
                for i in intents:
                    self._metrics.record(i)
                    self._ctx.update_from_intent(i)

    # ── API batch (v6) ────────────────────────────────────────

    def batch_process(self, texts: List[str]) -> List[Intent]:
        """Traite plusieurs phrases en parallèle (ThreadPoolExecutor)."""
        futures = [self._executor.submit(self._process_single, t) for t in texts]
        results: List[Intent] = []
        for f in futures:
            try:
                r = f.result(timeout=10.0)
                results.extend(r)
            except Exception as e:
                error_logger.error(f"batch_process: {e}")
                self._metrics.record_error()
        return results

    def _process_single(self, text: str) -> List[Intent]:
        """Version thread-safe de _process pour le batch."""
        return self._process(text.strip())

    # ── Traitement principal ───────────────────────────────────

    def _preprocess(self, text: str) -> Tuple[str, List[Tuple[str,str]]]:
        text = re.sub(r'[^\w\s\u00C0-\u00FF?.!,;:\'"()-]', '', text)
        if self._enable_corrector:
            corrected, corrections = FrenchTextCorrector.correct(text)
            for orig, fix in corrections:
                self._corrections_stats[f"{orig}→{fix}"] += 1
            return corrected, corrections
        return text, []

    def _process(self, text: str) -> List[Intent]:
        if self._frame.is_expired(): self._frame.reset()
        if self._ctx.is_expired():   self._ctx = ConversationContext()

        # Résolution ellipses + pronoms (v6)
        text = self._ctx.resolve_ellipsis(text)
        text = self._ctx.resolve_pronouns(text)

        text_clean, corrections = self._preprocess(text)
        clauses = split_clauses(text_clean)

        # Fragment silencieux → accumulation frame
        if len(clauses) == 1:
            doc  = self._nlp(text_clean)
            verb = _extract_verb(doc)
            who_p, who_r, with_who, where, _ = _extract_all_slots(doc)
            temporal = _extract_temporal_v4(doc, verb.tense if verb else "unknown",
                                            self._resolver)
            slots = {"who":who_p,"who_raw":who_r,"with_who":with_who,
                     "where":where,"when":temporal,"what":_extract_what(doc)}
            if _is_fragment(doc, slots):
                self._frame.update(fragment=text_clean,
                                   **{k:v for k,v in slots.items() if v is not None})
                logger.debug(f"Fragment accumulé : '{text_clean}'")
                return []

        assembled = self._frame.flush_pending()
        results   = []
        for idx, clause in enumerate(clauses):
            intent = self._process_clause(
                clause, clause_index=idx,
                assembled_from=assembled if idx==0 else [],
                corrections=corrections  if idx==0 else [],
            )
            if intent:
                results.append(intent)
        return results

    def _process_clause(
        self,
        text:           str,
        clause_index:   int          = 0,
        assembled_from: List[str]    = None,
        corrections:    List[Tuple]  = None,
    ) -> Optional[Intent]:
        t0 = time.time()
        assembled_from = assembled_from or []
        corrections    = corrections    or []

        doc   = self._nlp(text)
        verb  = _extract_verb(doc)
        tense = verb.tense if verb else "unknown"

        if self._debug:
            print(f"\n  [{clause_index}] '{text}'")
            print(f"    verb={verb.lemma+'/'+verb.tense if verb else 'none'} "
                  f"mood={verb.mood if verb else '?'} "
                  f"modal={verb.modal if verb else '?'} "
                  f"polarity={verb.polarity if verb else '?'}")

        # Arbre syntaxique (v6)
        syntax_tree = SyntacticTreeExtractor.extract(doc)

        who_p, who_r, with_who, where, entities = _extract_all_slots(doc)
        temporal = _extract_temporal_v4(doc, tense, self._resolver)
        what     = _extract_what(doc)

        if self._debug and temporal:
            print(f"    when={temporal.raw} → {temporal.iso_start} "
                  f"[{temporal.timex_type}/{temporal.source}]"
                  + (f" event={temporal.named_event}" if temporal.named_event else ""))

        durations = _parse_durations(text) or self._frame.durations
        if temporal and durations:
            temporal = _merge_time_duration(temporal, durations)
        elif not temporal and self._frame.when and durations:
            temporal = _merge_time_duration(self._frame.when, durations)

        actions      = _extract_actions(doc)
        cur_subjects = [s for a in actions for s in a.get("subjects",[])]
        if not cur_subjects and self._frame.subjects:
            for a in actions:
                if not a.get("subjects"):
                    a["subjects"] = self._frame.subjects

        merged = {
            "who":     who_p      or self._frame.who,
            "who_raw": who_r      or self._frame.who_raw,
            "with_who":with_who   or self._frame.with_who,
            "when":    temporal   or self._frame.when,
            "where":   where      or self._frame.where,
            "what":    what       or self._frame.what,
        }

        if clause_index == 0:
            self._frame.update(who=who_p, who_raw=who_r, with_who=with_who,
                               when=temporal, where=where,
                               subjects=cur_subjects or None,
                               durations=durations or None)

        reg = _analyze_register(doc, text)
        if reg.style != "neutre":
            self._frame.update(register=reg.style)

        clf    = self._classifier.classify(text)
        action = target = None
        if clf["intent"] == "action_device":
            action, target = _extract_device(doc)

        intent_type = clf["intent"]
        if intent_type not in ("action_device",) and \
           clf["confidence"] < 0.80 and _is_information_input(doc, verb):
            intent_type = "information_input"

        memory_hint = None
        if intent_type == "information_input":
            memory_hint = {
                "subject":  _infer_info_subject(doc),
                "salience": _compute_salience(doc, entities, verb),
                "entities": entities,
                "raw_info": text,
            }

        ms = (time.time() - t0) * 1000

        if self._debug:
            print(f"    intent={intent_type} conf={clf['confidence']:.2f} "
                  f"who={merged['who']}({merged['who_raw']}) "
                  f"where={merged['where']} {ms:.0f}ms")

        logger.info(
            f"  [{clause_index}] {intent_type} ({clf['confidence']:.2f}) "
            f"verb={verb.lemma+'/'+verb.tense if verb else 'none'} "
            f"who={merged['who']}({merged['who_raw']}) "
            f"where={merged['where']} "
            f"when={merged['when'].raw if merged['when'] else None} "
            f"{ms:.0f}ms"
            + (f" ← {assembled_from}" if assembled_from else "")
        )

        return Intent(
            text=text, intent=intent_type, confidence=clf["confidence"],
            uncertain=clf["uncertain"], scores=clf["scores"],
            verb=verb, who=merged["who"], who_raw=merged["who_raw"],
            with_who=merged["with_who"], when=merged["when"],
            where=merged["where"], what=merged["what"],
            action=action, target=target, actions=actions,
            entities=entities, memory_hint=memory_hint,
            register=asdict(reg), assembled_from=assembled_from,
            syntax_tree=syntax_tree, corrections=corrections,
            clause_index=clause_index, processing_ms=ms, ts=time.time(),
        )

    def get_metrics(self) -> dict:
        return {
            "pipeline":        self._metrics.summary(),
            "temporal_cache":  self._resolver.get_stats(),
            "corrections_top": dict(sorted(
                self._corrections_stats.items(),
                key=lambda x:x[1], reverse=True)[:10]),
        }

    def benchmark(self, n: int = 20) -> dict:
        samples = ["comment tu vas ?","t'étais où hier soir ?",
                   "allume la lumière","je jardinerai demain matin pendant 2h",
                   "pendant les vacances d'été on ira à la mer"]
        times = []
        for i in range(n):
            t0 = time.time()
            self._process(samples[i % len(samples)])
            times.append((time.time()-t0)*1000)
        return {"n":n,"avg_ms":round(np.mean(times),1),
                "min_ms":round(np.min(times),1),
                "max_ms":round(np.max(times),1),
                "p95_ms":round(np.percentile(times,95),1)}


# ============================================================
# SECTION 15 — EventGraph + EventGraphConsumer (v3)
# ============================================================

@dataclass
class EpisodicEntity:
    name:  str
    type:  str
    count: int = 1


@dataclass
class EpisodicEvent:
    id:          int
    action:      str
    agent:       List[str]
    objects:     List[str]
    location:    Optional[str]
    concept:     Optional[str]
    start_time:  Optional[datetime.datetime]
    end_time:    Optional[datetime.datetime]
    scope:       str
    salience:    float
    trigger_id:  Optional[int]
    intent_type: str
    raw_text:    str
    ts:          float


@dataclass
class GraphContext:
    subjects:  List[str]                    = field(default_factory=list)
    location:  Optional[str]               = None
    time_base: Optional[datetime.datetime] = None
    time_end:  Optional[datetime.datetime] = None


class EventGraph:
    def __init__(self):
        self._lock     = threading.RLock()
        self.entities: Dict[str, EpisodicEntity] = {}
        self.events:   List[EpisodicEvent]       = []
        self.context   = GraphContext()
        self._counter  = 0

    def set_location(self, loc: Optional[str]):
        with self._lock:
            if loc: self.context.location = loc.lower()

    def set_time(self, iso_start: Optional[str], iso_end: Optional[str]=None):
        with self._lock:
            if iso_start:
                try: self.context.time_base = datetime.datetime.fromisoformat(iso_start)
                except: pass
            if iso_end:
                try: self.context.time_end  = datetime.datetime.fromisoformat(iso_end)
                except: pass

    def set_subjects(self, subjects: List[str]):
        with self._lock:
            if subjects: self.context.subjects = subjects

    def upsert_entity(self, name: str, etype: str):
        key = name.lower()
        with self._lock:
            if key in self.entities: self.entities[key].count += 1
            else: self.entities[key] = EpisodicEntity(name=name,type=etype,count=1)

    def add_event(self, action: str, agent: List[str]=None, objects: List[str]=None,
                  concept: str=None, scope: str="UNKNOWN", salience: float=0.0,
                  trigger_id: int=None, intent_type: str="unknown",
                  raw_text: str="") -> EpisodicEvent:
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

    def last_events(self, n: int=5) -> List[EpisodicEvent]:
        with self._lock: return list(self.events[-n:])

    def top_entities(self, n: int=5) -> List[EpisodicEntity]:
        with self._lock:
            return sorted(self.entities.values(), key=lambda e:e.count, reverse=True)[:n]

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_events":   len(self.events),
                "total_entities": len(self.entities),
                "by_scope": {s:sum(1 for e in self.events if e.scope==s)
                             for s in ("PAST","PRESENT","FUTURE","HYPOTHETICAL","UNKNOWN")},
                "top_entities": [{"name":e.name,"type":e.type,"count":e.count}
                                 for e in self.top_entities()],
            }


class EventGraphConsumer(threading.Thread):
    """Consomme les listes d'Intent dicts. Compatible dict (legacy) et liste."""

    EVENT_INTENTS = {"information_input","action_device","chit_chat"}

    def __init__(self, q_in: Queue, graph: EventGraph):
        super().__init__(name="eventgraph-consumer", daemon=True)
        self._q_in    = q_in
        self._graph   = graph
        self._running = True

    def stop(self):
        self._running = False
        self._q_in.put(None)

    def run(self):
        logger.info("EventGraphConsumer démarré.")
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
        logger.info("EventGraphConsumer arrêté.")

    def _consume(self, intent: dict):
        self._graph.set_location(intent.get("where"))
        when = intent.get("when")
        if when:
            self._graph.set_time(when.get("iso_start"), when.get("iso_end"))
        actions = intent.get("actions") or []
        agents  = list({s for a in actions for s in (a.get("subjects") or [])
                       if s.lower() not in ("je","j'","j","on")})
        if any(s.lower().rstrip("'") in ("je","j","moi","on","nous")
               for a in actions for s in (a.get("subjects") or [])):
            agents = ["human"] + agents
        if agents:
            self._graph.set_subjects(agents)
        for ent in (intent.get("entities") or []):
            if ent.get("type") in ("person","location","organization","event","product"):
                self._graph.upsert_entity(ent["raw"], ent["type"])
        for p in (intent.get("with_who") or []):
            self._graph.upsert_entity(p, "person")
        if intent.get("intent") not in self.EVENT_INTENTS:
            return
        verb    = intent.get("verb") or {}
        scope   = verb.get("scope","UNKNOWN")
        salience = (intent.get("memory_hint") or {}).get("salience", 0.3)
        prev_id  = None
        for act in (actions or [{"verb":intent.get("action","unknown"),
                                  "subjects":[],"objects":[intent.get("target")],"concept":None}]):
            ev = self._graph.add_event(
                action=act.get("verb","unknown"),
                agent=act.get("subjects") or [],
                objects=[o for o in (act.get("objects") or []) if o],
                concept=act.get("concept"),
                scope=scope, salience=salience,
                trigger_id=prev_id,
                intent_type=intent.get("intent","unknown"),
                raw_text=intent.get("text",""),
            )
            prev_id = ev.id


# ============================================================
# SECTION 16 — Tests (v3 + cas v6)
# ============================================================

def run_tests():
    print("\n"+"="*68)
    print("  TESTS IntentPipeline v4")
    print("="*68)
    passed = failed = 0

    def check(label, cond, detail=""):
        nonlocal passed, failed
        sym = "OK  " if cond else "ECHEC"
        print(f"  {sym} {label}" + (f"  ({detail})" if detail else ""))
        if cond: passed += 1
        else:    failed += 1

    # ── FrenchTextCorrector ──────────────────────────────────
    print("\n  [FrenchTextCorrector]")
    for raw, expected_word in [("jsp pkoi", "sais"), ("ajd je suis là", "aujourd'hui")]:
        result, _ = FrenchTextCorrector.correct(raw)
        check(f"correct '{raw}'", expected_word in result, f"→'{result}'")

    # ── ConversationContext ──────────────────────────────────
    print("\n  [ConversationContext]")
    ctx = ConversationContext()
    ctx.last_action = "jardiner"
    ctx.last_intent_type = "information_input"
    ctx.timestamp = time.time()
    check("ellipsis 'encore'", ctx.resolve_ellipsis("encore") == "refais jardiner")
    ctx.last_location = "jardin"
    check("pronom 'y'",
          "jardin" in ctx.resolve_pronouns("j'y vais demain"))

    # ── split_clauses ────────────────────────────────────────
    print("\n  [split_clauses]")
    for text, expected in [
        ("je jardinerai et je lirai",         2),
        ("Marie et Paul viendront",           1),
        ("j'ai mangé puis je suis sorti",     2),
        ("il pleut aujourd'hui",              1),
    ]:
        parts = split_clauses(text)
        check(f"split '{text[:40]}'", len(parts)==expected,
              f"attendu={expected} obtenu={len(parts)}")

    try:
        import spacy
        nlp = spacy.load(SPACY_MODEL)
    except Exception as e:
        print(f"\n  spaCy absent ({e}) — tests NLU ignorés")
        print(f"\n{'='*68}\n  {passed} OK / {failed} ECHEC\n{'='*68}")
        return

    # ── _extract_all_slots — who ─────────────────────────────
    print("\n  [_extract_all_slots — who clitiques]")
    for text, expected in [
        ("Marie m'a appelé",           "1st_sg"),
        ("je t'ai envoyé un message",  "2nd_sg"),
        ("il nous a expliqué ça",      "1st_pl"),
    ]:
        doc = nlp(text)
        who_p, who_r, _, _, _ = _extract_all_slots(doc)
        check(f"who '{text}'", who_p==expected, f"attendu={expected} obtenu={who_p}({who_r})")

    # ── _extract_verb ────────────────────────────────────────
    print("\n  [_extract_verb]")
    for text, exp_lemma, exp_tense, exp_pol, exp_modal in [
        ("je mange une pomme",    "manger","present","positive",None),
        ("il n'est pas venu",     "venir", "past",   "negative",None),
        ("nous irons demain",     "aller", "future",  "positive",None),
        ("tu devrais partir",     None,    None,      "positive","obligation"),
    ]:
        doc  = nlp(text)
        verb = _extract_verb(doc)
        if verb:
            if exp_lemma:
                check(f"lemma '{text}'", verb.lemma==exp_lemma,
                      f"attendu={exp_lemma} obtenu={verb.lemma}")
            if exp_tense:
                check(f"tense '{text}'", verb.tense==exp_tense,
                      f"attendu={exp_tense} obtenu={verb.tense}")
            check(f"polarity '{text}'", verb.polarity==exp_pol,
                  f"attendu={exp_pol} obtenu={verb.polarity}")
            if exp_modal:
                check(f"modal '{text}'", verb.modal==exp_modal,
                      f"attendu={exp_modal} obtenu={verb.modal}")
        else:
            check(f"verb détecté '{text}'", False, "None")

    # ── FrenchTemporalResolver ───────────────────────────────
    print("\n  [FrenchTemporalResolver]")
    resolver = FrenchTemporalResolver()
    ref      = datetime.datetime(2025, 6, 15, 10, 0)
    for expr, exp_type, exp_key in [
        ("ce soir",                   "INTERVAL",  "iso_start"),
        ("demain",                    "DATE",       "iso_start"),
        ("pendant les vacances d'été","INTERVAL",  "iso_start"),
        ("noël",                      "DATE",       "iso_start"),
        ("lundi prochain",            "DATE",       "iso_start"),
        ("pendant 2 heures",          "DURATION",   "duration_value"),
    ]:
        r = resolver.resolve(expr, ref)
        check(f"temporal '{expr}'", r is not None and exp_key in r,
              f"type={r.get('timex_type') if r else None} src={r.get('source') if r else None}")

    # ── SEMANTIC_MAP ─────────────────────────────────────────
    print("\n  [SEMANTIC_MAP — 11 concepts]")
    for verb, expected in [
        ("manger","INGEST"),("créer","CREATE"),("donner","TRANSFER"),
        ("penser","COGNITION"),("aimer","EMOTION"),("remercier","SOCIAL_ACT"),
        ("dormir","HEALTH"),("voir","PERCEIVE"),("jardiner","CREATE"),
    ]:
        check(f"'{verb}'→{expected}", SEMANTIC_MAP.get(verb)==expected,
              f"obtenu={SEMANTIC_MAP.get(verb)}")

    # ── FrenchVerbAnalyzer ───────────────────────────────────
    print("\n  [FrenchVerbAnalyzer]")
    check("futur 'jardinerai'",
          FrenchVerbAnalyzer.refine_tense("jardinerai","jardiner","unknown")=="future")
    check("présent irrégulier 'vais'",
          FrenchVerbAnalyzer.refine_tense("vais","aller","unknown")=="present")
    check("impersonnel 'falloir'",
          FrenchVerbAnalyzer.is_impersonal("falloir"))
    check("compound 'ai'",
          FrenchVerbAnalyzer.is_compound("ai")=="avoir")

    # ── HybridIntentClassifier — règles ──────────────────────
    print("\n  [HybridIntentClassifier — règles]")
    clf = HybridIntentClassifier(MODEL_DIR/"finetuned")
    for text, exp_intent in [
        ("allume la lumière",       "action_device"),
        ("où es-tu ?",              "query_state"),
        ("bonjour comment vas-tu",  "chit_chat"),
        ("je suis allé au marché",  "information_input"),
    ]:
        r = clf._rules(text.lower())
        check(f"rules '{text}'", r["intent"]==exp_intent,
              f"attendu={exp_intent} obtenu={r['intent']}")

    # ── to_cognitive_frame ───────────────────────────────────
    print("\n  [Intent.to_cognitive_frame]")
    dummy_verb = VerbAnalysis(
        lemma="jardiner", tense="future", scope="FUTURE",
        concept="CREATE", person="1st_sg", number="sg",
        mood="indicative", polarity="positive", modal=None,
    )
    dummy_when = TemporalSpan(raw="demain", iso_start="2025-06-16T08:00:00",
                               iso_end="2025-06-16T10:00:00", timex_type="DATE")
    dummy = Intent(
        text="je jardinerai demain matin",
        intent="information_input", confidence=0.88, uncertain=False,
        scores={}, verb=dummy_verb, who="1st_sg", who_raw="je",
        with_who=[], when=dummy_when, where="jardin",
        what="CREATE", action=None, target=None, actions=[], entities=[],
        memory_hint=None, register=None, assembled_from=[],
    )
    cf = dummy.to_cognitive_frame()
    check("to_cognitive_frame verb",     cf["verb"]=="jardiner")
    check("to_cognitive_frame scope",    cf["scope"]=="FUTURE")
    check("to_cognitive_frame time_end", cf["time_end"]=="2025-06-16T10:00:00")
    check("to_cognitive_frame location", cf["location"]=="jardin")

    print(f"\n{'='*68}")
    print(f"  {passed} OK  /  {failed} ECHEC  /  {passed+failed} total")
    print("="*68+"\n")


def run_graph_tests():
    print("\n"+"="*68)
    print("  TESTS EventGraph v4")
    print("="*68)
    passed = failed = 0

    def check(label, cond, detail=""):
        nonlocal passed, failed
        sym = "OK  " if cond else "ECHEC"
        print(f"  {sym} {label}" + (f"  ({detail})" if detail else ""))
        if cond: passed += 1
        else:    failed += 1

    graph = EventGraph()
    q     = Queue()
    cons  = EventGraphConsumer(q, graph)
    cons.start()

    batch = [
        {"text":"je planterai des fleurs","intent":"information_input",
         "verb":{"scope":"FUTURE"},"who":"1st_sg","who_raw":"je","with_who":[],
         "when":{"iso_start":"2025-06-16T08:00:00","iso_end":"2025-06-16T10:00:00"},
         "where":"jardin","what":"CREATE","action":None,"target":None,"corrections":[],
         "actions":[{"verb":"planter","subjects":["je"],"objects":["fleurs"],"concept":"CREATE"}],
         "entities":[],"register":None,
         "memory_hint":{"subject":"human","salience":0.42,"entities":[],"raw_info":"..."},
         "assembled_from":[],"clause_index":0,"scores":{},"uncertain":False,
         "confidence":0.88,"processing_ms":45.0,"ts":time.time()},
        {"text":"je jardinerai ensuite","intent":"information_input",
         "verb":{"scope":"FUTURE"},"who":"1st_sg","who_raw":"je","with_who":[],
         "when":None,"where":"jardin","what":"CREATE","action":None,"target":None,"corrections":[],
         "actions":[{"verb":"jardiner","subjects":["je"],"objects":[],"concept":"CREATE"}],
         "entities":[],"register":None,
         "memory_hint":{"subject":"human","salience":0.38,"entities":[],"raw_info":"..."},
         "assembled_from":[],"clause_index":1,"scores":{},"uncertain":False,
         "confidence":0.85,"processing_ms":40.0,"ts":time.time()},
    ]
    q.put(batch)
    q.put([{"text":"allume la lumière","intent":"action_device",
            "verb":{"scope":"PRESENT"},"who":None,"who_raw":None,"with_who":[],
            "when":None,"where":"salon","what":None,"action":"turn_on","target":"light",
            "corrections":[],"actions":[{"verb":"allumer","subjects":[],"objects":["lumière"],"concept":None}],
            "entities":[],"register":None,"memory_hint":None,"assembled_from":[],"clause_index":0,
            "scores":{},"uncertain":False,"confidence":0.97,"processing_ms":22.0,"ts":time.time()}])
    q.put([{"text":"qu'as-tu fait ?","intent":"query_narrative",
            "verb":{"scope":"PAST"},"who":"2nd_sg","who_raw":"tu","with_who":[],
            "when":None,"where":None,"what":None,"action":None,"target":None,"corrections":[],
            "actions":[],"entities":[],"register":None,"memory_hint":None,
            "assembled_from":[],"clause_index":0,"scores":{},"uncertain":False,
            "confidence":0.92,"processing_ms":30.0,"ts":time.time()}])

    time.sleep(0.5)
    cons.stop(); cons.join(timeout=2.0)

    s = graph.summary()
    check("events >= 3",       s["total_events"]>=3,    f"total={s['total_events']}")
    check("query sans event",  not any(e.intent_type=="query_narrative" for e in graph.events))
    check("FUTURE >= 2",       s["by_scope"]["FUTURE"]>=2)
    check("action_device",     any(e.intent_type=="action_device" for e in graph.events))
    check("location jardin",   any(e.location=="jardin" for e in graph.events))
    check("time_base set",     any(e.start_time is not None for e in graph.events))

    print(f"\n  Résumé : {s}")
    print("  Events:")
    for e in graph.events:
        print(f"    #{e.id} [{e.scope}] {e.action}({e.agent}) "
              f"loc={e.location} t={e.start_time} sal={e.salience:.2f}")

    print(f"\n{'='*68}")
    print(f"  {passed} OK  /  {failed} ECHEC  /  {passed+failed} total")
    print("="*68+"\n")


# ============================================================
# SECTION 17 — Point d'entrée
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IntentPipeline v4 — NLU français fusionné v3+v6")
    parser.add_argument("--download",    action="store_true")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--data",        type=str, default=str(DATA_FILE))
    parser.add_argument("--test",        action="store_true")
    parser.add_argument("--test-graph",  action="store_true")
    parser.add_argument("--benchmark",   action="store_true")
    parser.add_argument("--predict",     type=str)
    parser.add_argument("--batch",       type=str, nargs="+")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--no-corrector",action="store_true")
    args = parser.parse_args()

    if args.download:
        download_model()

    elif args.train:
        train(Path(args.data))

    elif args.test:
        run_tests()

    elif args.test_graph:
        run_graph_tests()

    elif args.benchmark:
        q_in, q_out = Queue(), Queue()
        p = IntentPipeline(q_in, q_out, debug=False)
        p.load()
        print(json.dumps(p.benchmark(n=30), indent=2))
        print(json.dumps(p.get_metrics(), indent=2))

    elif args.predict:
        q_in, q_out = Queue(), Queue()
        p = IntentPipeline(q_in, q_out, debug=args.debug)
        if args.no_corrector: p.enable_corrector(False)
        p.load(); p.start()
        q_in.put(args.predict)
        try:
            result = q_out.get(timeout=10.0)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Empty:
            print("Timeout")
        p.stop()

    elif args.batch:
        q_in, q_out = Queue(), Queue()
        p = IntentPipeline(q_in, q_out, debug=args.debug)
        p.load()
        results = p.batch_process(args.batch)
        for r in results:
            print(json.dumps(r.to_cognitive_frame(), ensure_ascii=False, indent=2))

    elif args.interactive:
        print("Mode interactif — 'quit' pour quitter, 'metrics' pour les stats.")
        q_in, q_out = Queue(), Queue()
        p = IntentPipeline(q_in, q_out, debug=args.debug)
        if args.no_corrector: p.enable_corrector(False)
        p.load(); p.start()
        try:
            while True:
                raw = input("\n> ").strip()
                if not raw: continue
                if raw.lower() in ("quit","exit","q"): break
                if raw == "metrics":
                    print(json.dumps(p.get_metrics(), indent=2, ensure_ascii=False))
                    continue
                q_in.put(raw)
                try:
                    result = q_out.get(timeout=10.0)
                    for intent in result:
                        cf = Intent(**{k:v for k,v in intent.items()
                                       if k in Intent.__dataclass_fields__})
                        print(json.dumps(cf.to_cognitive_frame(),
                                         ensure_ascii=False, indent=2))
                except Empty:
                    print("Timeout")
        except KeyboardInterrupt:
            print("\nAu revoir.")
        finally:
            p.stop()

    else:
        parser.print_help()
