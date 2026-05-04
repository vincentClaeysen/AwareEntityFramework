#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intent_pipeline_v4.py — Pipeline NLU français → Intent(s) structuré(s)
=======================================================================
Version finale pour ASE (Aware Synthetic Entity)
Raspberry Pi 5, 100 % offline, entièrement déterministe.

Composants intégrés :
  - NLU complet (verbes, temps, participants, registre)
  - Relations sémantiques (11 types + extensible)
  - Coreférence (pronom → antécédent)
  - Attributs (état, qualité, adjectifs épithètes)
  - Événements (action, acteur, patient, instrument)
  - Quantifieurs, négations, modalités, comparaisons, rôles
  - Propositions subordonnées, intentions secondaires, temps relatifs
  - Résolution temporelle française (saisons, vacances, jours fériés)
  - Exports triplets RDF pour CognitionCore
  - Imports optionnels, fallbacks, cache LRU
  - Multi-intent (split_clauses)
  - Contexte conversationnel (ellipses, pronoms)
  - Thread-safe (File d'attente, batch optionnel)

Dépendances minimales :
    pip install spacy numpy dateparser python-dateutil workalendar
    python -m spacy download fr_core_news_sm

Pour tout faire fonctionner (recommandé) :
    pip install spacy numpy dateparser python-dateutil workalendar vacances-scolaires python-Levenshtein timexy
"""

import datetime
import json
import logging
import re
import threading
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List, Literal, Optional, Tuple, Set, Any

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
logger = logging.getLogger("IntentPipeline")
error_logger = logging.getLogger("IntentPipeline.Errors")
_err_handler = logging.FileHandler("pipeline_errors.log")
_err_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
error_logger.addHandler(_err_handler)
error_logger.setLevel(logging.WARNING)

# ============================================================
# Types stricts
# ============================================================

IntentType = Literal["query_narrative", "query_state", "query_intention",
                      "action_device", "information_input", "chit_chat",
                      "reminder", "warning", "condition", "suggestion", "obligation",
                      "scheduled_action"]  # NOUVEAU
TenseType = Literal["past", "present", "future", "conditional", "unknown"]
PersonType = Literal["1st_sg", "2nd_sg", "3rd_sg", "1st_pl", "2nd_pl", "3rd_pl", "unknown"]
MoodType = Literal["indicative", "subjunctive", "imperative", "conditional", "infinitive"]
PolarityType = Literal["positive", "negative"]
ModalType = Literal["obligation", "possibility", "volition", "wish", None]
ConceptType = Literal["PERCEIVE", "COMMUNICATE", "BE", "MOVE", "INGEST", "CREATE",
                       "TRANSFER", "COGNITION", "EMOTION", "SOCIAL_ACT", "HEALTH", None]
ScopeType = Literal["PAST", "PRESENT", "FUTURE", "HYPOTHETICAL", "UNKNOWN"]
RoleType = Literal["user", "ase", "other"]

# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path("./model_camembert")
SPACY_MODEL = "fr_core_news_sm"
LANGUAGE = "fr"
MAX_LENGTH = 64
CONFIDENCE_THRESHOLD = 0.60
CONTEXT_RESET_SEC = 30.0
CACHE_SIZE = 256
MAX_WORKERS = 2

INTENTS: List[IntentType] = [
    "query_narrative", "query_state", "query_intention",
    "action_device", "information_input", "chit_chat",
    "reminder", "warning", "condition", "suggestion", "obligation", "scheduled_action",
]

NER_TYPE_MAP = {
    "PER": "person", "LOC": "location", "GPE": "location", "ORG": "organization",
    "DATE": "date", "TIME": "time", "CARDINAL": "number", "MONEY": "money",
}

UNIT_PATTERNS = {
    "km": "km", "kilometre": "km", "m": "m", "metre": "m", "cm": "cm",
    "kg": "kg", "g": "g", "l": "L", "cl": "cL", "ml": "mL",
    "h": "h", "heure": "h", "min": "min", "minute": "min", "s": "s", "seconde": "s",
}

INFO_SUBJECTS = {"je", "j'", "j", "moi", "nous", "on"}

CLITIC_TO_PERSON: Dict[str, PersonType] = {
    "me": "1st_sg", "m'": "1st_sg", "moi": "1st_sg", "te": "2nd_sg", "t'": "2nd_sg",
    "toi": "2nd_sg", "se": "3rd_sg", "s'": "3rd_sg", "lui": "3rd_sg", "elle": "3rd_sg",
    "nous": "1st_pl", "on": "1st_pl", "vous": "2nd_pl", "leur": "3rd_pl", "eux": "3rd_pl",
}

MODAL_MARKERS: Dict[str, ModalType] = {
    "devoir": "obligation", "falloir": "obligation", "pouvoir": "possibility",
    "vouloir": "volition", "souhaiter": "volition", "aimer": "wish",
}

TENSE_TO_SCOPE: Dict[str, ScopeType] = {
    "past": "PAST", "present": "PRESENT", "future": "FUTURE",
    "conditional": "HYPOTHETICAL", "unknown": "UNKNOWN",
}

ACTION_VERBS: Dict[str, str] = {
    "allumer": "turn_on", "activer": "turn_on", "lancer": "turn_on", "ouvrir": "turn_on",
    "éteindre": "turn_off", "couper": "turn_off", "désactiver": "turn_off", "arrêter": "turn_off",
    "monter": "set_up", "augmenter": "set_up", "baisser": "set_down", "diminuer": "set_down",
    "régler": "set", "mettre": "set", "configurer": "set", "programmer": "set",
}

DEVICE_NOUNS: Dict[str, str] = {
    "lumière": "light", "lampe": "light", "télé": "tv", "écran": "tv", "musique": "music",
    "chauffage": "heating", "radiateur": "heating", "volet": "shutter", "porte": "door",
    "alarme": "alarm", "clim": "ac", "ventilateur": "fan",
}

# ============================================================
# SEMANTIC_MAP
# ============================================================

SEMANTIC_MAP: Dict[str, ConceptType] = {
    "voir": "PERCEIVE", "regarder": "PERCEIVE", "entendre": "PERCEIVE", "écouter": "PERCEIVE",
    "dire": "COMMUNICATE", "parler": "COMMUNICATE", "demander": "COMMUNICATE", "répondre": "COMMUNICATE",
    "aller": "MOVE", "venir": "MOVE", "partir": "MOVE", "arriver": "MOVE", "marcher": "MOVE",
    "manger": "INGEST", "boire": "INGEST",
    "faire": "CREATE", "fabriquer": "CREATE", "construire": "CREATE", "créer": "CREATE",
    "donner": "TRANSFER", "recevoir": "TRANSFER", "envoyer": "TRANSFER",
    "penser": "COGNITION", "savoir": "COGNITION", "comprendre": "COGNITION", "apprendre": "COGNITION",
    "aimer": "EMOTION", "adorer": "EMOTION", "détester": "EMOTION",
    "saluer": "SOCIAL_ACT", "remercier": "SOCIAL_ACT",
    "dormir": "HEALTH", "se reposer": "HEALTH",
    "être": "BE", "rester": "BE", "devenir": "BE",
}

REGISTER_LEXICON = {
    "familier": {"ouais", "nan", "jsuis", "y'a", "wesh", "bah", "ben", "quoi", "trop", "grave", "cool", "mdr"},
    "soutenu": {"néanmoins", "cependant", "toutefois", "ainsi", "permettez", "veuillez"},
    "affectif": {"oh", "ah", "hélas", "magnifique", "incroyable"},
    "technique": {"paramètre", "configuration", "module", "interface"},
}

NEGATION_COMPLETE = {"ne", "n'"}
TUTOIEMENT_MARKERS = {"tu", "toi", "t'", "te"}
VOUVOIEMENT_MARKERS = {"vous", "votre", "vos"}

# ============================================================
# NOUVELLES STRUCTURES
# ============================================================

@dataclass
class ScheduledTime:
    """Informations de temps programmé pour une action."""
    raw: str
    iso: str
    human: str
    delay_seconds: Optional[int] = None
    is_recurring: bool = False
    recurrence_rule: Optional[str] = None  # "daily", "weekly", "monthly"

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "iso": self.iso,
            "human": self.human,
            "delay_seconds": self.delay_seconds,
            "is_recurring": self.is_recurring,
            "recurrence_rule": self.recurrence_rule,
        }


@dataclass
class Relation:
    predicate: str
    confidence: float = 0.0
    subject: Optional[str] = None
    object: Optional[str] = None
    source_text: str = ""
    position_start: int = 0
    position_end: int = 0
    is_known: bool = True
    unknown_text: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "predicate": self.predicate,
            "confidence": self.confidence,
            "subject": self.subject,
            "object": self.object,
            "source_text": self.source_text,
            "is_known": self.is_known,
            "unknown_text": self.unknown_text,
            "arguments": self.arguments,
            "properties": self.properties,
            "tags": self.tags,
        }

    def to_triple(self) -> Tuple[Optional[str], str, Optional[str]]:
        return (self.subject, self.predicate, self.object)


@dataclass
class Coreference:
    pronoun: str
    antecedent: str
    position_pronoun: int
    position_antecedent: int
    confidence: float = 0.9
    gender_match: bool = False
    number_match: bool = False
    cluster_id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Attribute:
    entity: str
    attribute: str
    attribute_type: str = "state"
    confidence: float = 0.8
    source_text: str = ""
    is_temporary: bool = True
    is_epithet: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Event:
    action: str
    actor: Optional[str] = None
    patient: Optional[str] = None
    instrument: Optional[str] = None
    location: Optional[str] = None
    time: Optional[str] = None
    duration: Optional[str] = None
    confidence: float = 0.8
    source_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Quantifier:
    entity: str
    quantifier: str
    quantifier_type: str
    value: Optional[float] = None
    confidence: float = 0.9

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Negation:
    negated_element: str
    negation_word: str
    scope: str
    confidence: float = 0.9

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Modality:
    statement: str
    modality_type: str
    strength: float = 0.8
    source_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Comparison:
    subject: str
    comparator: str
    attribute: str
    object: str
    degree: str
    confidence: float = 0.85

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Role:
    entity: str
    role: str
    context: Optional[str] = None
    confidence: float = 0.85

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TemporalSequence:
    first_event: str
    second_event: str
    relation: str
    confidence: float = 0.85

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubordinateClause:
    main_clause: str
    sub_clause: str
    relation: str
    confidence: float = 0.85
    source_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SecondaryIntent:
    primary_intent: str
    secondary_intent: str
    trigger: str
    action: Optional[str] = None
    target: Optional[str] = None
    scheduled_time: Optional[dict] = None
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "primary_intent": self.primary_intent,
            "secondary_intent": self.secondary_intent,
            "trigger": self.trigger,
            "action": self.action,
            "target": self.target,
            "scheduled_time": self.scheduled_time,
            "confidence": self.confidence,
        }


@dataclass
class RelativeTense:
    before_event: str
    after_event: str
    relation: str
    time_gap: Optional[str] = None
    confidence: float = 0.85

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# STRUCTURES EXISTANTES
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
    scheduled: Optional[ScheduledTime] = None  # NOUVEAU


@dataclass
class SyntacticTree:
    phrase: str
    root: Optional[Any] = None
    subject: Optional[str] = None
    subject_role: Optional[RoleType] = None
    object_: Optional[str] = None

    def to_dict(self) -> dict:
        return {"phrase": self.phrase, "subject": self.subject, "object": self.object_}


@dataclass
class RegisterAnalysis:
    style: str = "neutre"
    confidence: float = 0.5
    markers: List[str] = field(default_factory=list)
    politeness: bool = False  # NOUVEAU


# ============================================================
# INTENT COMPLET
# ============================================================

@dataclass
class Intent:
    # Champs existants
    text: str
    intent: IntentType
    confidence: float
    uncertain: bool
    scores: dict
    verb: Optional[VerbAnalysis]
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

    # Nouveaux champs pour graphe mémoire
    relations: List[Relation] = field(default_factory=list)
    triplets: List[Tuple[str, str, str]] = field(default_factory=list)
    coreferences: List[Coreference] = field(default_factory=list)
    attributes: List[Attribute] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    quantifiers: List[Quantifier] = field(default_factory=list)
    negations: List[Negation] = field(default_factory=list)
    modalities: List[Modality] = field(default_factory=list)
    comparisons: List[Comparison] = field(default_factory=list)
    roles: List[Role] = field(default_factory=list)
    temporal_sequences: List[TemporalSequence] = field(default_factory=list)
    subordinate_clauses: List[SubordinateClause] = field(default_factory=list)
    secondary_intents: List[SecondaryIntent] = field(default_factory=list)
    relative_tenses: List[RelativeTense] = field(default_factory=list)
    scheduled_time: Optional[ScheduledTime] = None  # NOUVEAU
    entities_set: Set[str] = field(default_factory=set)
    facts: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["relations"] = [r.to_dict() for r in self.relations]
        d["coreferences"] = [c.to_dict() for c in self.coreferences]
        d["attributes"] = [a.to_dict() for a in self.attributes]
        d["events"] = [e.to_dict() for e in self.events]
        d["quantifiers"] = [q.to_dict() for q in self.quantifiers]
        d["negations"] = [n.to_dict() for n in self.negations]
        d["modalities"] = [m.to_dict() for m in self.modalities]
        d["comparisons"] = [c.to_dict() for c in self.comparisons]
        d["roles"] = [r.to_dict() for r in self.roles]
        d["temporal_sequences"] = [t.to_dict() for t in self.temporal_sequences]
        d["subordinate_clauses"] = [s.to_dict() for s in self.subordinate_clauses]
        d["secondary_intents"] = [s.to_dict() for s in self.secondary_intents]
        d["relative_tenses"] = [r.to_dict() for r in self.relative_tenses]
        if self.scheduled_time:
            d["scheduled_time"] = self.scheduled_time.to_dict()
        d["entities_set"] = list(self.entities_set)
        d["facts"] = list(self.facts)
        return d

    def to_cognitive_frame(self) -> dict:
        return {
            "type": "intent",
            "intent_type": self.intent,
            "verb": self.verb.lemma if self.verb else None,
            "concept": self.verb.concept if self.verb else None,
            "scope": self.verb.scope if self.verb else "UNKNOWN",
            "who": self.who,
            "who_raw": self.who_raw,
            "with_who": self.with_who,
            "time_start": self.when.iso_start if self.when else None,
            "time_end": self.when.iso_end if self.when else None,
            "location": self.where,
            "action": self.action,
            "target": self.target,
            "confidence": self.confidence,
            "processing_ms": self.processing_ms,
            "scheduled_time": self.scheduled_time.to_dict() if self.scheduled_time else None,
            "relations": [r.to_dict() for r in self.relations],
            "triplets": self.triplets,
            "coreferences": [c.to_dict() for c in self.coreferences],
            "attributes": [a.to_dict() for a in self.attributes],
            "events": [e.to_dict() for e in self.events],
            "quantifiers": [q.to_dict() for q in self.quantifiers],
            "negations": [n.to_dict() for n in self.negations],
            "modalities": [m.to_dict() for m in self.modalities],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "roles": [r.to_dict() for r in self.roles],
            "temporal_sequences": [t.to_dict() for t in self.temporal_sequences],
            "subordinate_clauses": [s.to_dict() for s in self.subordinate_clauses],
            "secondary_intents": [s.to_dict() for s in self.secondary_intents],
            "relative_tenses": [r.to_dict() for r in self.relative_tenses],
            "entities": list(self.entities_set),
            "facts": list(self.facts),
        }


# ============================================================
# EXTRACTEURS
# ============================================================

class ScheduledTimeExtractor:
    """Extrait les temps programmés pour les actions."""
    
    TIME_UNITS = {
        "seconde": 1, "secondes": 1, "s": 1,
        "minute": 60, "minutes": 60, "min": 60,
        "heure": 3600, "heures": 3600, "h": 3600,
        "jour": 86400, "jours": 86400, "j": 86400,
        "semaine": 604800, "semaines": 604800, "sem": 604800,
        "mois": 2592000,
        "an": 31536000, "ans": 31536000, "année": 31536000, "années": 31536000,
    }
    
    DELAY_PATTERNS = [
        r"dans\s+(\d+(?:[.,]\d+)?)\s*(secondes?|s?|minutes?|min?|heures?|h?|jours?|j?|semaines?|sem?|mois|ans?|années?)",
        r"dans\s+(une|un)\s+(seconde|minute|heure|jour|semaine|mois|an|année)",
        r"dans\s+(\d{1,2})h(\d{1,2})",
    ]
    
    ABSOLUTE_PATTERNS = [
        r"le\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})(?:\s+à\s+(\d{1,2})[h:](\d{0,2}))?",
        r"le\s+(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)(?:\s+(\d{4}))?(?:\s+à\s+(\d{1,2})[h:](\d{0,2}))?",
        r"à\s+(\d{1,2})[h:](\d{0,2})",
        r"(\d{1,2})[h:](\d{0,2})",
        r"(\d{1,2}):(\d{2})",
        r"demain\s+à\s+(\d{1,2})[h:](\d{0,2})",
        r"ce\s+(soir|matin|après-midi)",
    ]
    
    RECURRING_PATTERNS = [
        (r"tous\s+les\s+(jours|soirs|matins)", "daily"),
        (r"chaque\s+(jour|soir|matin)", "daily"),
        (r"toutes\s+les\s+(semaines|semaine)", "weekly"),
        (r"chaque\s+semaine", "weekly"),
        (r"tous\s+les\s+(mois|mois)", "monthly"),
        (r"chaque\s+mois", "monthly"),
    ]
    
    @classmethod
    def extract(cls, text: str, ref: datetime.datetime = None) -> Optional[ScheduledTime]:
        if ref is None:
            ref = datetime.datetime.now()
        
        text_lower = text.lower()
        
        # Vérifier si c'est récurrent
        is_recurring = False
        recurrence_rule = None
        for pattern, rule in cls.RECURRING_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                is_recurring = True
                recurrence_rule = rule
                break
        
        # Délais
        for pattern in cls.DELAY_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return cls._parse_delay(match, ref, is_recurring, recurrence_rule)
        
        # Dates absolues
        for pattern in cls.ABSOLUTE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                return cls._parse_absolute(match, ref, is_recurring, recurrence_rule)
        
        return None
    
    @classmethod
    def _parse_delay(cls, match, ref, is_recurring, recurrence_rule) -> Optional[ScheduledTime]:
        groups = match.groups()
        raw = match.group(0)
        
        # Cas "dans 2h30"
        if len(groups) >= 2 and groups[1] and groups[0] and not groups[0].isalpha():
            try:
                hours = int(groups[0])
                mins = int(groups[1]) if groups[1] else 0
                total_seconds = hours * 3600 + mins * 60
                target = ref + datetime.timedelta(seconds=total_seconds)
                return ScheduledTime(
                    raw=raw,
                    iso=target.isoformat(),
                    human=target.strftime("%A %d %B %Y à %H:%M"),
                    delay_seconds=total_seconds,
                    is_recurring=is_recurring,
                    recurrence_rule=recurrence_rule,
                )
            except (ValueError, TypeError):
                pass
        
        # Cas "dans une heure"
        if groups[0] in ("une", "un"):
            unit = groups[1] if len(groups) > 1 else groups[0]
            unit_clean = unit.rstrip("s")
            seconds = cls.TIME_UNITS.get(unit_clean, 0)
            target = ref + datetime.timedelta(seconds=seconds)
            return ScheduledTime(
                raw=raw,
                iso=target.isoformat(),
                human=target.strftime("%A %d %B %Y à %H:%M"),
                delay_seconds=seconds,
                is_recurring=is_recurring,
                recurrence_rule=recurrence_rule,
            )
        
        # Cas "dans 20 minutes"
        try:
            value = float(groups[0].replace(",", "."))
            unit = groups[1].rstrip("s")
            seconds = int(value * cls.TIME_UNITS.get(unit, 0))
            target = ref + datetime.timedelta(seconds=seconds)
            return ScheduledTime(
                raw=raw,
                iso=target.isoformat(),
                human=target.strftime("%A %d %B %Y à %H:%M"),
                delay_seconds=seconds,
                is_recurring=is_recurring,
                recurrence_rule=recurrence_rule,
            )
        except (ValueError, TypeError, KeyError):
            pass
        
        return None
    
    @classmethod
    def _parse_absolute(cls, match, ref, is_recurring, recurrence_rule) -> Optional[ScheduledTime]:
        groups = match.groups()
        raw = match.group(0)
        
        # Cas "demain à 20h30"
        if "demain" in raw.lower():
            target = ref + datetime.timedelta(days=1)
            hour = int(groups[0]) if groups[0] else 20
            minute = int(groups[1]) if len(groups) > 1 and groups[1] else 0
            target = target.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return ScheduledTime(
                raw=raw,
                iso=target.isoformat(),
                human=target.strftime("%A %d %B %Y à %H:%M"),
                is_recurring=is_recurring,
                recurrence_rule=recurrence_rule,
            )
        
        # Cas "ce soir" / "ce matin" / "cet après-midi"
        if "ce soir" in raw.lower():
            target = ref.replace(hour=20, minute=0, second=0, microsecond=0)
            if target < ref:
                target += datetime.timedelta(days=1)
            return ScheduledTime(
                raw=raw,
                iso=target.isoformat(),
                human=target.strftime("%A %d %B %Y à %H:%M"),
                is_recurring=is_recurring,
                recurrence_rule=recurrence_rule,
            )
        
        if "ce matin" in raw.lower():
            target = ref.replace(hour=8, minute=0, second=0, microsecond=0)
            if target < ref:
                target += datetime.timedelta(days=1)
            return ScheduledTime(
                raw=raw,
                iso=target.isoformat(),
                human=target.strftime("%A %d %B %Y à %H:%M"),
                is_recurring=is_recurring,
                recurrence_rule=recurrence_rule,
            )
        
        if "cet après-midi" in raw.lower():
            target = ref.replace(hour=14, minute=0, second=0, microsecond=0)
            if target < ref:
                target += datetime.timedelta(days=1)
            return ScheduledTime(
                raw=raw,
                iso=target.isoformat(),
                human=target.strftime("%A %d %B %Y à %H:%M"),
                is_recurring=is_recurring,
                recurrence_rule=recurrence_rule,
            )
        
        # Cas "à 20h30" ou "20h30"
        if len(groups) >= 2 and groups[0] and groups[0].isdigit() and len(groups[0]) <= 2:
            hour = int(groups[0])
            minute = int(groups[1]) if groups[1] else 0
            target = ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target < ref:
                target += datetime.timedelta(days=1)
            return ScheduledTime(
                raw=raw,
                iso=target.isoformat(),
                human=target.strftime("%A %d %B %Y à %H:%M"),
                is_recurring=is_recurring,
                recurrence_rule=recurrence_rule,
            )
        
        # Cas date complète
        if len(groups) >= 3 and groups[0]:
            date_str = groups[0]
            hour = int(groups[2]) if len(groups) > 2 and groups[2] else 0
            minute = int(groups[3]) if len(groups) > 3 and groups[3] else 0
            
            if DATEPARSER_AVAILABLE:
                dt = dateparser.parse(date_str, languages=["fr"])
                if dt:
                    target = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    return ScheduledTime(
                        raw=raw,
                        iso=target.isoformat(),
                        human=target.strftime("%A %d %B %Y à %H:%M"),
                        is_recurring=is_recurring,
                        recurrence_rule=recurrence_rule,
                    )
        
        return None


class RelationExtractor:
    """Extrait les relations sémantiques."""
    
    FR_TO_PREDICATE = {
        "fils": "SON_OF", "fille": "DAUGHTER_OF", "père": "FATHER_OF",
        "mère": "MOTHER_OF", "frère": "BROTHER_OF", "soeur": "SISTER_OF",
        "mari": "HUSBAND_OF", "femme": "WIFE_OF", "parent": "PARENT_OF",
        "enfant": "CHILD_OF", "possède": "OWNS", "appartient": "BELONGS_TO",
        "ami": "FRIEND_OF", "collègue": "COLLEAGUE_OF", "voisin": "NEIGHBOR_OF",
        "donne": "GIVES_TO", "envoie": "SENDS_TO", "reçoit": "RECEIVES_FROM",
    }
    
    PREP_TO_PREDICATE = {
        "de": "BELONGS_TO", "du": "BELONGS_TO", "des": "BELONGS_TO",
        "à": "GOES_TO", "au": "GOES_TO", "avec": "ACCOMPANIED_BY",
        "pour": "FOR", "sans": "WITHOUT", "dans": "LOCATED_IN",
        "sur": "ON", "sous": "UNDER", "chez": "AT", "vers": "TOWARDS",
    }
    
    @classmethod
    def extract(cls, doc) -> List[Relation]:
        relations = []
        relations.extend(cls._extract_nmod_relations(doc))
        relations.extend(cls._extract_prep_relations(doc))
        relations.extend(cls._extract_copula_relations(doc))
        relations.extend(cls._extract_verb_relations(doc))
        relations.extend(cls._extract_coordination_relations(doc))
        relations.extend(cls._extract_relative_clause_relations(doc))
        relations.extend(cls._extract_presentation_relations(doc))
        return relations
    
    @classmethod
    def _extract_coordination_relations(cls, doc) -> List[Relation]:
        relations = []
        for token in doc:
            if token.dep_ == "cc" and token.text.lower() in ("et", "ou"):
                conj = token.head
                if conj.dep_ == "conj":
                    left = None
                    right = None
                    for child in conj.children:
                        if child.dep_ == "conj" and child.i < conj.i:
                            left = child.text
                        elif child.dep_ == "conj" and child.i > conj.i:
                            right = child.text
                    if left and right:
                        relations.append(Relation(
                            subject=left, predicate="COORDINATED_WITH",
                            object=right, confidence=0.85,
                            source_text=f"{left} {token.text} {right}",
                            tags=["coordination"],
                        ))
        return relations
    
    @classmethod
    def _extract_relative_clause_relations(cls, doc) -> List[Relation]:
        relations = []
        for token in doc:
            if token.dep_ == "relcl" and token.pos_ == "VERB":
                antecedent = None
                for child in token.children:
                    if child.dep_ == "nsubj":
                        antecedent = child.text
                        break
                object_ = None
                for child in doc:
                    if child.dep_ == "obj" and child.head == token:
                        object_ = child.text
                        break
                if antecedent and object_:
                    relations.append(Relation(
                        subject=antecedent, predicate="RELATIVE_TO",
                        object=object_, confidence=0.80,
                        source_text=f"{antecedent} {token.text} {object_}",
                        tags=["relative_clause"],
                    ))
        return relations
    
    @classmethod
    def _extract_presentation_relations(cls, doc) -> List[Relation]:
        relations = []
        for token in doc:
            if token.text.lower() == "c'" and token.head.text.lower() == "est":
                verb = token.head
                for child in verb.children:
                    if child.dep_ == "attr":
                        subject = child.text
                    if child.dep_ == "advcl" and child.text.lower() == "qui":
                        for grandchild in child.children:
                            if grandchild.dep_ == "ROOT":
                                action = grandchild.lemma_
                                for gchild in grandchild.children:
                                    if gchild.dep_ == "obj":
                                        object_ = gchild.text
                                if subject and action:
                                    relations.append(Relation(
                                        subject=subject, predicate=action.upper(),
                                        object=object_, confidence=0.85,
                                        source_text=f"c'est {subject} qui {grandchild.text}",
                                        tags=["presentation"],
                                    ))
        return relations
    
    @classmethod
    def _extract_nmod_relations(cls, doc) -> List[Relation]:
        relations = []
        for token in doc:
            if token.dep_ == "nmod" and token.head.pos_ in ("NOUN", "PROPN"):
                subject = token.head.text
                preposition = None
                object_ = None
                for child in token.children:
                    if child.dep_ == "case":
                        preposition = child.text.lower()
                    elif child.dep_ == "nmod" or child.pos_ == "NOUN":
                        object_ = child.text
                if object_:
                    predicate = cls.FR_TO_PREDICATE.get(subject.lower(), "UNKNOWN")
                    if predicate == "UNKNOWN" and preposition:
                        predicate = cls.PREP_TO_PREDICATE.get(preposition, "UNKNOWN")
                    relations.append(Relation(
                        subject=subject, predicate=predicate, object=object_,
                        confidence=0.85 if predicate != "UNKNOWN" else 0.5,
                        source_text=f"{subject} {preposition or ''} {object_}",
                        is_known=predicate != "UNKNOWN",
                        unknown_text=subject if predicate == "UNKNOWN" else None,
                    ))
        return relations
    
    @classmethod
    def _extract_prep_relations(cls, doc) -> List[Relation]:
        relations = []
        for token in doc:
            if token.dep_ == "prep":
                preposition = token.text.lower()
                parent = token.head
                object_ = None
                for child in token.children:
                    if child.dep_ == "pobj":
                        object_ = child.text
                        break
                if object_ and parent.pos_ in ("VERB", "NOUN", "PROPN"):
                    subject = cls._find_subject(doc, parent)
                    predicate = cls.PREP_TO_PREDICATE.get(preposition, "UNKNOWN")
                    relations.append(Relation(
                        subject=subject or parent.text, predicate=predicate, object=object_,
                        confidence=0.80 if predicate != "UNKNOWN" else 0.5,
                        source_text=f"{parent.text} {preposition} {object_}",
                        is_known=predicate != "UNKNOWN",
                        unknown_text=preposition if predicate == "UNKNOWN" else None,
                    ))
        return relations
    
    @classmethod
    def _extract_copula_relations(cls, doc) -> List[Relation]:
        relations = []
        for token in doc:
            if token.lemma_ in ("être", "devenir") and token.dep_ == "ROOT":
                subject = cls._find_subject(doc, token)
                if not subject:
                    continue
                for child in token.children:
                    if child.dep_ == "attr" and child.pos_ in ("NOUN", "PROPN", "ADJ"):
                        attribute = child.text
                        has_nmod = False
                        for grandchild in child.children:
                            if grandchild.dep_ == "nmod":
                                obj = cls._extract_nmod_object(grandchild)
                                if obj:
                                    predicate = cls.FR_TO_PREDICATE.get(attribute.lower(), "UNKNOWN")
                                    relations.append(Relation(
                                        subject=subject, predicate=predicate, object=obj,
                                        confidence=0.90, source_text=f"{attribute} de {obj}",
                                        is_known=predicate != "UNKNOWN",
                                    ))
                                    has_nmod = True
                        if not has_nmod:
                            relations.append(Relation(
                                subject=subject, predicate="IS", object=attribute,
                                confidence=0.70, source_text=f"{subject} est {attribute}",
                                is_known=True,
                            ))
        return relations
    
    @classmethod
    def _extract_verb_relations(cls, doc) -> List[Relation]:
        relations = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                subject = cls._find_subject(doc, token)
                direct_object = cls._find_direct_object(token)
                for child in token.children:
                    if child.dep_ == "prep":
                        indirect_object = cls._find_pobj(child)
                        if indirect_object:
                            predicate = cls.PREP_TO_PREDICATE.get(child.text.lower(), "UNKNOWN")
                            if token.lemma_ == "donner" and child.text.lower() == "à":
                                predicate = "GIVES_TO"
                            relations.append(Relation(
                                subject=subject, predicate=predicate, object=indirect_object,
                                confidence=0.85,
                                source_text=f"{subject} {token.text} {direct_object or ''} {child.text} {indirect_object}",
                                arguments={"direct_object": direct_object} if direct_object else {},
                                is_known=predicate != "UNKNOWN",
                            ))
        return relations
    
    @classmethod
    def _find_subject(cls, doc, token) -> Optional[str]:
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubj:pass"):
                return child.text
        return None
    
    @classmethod
    def _find_direct_object(cls, token) -> Optional[str]:
        for child in token.children:
            if child.dep_ == "obj":
                return child.text
        return None
    
    @classmethod
    def _find_pobj(cls, prep_token) -> Optional[str]:
        for child in prep_token.children:
            if child.dep_ == "pobj":
                return child.text
        return None
    
    @classmethod
    def _extract_nmod_object(cls, nmod_token) -> Optional[str]:
        for child in nmod_token.children:
            if child.dep_ == "case":
                continue
            if child.pos_ in ("NOUN", "PROPN"):
                return child.text
        return None


class CoreferenceExtractor:
    """Extrait les coréférences."""
    
    PRONOUNS = {
        "il": {"gender": "M", "number": "S", "type": "subject"},
        "elle": {"gender": "F", "number": "S", "type": "subject"},
        "ils": {"gender": "M", "number": "P", "type": "subject"},
        "elles": {"gender": "F", "number": "P", "type": "subject"},
        "le": {"gender": "M", "number": "S", "type": "object"},
        "la": {"gender": "F", "number": "S", "type": "object"},
        "les": {"gender": "M", "number": "P", "type": "object"},
        "lui": {"gender": "M", "number": "S", "type": "indirect"},
        "leur": {"gender": "M", "number": "P", "type": "indirect"},
        "y": {"gender": None, "number": None, "type": "place"},
        "en": {"gender": None, "number": None, "type": "quantity"},
    }
    
    @classmethod
    def extract(cls, doc) -> List[Coreference]:
        coreferences = []
        
        if hasattr(doc, "_.coref_clusters"):
            for cluster_id, cluster in enumerate(doc._.coref_clusters):
                mentions = list(cluster.mentions)
                if len(mentions) >= 2:
                    antecedent = mentions[0].text
                    for mention in mentions[1:]:
                        if mention.text.lower() != antecedent.lower():
                            coreferences.append(Coreference(
                                pronoun=mention.text,
                                antecedent=antecedent,
                                position_pronoun=mention.start,
                                position_antecedent=mentions[0].start,
                                confidence=0.95,
                                cluster_id=cluster_id,
                            ))
        else:
            coreferences.extend(cls._simple_coreference(doc))
        
        coreferences.extend(cls._object_pronoun_coreference(doc))
        
        return coreferences
    
    @classmethod
    def _simple_coreference(cls, doc) -> List[Coreference]:
        coreferences = []
        entities = []
        
        for token in doc:
            if token.ent_type_ == "PER":
                entities.append({"text": token.text, "position": token.i, "type": "entity"})
            elif token.pos_ == "NOUN" and token.dep_ in ("nsubj", "obj"):
                entities.append({"text": token.text, "position": token.i, "type": "noun"})
        
        for token in doc:
            if token.pos_ == "PRON" and token.text.lower() in cls.PRONOUNS:
                for ent in reversed(entities):
                    if ent["position"] < token.i:
                        distance = token.i - ent["position"]
                        confidence = 0.9 if distance <= 5 else 0.8 if distance <= 10 else 0.7
                        coreferences.append(Coreference(
                            pronoun=token.text,
                            antecedent=ent["text"],
                            position_pronoun=token.i,
                            position_antecedent=ent["position"],
                            confidence=confidence,
                        ))
                        break
        
        return coreferences
    
    @classmethod
    def _object_pronoun_coreference(cls, doc) -> List[Coreference]:
        coreferences = []
        
        for token in doc:
            if token.pos_ == "PRON" and token.text.lower() in ("le", "la", "les", "lui", "leur"):
                for verb in doc:
                    if verb.pos_ == "VERB":
                        for child in verb.children:
                            if child.dep_ == "obj":
                                coreferences.append(Coreference(
                                    pronoun=token.text,
                                    antecedent=child.text,
                                    position_pronoun=token.i,
                                    position_antecedent=child.i,
                                    confidence=0.85,
                                ))
                                return coreferences
        
        return coreferences


class AttributeExtractor:
    """Extrait les attributs des entités."""
    
    @classmethod
    def extract(cls, doc) -> List[Attribute]:
        attributes = []
        
        for token in doc:
            if token.lemma_ in ("être", "devenir", "sembler", "paraître") and token.dep_ == "ROOT":
                subject = None
                attr = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubj:pass"):
                        subject = child.text
                    if child.dep_ == "attr" and child.pos_ in ("NOUN", "PROPN", "ADJ"):
                        attr = child.text
                if subject and attr:
                    is_temporary = token.lemma_ in ("être", "sembler", "paraître")
                    attributes.append(Attribute(
                        entity=subject, attribute=attr,
                        attribute_type="state" if attr in ("fatigué", "heureux", "triste") else "quality",
                        is_temporary=is_temporary,
                        is_epithet=False,
                        source_text=f"{subject} {token.text} {attr}"
                    ))
        
        for token in doc:
            if token.dep_ == "amod":
                attributes.append(Attribute(
                    entity=token.head.text,
                    attribute=token.text,
                    attribute_type="quality",
                    confidence=0.85,
                    is_temporary=False,
                    is_epithet=True,
                    source_text=f"{token.head.text} {token.text}"
                ))
        
        for token in doc:
            if token.pos_ == "ADJ" and token.dep_ not in ("amod", "attr"):
                for child in token.head.children:
                    if child.pos_ in ("NOUN", "PROPN"):
                        attributes.append(Attribute(
                            entity=child.text,
                            attribute=token.text,
                            attribute_type="quality",
                            confidence=0.75,
                            is_temporary=True,
                            is_epithet=False,
                            source_text=f"{child.text} {token.text}"
                        ))
                        break
        
        return attributes


class EventExtractor:
    """Extrait les événements avec instrument."""
    
    @classmethod
    def extract(cls, doc) -> List[Event]:
        events = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                actor = None
                patient = None
                instrument = None
                location = None
                time = None
                
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubj:pass"):
                        actor = child.text
                    if child.dep_ == "obj":
                        patient = child.text
                    if child.dep_ == "obl":
                        for grandchild in child.children:
                            if grandchild.text.lower() in ("avec", "en utilisant", "à l'aide de"):
                                instrument = child.text
                                break
                        for grandchild in child.children:
                            if grandchild.text.lower() in ("dans", "à", "sur", "chez"):
                                location = child.text
                                break
                
                for ent in doc.ents:
                    if ent.label_ in ("DATE", "TIME"):
                        time = ent.text
                
                events.append(Event(
                    action=token.lemma_,
                    actor=actor,
                    patient=patient,
                    instrument=instrument,
                    location=location,
                    time=time,
                    source_text=token.text
                ))
        return events


class SubordinateClauseExtractor:
    """Extrait les propositions subordonnées."""
    
    SUBORDINATE_CONJUNCTIONS = {
        "parce que": "cause", "car": "cause", "puisque": "cause",
        "bien que": "concession", "quoique": "concession", "même si": "concession",
        "si": "condition", "à condition que": "condition", "pourvu que": "condition",
        "quand": "time", "lorsque": "time", "après que": "time",
        "avant que": "time", "pendant que": "time", "pour que": "purpose",
        "afin que": "purpose",
    }
    
    @classmethod
    def extract(cls, doc) -> List[SubordinateClause]:
        clauses = []
        for token in doc:
            if token.dep_ == "mark" and token.text.lower() in cls.SUBORDINATE_CONJUNCTIONS:
                relation = cls.SUBORDINATE_CONJUNCTIONS[token.text.lower()]
                sub_clause_token = token.head
                sub_clause = sub_clause_token.text
                main_clause = None
                for parent in sub_clause_token.ancestors:
                    if parent.dep_ == "ROOT":
                        main_clause = parent.text
                        break
                if main_clause and sub_clause:
                    clauses.append(SubordinateClause(
                        main_clause=main_clause,
                        sub_clause=sub_clause,
                        relation=relation,
                        source_text=token.text,
                    ))
        return clauses


class SecondaryIntentExtractor:
    """Extrait les intentions secondaires."""
    
    SECONDARY_TYPES = {
        "reminder": ["rappelle-moi", "souviens-toi", "pense à", "n'oublie pas de"],
        "scheduled_action": ["dans", "le", "à", "demain", "ce soir", "ce matin", "cet après-midi"],
        "warning": ["attention", "prudence", "danger"],
        "condition": ["si", "à condition que"],
        "suggestion": ["peut-être", "tu pourrais", "si tu veux"],
        "obligation": ["tu dois", "il faut", "absolument"],
    }
    
    @classmethod
    def extract(cls, doc, primary_intent: str, scheduled_time: dict = None) -> List[SecondaryIntent]:
        intents = []
        text_lower = doc.text.lower()
        
        for intent_type, triggers in cls.SECONDARY_TYPES.items():
            for trigger in triggers:
                if trigger in text_lower:
                    action, target = cls._extract_action_target(doc)
                    intents.append(SecondaryIntent(
                        primary_intent=primary_intent,
                        secondary_intent=intent_type,
                        trigger=trigger,
                        action=action,
                        target=target,
                        scheduled_time=scheduled_time,
                        confidence=0.85 if action else 0.7,
                    ))
        
        return intents
    
    @classmethod
    def _extract_action_target(cls, doc) -> Tuple[Optional[str], Optional[str]]:
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                action = token.lemma_
                target = None
                for child in token.children:
                    if child.dep_ == "obj":
                        target = child.text
                        break
                return action, target
        return None, None


class RelativeTenseExtractor:
    """Extrait les temps relatifs entre événements."""
    
    TENSE_MARKERS = {
        "avant": "before", "avant que": "before", "avant de": "before",
        "après": "after", "après que": "after", "après avoir": "after",
        "puis": "after", "ensuite": "after", "alors": "after",
        "pendant": "during", "pendant que": "during",
        "en même temps": "simultaneous", "alors que": "simultaneous", "tandis que": "simultaneous",
        "depuis": "since", "jusqu'à": "until",
    }
    
    @classmethod
    def extract(cls, doc) -> List[RelativeTense]:
        tenses = []
        
        for i, token in enumerate(doc):
            marker = None
            for m in cls.TENSE_MARKERS:
                if m in token.text.lower() or (i + 1 < len(doc) and f"{token.text.lower()} {doc[i+1].text.lower()}" == m):
                    marker = m
                    break
            
            if marker:
                relation = cls.TENSE_MARKERS[marker]
                before_event = cls._find_event_before(doc, token.i)
                after_event = cls._find_event_after(doc, token.i)
                
                if before_event and after_event:
                    tenses.append(RelativeTense(
                        before_event=before_event,
                        after_event=after_event,
                        relation=relation,
                        time_gap=cls._extract_time_gap(doc, token.i),
                        confidence=0.85,
                    ))
                elif before_event and relation in ("after", "since"):
                    tenses.append(RelativeTense(
                        before_event=before_event,
                        after_event=cls._find_event_main_verb(doc),
                        relation=relation,
                        confidence=0.80,
                    ))
                elif after_event and relation in ("before", "until"):
                    tenses.append(RelativeTense(
                        before_event=cls._find_event_main_verb(doc),
                        after_event=after_event,
                        relation=relation,
                        confidence=0.80,
                    ))
        
        return tenses
    
    @classmethod
    def _find_event_before(cls, doc, position) -> Optional[str]:
        for i in range(position - 1, -1, -1):
            if doc[i].pos_ == "VERB":
                return doc[i].lemma_
        return None
    
    @classmethod
    def _find_event_after(cls, doc, position) -> Optional[str]:
        for i in range(position + 1, len(doc)):
            if doc[i].pos_ == "VERB":
                return doc[i].lemma_
        return None
    
    @classmethod
    def _find_event_main_verb(cls, doc) -> Optional[str]:
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                return token.lemma_
        return None
    
    @classmethod
    def _extract_time_gap(cls, doc, position) -> Optional[str]:
        for i in range(max(0, position - 5), min(len(doc), position + 5)):
            if doc[i].like_num:
                for j in range(i + 1, min(i + 3, len(doc))):
                    if doc[j].text.lower() in ("heure", "heures", "jour", "jours", "semaine", "semaines", "mois", "an", "ans"):
                        return f"{doc[i].text} {doc[j].text}"
        return None


# ============================================================
# FONCTIONS EXISTANTES (simplifiées)
# ============================================================

def _get_dateparser():
    global _DATE_PARSER
    if _DATE_PARSER is None:
        import dateparser as _dp
        _DATE_PARSER = _dp
    return _DATE_PARSER


_DATE_PARSER = None
_TIMEXY_CHECKED = False
_TIMEXY_AVAILABLE = False

COMMON_WORDS_FR = {
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais",
    "donc", "car", "est", "sont", "était", "ai", "as", "a", "avons", "avez", "ont",
    "vais", "vas", "va", "peux", "peut", "veux", "veut", "fais", "fait", "dis", "dit",
    "pour", "par", "avec", "sans", "dans", "sur", "sous",
}


class FrenchTextCorrector:
    """Normalise le texte avant analyse NLU."""
    
    COMMON_MISTAKES: Dict[str, str] = {
        "sava": "ça va", "cetais": "c'était", "cété": "c'était",
        "jsp": "je sais pas", "pk": "pourquoi", "pcq": "parce que",
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


class FrenchVerbAnalyzer:
    """Enrichit l'analyse verbale spaCy."""
    
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


class FrenchTemporalResolver:
    """Résout les expressions temporelles françaises."""
    
    MOMENTS = {
        "midi": (12, 12, "point"), "minuit": (0, 0, "point"),
        "matin": (6, 12, "interval"), "matinée": (6, 12, "interval"),
        "après-midi": (12, 18, "interval"), "soir": (18, 22, "interval"),
        "soirée": (18, 22, "interval"), "nuit": (22, 6, "interval"),
        "journée": (8, 20, "interval"), "aube": (5, 7, "interval"),
        "crépuscule": (18, 20, "interval"),
    }
    
    MOMENT_PATTERNS = {
        "midi": ["midi", "à midi", "12h"], "minuit": ["minuit", "à minuit", "0h"],
        "matin": ["matin", "ce matin", "dans la matinée"],
        "après-midi": ["après-midi", "cet après-midi"],
        "soir": ["soir", "ce soir", "dans la soirée"],
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
    
    RELATIVE_DAYS = {"aujourd'hui": 0, "demain": 1, "après-demain": 2, "hier": -1, "avant-hier": -2}
    WEEKDAYS = {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4, "samedi": 5, "dimanche": 6}
    
    HOLIDAY_NAMES = {
        "nouvel an": "01-01", "1er janvier": "01-01", "1er mai": "05-01", "fête du travail": "05-01",
        "8 mai": "05-08", "victoire": "05-08", "14 juillet": "07-14", "fête nationale": "07-14",
        "15 août": "08-15", "assomption": "08-15", "1er novembre": "11-01", "toussaint": "11-01",
        "11 novembre": "11-11", "armistice": "11-11", "noël": "12-25", "25 décembre": "12-25",
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
            offset = (-2 if "avant-hier" in text else -1 if "hier" in text else
                      2 if "après-demain" in text else 1 if "demain" in text else 0)
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


@dataclass
class ConversationContext:
    """Résolution des ellipses et pronoms anaphoriques."""
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
        self.last_verb_lemma = intent.verb.lemma if intent.verb else None
        self.last_actor = intent.who_raw
        self.last_action = intent.action or (intent.verb.lemma if intent.verb else None)
        self.last_location = intent.where
        self.last_when_iso = intent.when.iso_start if intent.when else None
        self.timestamp = time.time()
    
    def resolve_ellipsis(self, text: str) -> str:
        if self.is_expired():
            return text
        tl = text.lower().strip()
        if tl in ("oui", "ouais", "ok", "d'accord", "okay", "bien sûr"):
            if self.last_intent_type == "query_narrative" and self.last_verb_lemma:
                return f"oui, {self.last_verb_lemma}"
        if tl in ("encore", "recommence", "refais", "re"):
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
    """Héritage de slots (who/when/where) entre fragments."""
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
            else:
                setattr(self, k, v)
        self.last_update = time.monotonic()
    
    def flush_pending(self) -> List[str]:
        frags, self.pending_fragments = list(self.pending_fragments), []
        return frags


def _extract_all_slots(doc):
    who_person = None
    who_raw = None
    with_who_set = set()
    where = None
    entities = []
    seen_spans = set()
    subjects = set()
    
    for ent in doc.ents:
        etype = NER_TYPE_MAP.get(ent.label_)
        if not etype:
            continue
        key = (ent.start, ent.end)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        raw = ent.text.strip()
        value = None
        unit = None
        if etype in ("number", "ordinal", "quantity", "percent", "money",
                     "time_ref", "date_ref", "duration"):
            nm = re.search(r"(\d+(?:[.,]\d+)?)", raw.replace("\u202f", "").replace(" ", ""))
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
        
        if not where and token.text.lower() in ("dans", "au", "en", "à", "chez") and token.dep_ == "case":
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


def _extract_verb(doc) -> Optional[VerbAnalysis]:
    for token in doc:
        if token.pos_ != "VERB":
            continue
        if token.dep_ not in ("ROOT", "acl", "relcl", "advcl", "xcomp", "ccomp"):
            continue
        morph = token.morph
        mood_map = {"Cnd": "conditional", "Imp": "imperative", "Sub": "subjunctive", "Ind": "indicative"}
        mood_raw = morph.get("Mood")
        mood: MoodType = "indicative"
        if mood_raw:
            mood = mood_map.get(mood_raw[0], "indicative")
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
                    c.pos_ == "AUX" and c.morph.get("Tense") and "Pres" in c.morph.get("Tense")
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
            pm = {("1", "Sing"): "1st_sg", ("2", "Sing"): "2nd_sg", ("3", "Sing"): "3rd_sg",
                  ("1", "Plur"): "1st_pl", ("2", "Plur"): "2nd_pl", ("3", "Plur"): "3rd_pl"}
            person = pm.get((person_raw[0], number_raw[0]), "unknown")
            number = "sg" if number_raw[0] == "Sing" else "pl" if number_raw[0] == "Plur" else "unknown"
        
        neg = [c for c in token.children if c.dep_ == "advmod" and c.lemma_ in ("ne", "pas", "plus", "jamais", "rien", "guère")]
        left_neg = any(t.dep_ == "advmod" and t.lemma_ in ("ne", "n") for t in token.lefts)
        polarity: PolarityType = "negative" if (neg or left_neg) else "positive"
        
        modal: ModalType = None
        for aux in token.children:
            if aux.pos_ == "AUX":
                modal = MODAL_MARKERS.get(aux.lemma_.lower())
                if modal:
                    break
        if not modal and mood == "conditional":
            modal = "wish"
        
        concept = SEMANTIC_MAP.get(token.lemma_.lower())
        if not concept:
            for child in token.children:
                if child.pos_ == "VERB":
                    concept = SEMANTIC_MAP.get(child.lemma_.lower())
                    if concept:
                        break
        
        return VerbAnalysis(
            lemma=token.lemma_, tense=raw_tense,
            scope=TENSE_TO_SCOPE.get(raw_tense, "UNKNOWN"),
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
    has_verb = any(t.pos_ == "VERB" and t.dep_ in ("ROOT", "acl", "relcl", "advcl", "xcomp", "ccomp") for t in doc)
    return not has_verb and any([slots.get("who"), slots.get("when"), slots.get("where"), slots.get("with_who")])


def _analyze_register(doc, text: str) -> RegisterAnalysis:
    toks = [t.text.lower().rstrip("'") for t in doc]
    votes = {s: 0.0 for s in REGISTER_LEXICON}
    marks = {s: [] for s in REGISTER_LEXICON}
    
    politeness = any(p in text.lower() for p in ("stp", "svp", "s'il te plaît", "s'il vous plaît", "merci de"))
    
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
    if politeness:
        all_markers.append("polite")
    
    neg = ("complete" if has_pas and has_ne else "incomplete" if has_pas else "absente")
    return RegisterAnalysis(style=best, confidence=conf, markers=all_markers[:8], tu_vous=tu_vous, negation=neg, politeness=politeness)


def split_clauses(text: str) -> List[str]:
    parts = _CLAUSE_COORD.split(text)
    if len(parts) == 1:
        return [text.strip()]
    valid = []
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


def _extract_temporal_v4(doc, tense: str = "unknown", resolver=None) -> Optional[TemporalSpan]:
    resolver = resolver or _get_temporal_resolver()
    prefer = "future" if tense in ("future", "conditional") else "past"
    dp = _get_dateparser() if DATEPARSER_AVAILABLE else None
    
    interval = _extract_interval_v4(doc.text, prefer)
    if interval:
        return interval
    
    ref = datetime.datetime.now()
    resolved = resolver.resolve(doc.text, ref)
    if resolved and resolved.get("source", "none") not in ("none", ""):
        span = resolver.to_temporal_span(resolved, tense)
        if span:
            return span
    
    if _check_timexy() and dp:
        TIMEX3 = {"DATE": "DATE", "TIME": "TIME", "DURATION": "DURATION", "SET": "SET"}
        cfg = {"RETURN_AS_TIMEZONE_AWARE": True, "TIMEZONE": "Europe/Paris", "PREFER_DATES_FROM": prefer}
        for ent in doc.ents:
            ttype = getattr(getattr(ent, "_", None), "timex_type", None)
            if ttype:
                tv = getattr(getattr(ent, "_", None), "timex_value", None)
                dt = None
                if tv:
                    try:
                        dt = datetime.datetime.fromisoformat(tv)
                    except (ValueError, TypeError):
                        pass
                if not dt:
                    dt = dp.parse(ent.text, languages=[LANGUAGE], settings=cfg)
                if dt:
                    iso = dt.isoformat()
                    return TemporalSpan(raw=ent.text, iso_start=iso, iso_end=iso,
                                        timex_type=TIMEX3.get(ttype, "DATE"), source="timexy")
    
    if dp:
        cfg = {"RETURN_AS_TIMEZONE_AWARE": True, "TIMEZONE": "Europe/Paris", "PREFER_DATES_FROM": prefer}
        for ent in doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                dt = dp.parse(ent.text, languages=[LANGUAGE], settings=cfg)
                if dt:
                    iso = dt.isoformat()
                    return TemporalSpan(raw=ent.text, iso_start=iso, iso_end=iso,
                                        timex_type="TIME" if ent.label_ == "TIME" else "DATE",
                                        source="spacy_ner")
        
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
    dp = _get_dateparser()
    cfg = {"RETURN_AS_TIMEZONE_AWARE": True, "TIMEZONE": "Europe/Paris", "PREFER_DATES_FROM": prefer}
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
        val = float(m.group(1).replace(",", "."))
        u_raw = m.group(2).lower()
        unit = ("h" if u_raw.startswith("h") else "min" if u_raw.startswith("min") else
                "s" if u_raw.startswith("s") else "day" if u_raw.startswith("j") else
                "week" if u_raw.startswith("sem") else "month")
        results.append({"type": "duration", "value": val, "unit": unit, "raw": m.group(0)})
    if DATEPARSER_AVAILABLE:
        dp = _get_dateparser()
        for m in re.finditer(
            r"jusqu['\u2019]?\s*[aà]\s+(\d{1,2}h\d{0,2}|\d{1,2}:\d{2})",
            text, re.IGNORECASE
        ):
            dt = dp.parse(m.group(1), languages=[LANGUAGE],
                          settings={"RETURN_AS_TIMEZONE_AWARE": True, "TIMEZONE": "Europe/Paris"})
            if dt:
                results.append({"type": "until", "iso": dt.isoformat(), "raw": m.group(0)})
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
    dur_val = None
    for d in durations:
        if d["type"] == "until":
            try:
                end_dt = datetime.datetime.fromisoformat(d["iso"])
                until_raw = d["raw"]
            except (ValueError, KeyError):
                pass
        elif d["type"] == "duration":
            v, u = d.get("value", 0), d.get("unit", "h")
            delta = (datetime.timedelta(hours=v) if u == "h" else
                     datetime.timedelta(minutes=v) if u == "min" else
                     datetime.timedelta(seconds=v) if u == "s" else
                     datetime.timedelta(days=v) if u == "day" else
                     datetime.timedelta(weeks=v) if u == "week" else
                     datetime.timedelta(days=v * 30))
            end_dt = start_dt + delta
            dur_raw = d["raw"]
            dur_unit = u
            dur_val = v
    from dataclasses import replace
    return replace(span, iso_end=end_dt.isoformat(),
                   duration_raw=dur_raw, duration_unit=dur_unit,
                   duration_value=dur_val, until_raw=until_raw)


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


_CHIT_CHAT_WORDS = frozenset({
    "bonjour", "salut", "coucou", "bonsoir", "bye", "merci", "svp", "bravo",
    "félicitations", "chapeau", "super", "génial",
})

_CHIT_CHAT_PHRASES = (
    "bonne nuit", "au revoir", "s'il vous plaît", "s'il te plaît",
    "ça va", "comment vas", "comment allez",
)

_INTERROGATIVE_WORDS = frozenset({
    "quel", "quelle", "quels", "quelles", "quoi", "comment", "pourquoi",
    "quand", "où", "qui", "combien", "lequel", "laquelle", "lesquels", "lesquelles",
    "est-ce", "est ce",
})

_ACTION_DEVICE_WORDS = frozenset({
    "allume", "allumer", "éteins", "éteindre", "ouvre", "ouvrir", "ferme", "fermer",
    "active", "activer", "désactive", "désactiver", "monte", "monter", "baisse", "baisser",
    "règle", "régler", "mets", "mettre", "configure", "configurer", "programme", "programmer",
    "démarre", "démarrer", "stoppe", "stopper", "coupe", "couper",
})


def _clf(intent: IntentType, confidence: float) -> dict:
    return {"intent": intent, "confidence": round(confidence, 3),
            "uncertain": confidence < CONFIDENCE_THRESHOLD, "scores": {intent: round(confidence, 3)}}


def classify_intent(verb: Optional[VerbAnalysis], doc, text: str) -> dict:
    tl = text.lower().strip()
    words = set(tl.split())
    
    if verb and verb.mood == "imperative":
        if (verb.lemma in ACTION_VERBS or words & _ACTION_DEVICE_WORDS or
                any(noun in tl for noun in DEVICE_NOUNS)):
            return _clf("action_device", 0.95)
        return _clf("action_device", 0.80)
    
    if (words & _ACTION_DEVICE_WORDS and any(noun in tl for noun in DEVICE_NOUNS)):
        return _clf("action_device", 0.88)
    
    if words & _CHIT_CHAT_WORDS or any(p in tl for p in _CHIT_CHAT_PHRASES):
        return _clf("chit_chat", 0.93)
    
    if (verb and verb.concept == "SOCIAL_ACT" and len(tl.split()) <= 7):
        return _clf("chit_chat", 0.88)
    
    has_question_mark = "?" in text
    has_interrogative = bool(words & _INTERROGATIVE_WORDS)
    
    if has_question_mark or has_interrogative:
        scope = verb.scope if verb else "UNKNOWN"
        if scope == "PAST":
            return _clf("query_narrative", 0.90)
        if scope == "FUTURE":
            return _clf("query_intention", 0.88)
        if verb and verb.tense == "past":
            return _clf("query_narrative", 0.85)
        return _clf("query_state", 0.85)
    
    if verb and verb.person in ("1st_sg", "1st_pl"):
        return _clf("information_input", 0.88)
    
    if verb and verb.tense in ("past", "present", "future"):
        return _clf("information_input", 0.78)
    
    return _clf("chit_chat", 0.55)


class IntentClassifier:
    def classify(self, text: str, verb: Optional[VerbAnalysis] = None, doc=None) -> dict:
        return classify_intent(verb, doc, text)
    
    def load(self):
        logger.info("IntentClassifier déterministe prêt (sans modèle).")


class SyntacticTreeExtractor:
    _USER = {"je", "j'", "moi", "me", "m'"}
    _ASE = {"tu", "toi", "te", "t'"}
    
    @classmethod
    def _role(cls, token) -> Optional[RoleType]:
        tl = token.text.lower()
        if tl in cls._USER:
            return "user"
        if tl in cls._ASE:
            return "ase"
        if token.ent_type_ == "PER":
            return "other"
        return None
    
    @classmethod
    def extract(cls, doc) -> SyntacticTree:
        tree = SyntacticTree(phrase=doc.text)
        nodes = {}
        for token in doc:
            node = TokenNode(
                text=token.text, lemma=token.lemma_, pos=token.pos_, dep=token.dep_,
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
        return {"text": self.text, "lemma": self.lemma, "pos": self.pos,
                "dep": self.dep, "role": self.role,
                "children": [c.to_dict() for c in self.children]}


class PipelineMetrics:
    def __init__(self):
        self.total = 0
        self.total_ms = 0.0
        self.errors = 0
        self.intent_dist = defaultdict(int)
        self.corrections = defaultdict(int)
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
            "top_corrections": sorted(self.corrections.items(), key=lambda x: x[1], reverse=True)[:10],
            "uptime_s": round(time.time() - self.start_time, 1),
        }


class IntentPipeline:
    def __init__(self, q_in: Queue, q_out: Queue, debug: bool = False):
        self._q_in = q_in
        self._q_out = q_out
        self._debug = debug
        self._nlp = None
        self._classifier = IntentClassifier()
        self._resolver = FrenchTemporalResolver()
        self._ctx = ConversationContext()
        self._frame = ConversationFrame()
        self._metrics = PipelineMetrics()
        self._thread = None
        self._running = False
        self._loaded = False
        self._lock = threading.Lock()
        self._enable_corrector = True
        self._corrections_stats = defaultdict(int)
    
    def enable_corrector(self, enable: bool = True):
        self._enable_corrector = enable
    
    def load(self):
        self._load_spacy()
        self._classifier.load()
        self._loaded = True
        logger.info("IntentPipeline v4.1 final prêt.")
    
    def _load_spacy(self):
        if not SPACY_AVAILABLE:
            raise ImportError("spacy requis")
        t0 = time.time()
        self._nlp = spacy.load(SPACY_MODEL)
        if _check_timexy():
            try:
                cfg = {"per": True, "dur": True, "set": True, "num": True, "lang": "fr"}
                if "timexy" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("timexy", config=cfg, last=True)
                logger.info(f"spaCy + timexy chargés en {time.time() - t0:.2f}s")
            except Exception:
                logger.info(f"spaCy chargé en {time.time() - t0:.2f}s")
        else:
            logger.info(f"spaCy chargé en {time.time() - t0:.2f}s")
    
    def start(self):
        if not self._loaded:
            raise RuntimeError("Appelle load() avant start().")
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="intent-pipeline-v4", daemon=True)
        self._thread.start()
        logger.info("Pipeline démarré")
    
    def stop(self):
        self._running = False
        self._q_in.put(None)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Pipeline arrêté")
    
    def _worker(self):
        while self._running:
            try:
                text = self._q_in.get(timeout=1.0)
            except Empty:
                if self._frame.is_expired():
                    self._frame.reset()
                if self._ctx.is_expired():
                    self._ctx = ConversationContext()
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
    
    def _preprocess(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        text = re.sub(r'[^\w\s\u00C0-\u00FF?.!,;:\'"()-]', '', text)
        if self._enable_corrector:
            corrected, corrections = FrenchTextCorrector.correct(text)
            for orig, fix in corrections:
                self._corrections_stats[f"{orig}→{fix}"] += 1
            return corrected, corrections
        return text, []
    
    def _process(self, text: str) -> List[Intent]:
        if self._frame.is_expired():
            self._frame.reset()
        if self._ctx.is_expired():
            self._ctx = ConversationContext()
        
        text = self._ctx.resolve_ellipsis(text)
        text = self._ctx.resolve_pronouns(text)
        
        text_clean, corrections = self._preprocess(text)
        clauses = split_clauses(text_clean)
        
        if len(clauses) == 1:
            doc = self._nlp(text_clean)
            verb = _extract_verb(doc)
            who_p, who_r, with_who, where, _ = _extract_all_slots(doc)
            temporal = _extract_temporal_v4(doc, verb.tense if verb else "unknown", self._resolver)
            slots = {"who": who_p, "who_raw": who_r, "with_who": with_who,
                     "where": where, "when": temporal, "what": _extract_what(doc)}
            if _is_fragment(doc, slots):
                self._frame.update(fragment=text_clean, **{k: v for k, v in slots.items() if v is not None})
                logger.debug(f"Fragment accumulé : '{text_clean}'")
                return []
        
        assembled = self._frame.flush_pending()
        results = []
        for idx, clause in enumerate(clauses):
            intent = self._process_clause(
                clause, clause_index=idx,
                assembled_from=assembled if idx == 0 else [],
                corrections=corrections if idx == 0 else [],
            )
            if intent:
                results.append(intent)
        return results
    
    def _process_clause(self, text: str, clause_index: int = 0,
                        assembled_from: List[str] = None, corrections: List[Tuple] = None) -> Optional[Intent]:
        t0 = time.time()
        assembled_from = assembled_from or []
        corrections = corrections or []
        
        doc = self._nlp(text)
        verb = _extract_verb(doc)
        tense = verb.tense if verb else "unknown"
        
        syntax_tree = SyntacticTreeExtractor.extract(doc)
        
        who_p, who_r, with_who, where, entities = _extract_all_slots(doc)
        temporal = _extract_temporal_v4(doc, tense, self._resolver)
        what = _extract_what(doc)
        
        # Nouveaux extracteurs
        relations = RelationExtractor.extract(doc)
        triplets = [(r.subject, r.predicate, r.object) for r in relations if r.subject and r.object]
        coreferences = CoreferenceExtractor.extract(doc)
        attributes = AttributeExtractor.extract(doc)
        events = EventExtractor.extract(doc)
        subordinate_clauses = SubordinateClauseExtractor.extract(doc)
        relative_tenses = RelativeTenseExtractor.extract(doc)
        
        # Extraction du temps programmé pour les actions
        scheduled_time = None
        if verb and verb.mood == "imperative":
            scheduled_time = ScheduledTimeExtractor.extract(text)
        
        # Quantifieurs
        quantifiers = []
        for token in doc:
            if token.text.lower() in ("tous", "toutes", "chaque", "aucun", "aucune", "quelques"):
                for child in token.children:
                    if child.pos_ in ("NOUN", "PROPN") and child.dep_ == "det":
                        qtype = "universal" if token.text.lower() in ("tous", "chaque") else "existential"
                        quantifiers.append(Quantifier(
                            entity=child.text, quantifier=token.text.lower(),
                            quantifier_type=qtype, confidence=0.85,
                        ))
        
        # Négations
        negations = []
        for token in doc:
            if token.lemma_ in ("ne", "pas", "plus", "jamais") and token.dep_ == "advmod":
                for child in token.head.children:
                    if child.dep_ in ("nsubj", "obj", "attr"):
                        negations.append(Negation(
                            negated_element=child.text, negation_word=token.text,
                            scope=token.head.text, confidence=0.9,
                        ))
        
        # Comparaisons
        comparisons = []
        for token in doc:
            if token.text.lower() in ("plus", "moins", "aussi"):
                subject = None
                attribute = None
                object_ = None
                for child in token.head.children:
                    if child.dep_ in ("nsubj", "nsubj:pass"):
                        subject = child.text
                    if child.dep_ == "attr" and child.pos_ == "ADJ":
                        attribute = child.text
                for child in token.children:
                    if child.text.lower() == "que":
                        for grandchild in child.children:
                            if grandchild.pos_ in ("NOUN", "PROPN"):
                                object_ = grandchild.text
                                break
                if subject and attribute and object_:
                    degree = "superiority" if token.text.lower() == "plus" else "inferiority" if token.text.lower() == "moins" else "equality"
                    comparisons.append(Comparison(
                        subject=subject, comparator=token.text.lower(),
                        attribute=attribute, object=object_, degree=degree, confidence=0.85,
                    ))
        
        # Rôles
        roles = []
        for token in doc:
            if token.lemma_ in ("être", "devenir") and token.dep_ == "ROOT":
                subject = None
                role = None
                context = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubj:pass"):
                        subject = child.text
                    if child.dep_ == "attr":
                        role = child.text
                        for grandchild in child.children:
                            if grandchild.dep_ == "nmod":
                                for gg in grandchild.children:
                                    if gg.dep_ == "case":
                                        context = gg.text
                                    elif gg.pos_ in ("NOUN", "PROPN"):
                                        context = f"{context or ''} {gg.text}" if context else gg.text
                if subject and role:
                    roles.append(Role(entity=subject, role=role, context=context, confidence=0.85))
        
        # Intentions secondaires
        primary_intent = classify_intent(verb, doc, text)["intent"]
        secondary_intents = SecondaryIntentExtractor.extract(doc, primary_intent, 
                                                             scheduled_time.to_dict() if scheduled_time else None)
        
        # Ensemble des entités
        entities_set = set()
        for ent in doc.ents:
            if ent.label_ == "PER":
                entities_set.add(ent.text)
        for token in doc:
            if token.pos_ == "NOUN" and token.dep_ == "nsubj":
                entities_set.add(token.text)
        
        # Durées et temps
        durations = _parse_durations(text) or self._frame.durations
        if temporal and durations:
            temporal = _merge_time_duration(temporal, durations)
        elif not temporal and self._frame.when and durations:
            temporal = _merge_time_duration(self._frame.when, durations)
        
        # Si on a un temps programmé, l'ajouter au TemporalSpan
        if scheduled_time and temporal:
            temporal.scheduled = scheduled_time
        
        actions = _extract_actions(doc)
        cur_subjects = [s for a in actions for s in a.get("subjects", [])]
        if not cur_subjects and self._frame.subjects:
            for a in actions:
                if not a.get("subjects"):
                    a["subjects"] = self._frame.subjects
        
        merged = {
            "who": who_p or self._frame.who,
            "who_raw": who_r or self._frame.who_raw,
            "with_who": with_who or self._frame.with_who,
            "when": temporal or self._frame.when,
            "where": where or self._frame.where,
            "what": what or self._frame.what,
        }
        
        if clause_index == 0:
            self._frame.update(who=who_p, who_raw=who_r, with_who=with_who,
                               when=temporal, where=where,
                               subjects=cur_subjects or None,
                               durations=durations or None)
        
        reg = _analyze_register(doc, text)
        if reg.style != "neutre":
            self._frame.update(register=reg.style)
        
        clf = self._classifier.classify(text, verb=verb, doc=doc)
        action, target = _extract_device(doc) if clf["intent"] == "action_device" else (None, None)
        
        memory_hint = None
        if clf["intent"] == "information_input":
            memory_hint = {
                "subject": _infer_info_subject(doc),
                "salience": _compute_salience(doc, entities, verb),
                "entities": entities,
                "raw_info": text,
            }
        
        processing_ms = (time.time() - t0) * 1000
        
        if self._debug:
            print(f"\n  [{clause_index}] '{text}'")
            print(f"    verb={verb.lemma + '/' + verb.tense if verb else 'none'}")
            print(f"    intent={clf['intent']} conf={clf['confidence']:.2f}")
            print(f"    relations: {len(relations)}")
            print(f"    coreferences: {len(coreferences)}")
            print(f"    attributes: {len(attributes)}")
            print(f"    events: {len(events)}")
            if scheduled_time:
                print(f"    scheduled: {scheduled_time.human}")
        
        logger.info(
            f"  [{clause_index}] {clf['intent']} ({clf['confidence']:.2f}) "
            f"verb={verb.lemma + '/' + verb.tense if verb else 'none'} "
            f"who={merged['who']}({merged['who_raw']}) "
            f"where={merged['where']} {processing_ms:.0f}ms"
        )
        
        return Intent(
            text=text, intent=clf["intent"], confidence=clf["confidence"],
            uncertain=clf["uncertain"], scores=clf["scores"],
            verb=verb, who=merged["who"], who_raw=merged["who_raw"],
            with_who=merged["with_who"], when=merged["when"],
            where=merged["where"], what=merged["what"],
            action=action, target=target, actions=actions,
            entities=entities, memory_hint=memory_hint,
            register=asdict(reg), assembled_from=assembled_from,
            syntax_tree=syntax_tree, corrections=corrections,
            clause_index=clause_index, processing_ms=processing_ms, ts=time.time(),
            relations=relations, triplets=triplets,
            coreferences=coreferences, attributes=attributes,
            events=events, quantifiers=quantifiers, negations=negations,
            modalities=[], comparisons=comparisons, roles=roles,
            temporal_sequences=[], subordinate_clauses=subordinate_clauses,
            secondary_intents=secondary_intents, relative_tenses=relative_tenses,
            scheduled_time=scheduled_time,
            entities_set=entities_set, facts=set(),
        )
    
    def get_metrics(self) -> dict:
        return {
            "pipeline": self._metrics.summary(),
            "temporal_cache": self._resolver.get_stats(),
            "corrections_top": dict(sorted(self._corrections_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
    
    def benchmark(self, n: int = 20) -> dict:
        samples = ["comment tu vas ?", "t'étais où hier soir ?",
                   "allume la lumière", "je jardinerai demain matin pendant 2h",
                   "rappelle-moi d'éteindre la lumière dans 20 minutes"]
        times = []
        for i in range(n):
            t0 = time.time()
            self._process(samples[i % len(samples)])
            times.append((time.time() - t0) * 1000)
        return {"n": n, "avg_ms": round(np.mean(times), 1),
                "min_ms": round(np.min(times), 1),
                "max_ms": round(np.max(times), 1),
                "p95_ms": round(np.percentile(times, 95), 1)}


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


class EventGraph:
    def __init__(self):
        self._lock = threading.RLock()
        self.entities: Dict[str, EpisodicEntity] = {}
        self.events: List[EpisodicEvent] = []
        self._counter = 0
    
    def add_event(self, action: str, agent: List[str] = None, objects: List[str] = None,
                  concept: str = None, scope: str = "UNKNOWN", salience: float = 0.0,
                  trigger_id: int = None, intent_type: str = "unknown", raw_text: str = "") -> EpisodicEvent:
        with self._lock:
            self._counter += 1
            ev = EpisodicEvent(
                id=self._counter, action=action, agent=agent or [],
                objects=objects or [], location=None, concept=concept,
                start_time=None, end_time=None, scope=scope,
                salience=salience, trigger_id=trigger_id,
                intent_type=intent_type, raw_text=raw_text, ts=time.time(),
            )
            self.events.append(ev)
            return ev
    
    def summary(self) -> dict:
        with self._lock:
            return {"total_events": len(self.events), "total_entities": len(self.entities)}


class EventGraphConsumer(threading.Thread):
    EVENT_INTENTS = {"information_input", "action_device", "chit_chat", "reminder", "scheduled_action"}
    
    def __init__(self, q_in: Queue, graph: EventGraph):
        super().__init__(name="eventgraph-consumer", daemon=True)
        self._q_in = q_in
        self._graph = graph
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
        verb = intent.get("verb") or {}
        scope = verb.get("scope", "UNKNOWN")
        salience = (intent.get("memory_hint") or {}).get("salience", 0.3)
        raw_text = intent.get("text", "")
        actions = intent.get("actions") or [{"verb": intent.get("action", "unknown"),
                                             "subjects": [], "objects": [intent.get("target")]}]
        self._graph.add_event(
            action=actions[0].get("verb", "unknown"),
            agent=actions[0].get("subjects") or [],
            objects=actions[0].get("objects") or [],
            concept=verb.get("concept"),
            scope=scope, salience=salience, intent_type=intent.get("intent", "unknown"), raw_text=raw_text,
        )


def run_tests():
    print("\n" + "=" * 68)
    print("  TESTS IntentPipeline v4.1 (version finale)")
    print("=" * 68)
    passed = failed = 0
    
    def check(label, cond, detail=""):
        nonlocal passed, failed
        sym = "✅" if cond else "❌"
        print(f"  {sym} {label}" + (f"  ({detail})" if detail else ""))
        if cond:
            passed += 1
        else:
            failed += 1
    
    print("\n  [FrenchTextCorrector]")
    result, _ = FrenchTextCorrector.correct("jsp pk")
    check("corrector 'jsp pk'", "pas" in result or "sais" in result)
    
    print("\n  [ConversationContext]")
    ctx = ConversationContext()
    ctx.last_action = "jardiner"
    check("ellipsis 'encore'", ctx.resolve_ellipsis("encore") == "refais jardiner")
    
    print("\n  [split_clauses]")
    parts = split_clauses("je jardinerai et je lirai")
    check("split coordonnée", len(parts) == 2, f"obtenu {len(parts)}")
    
    print("\n  [FrenchTemporalResolver]")
    resolver = FrenchTemporalResolver()
    ref = datetime.datetime(2025, 6, 15, 10, 0)
    r = resolver.resolve("demain", ref)
    check("temporal 'demain'", r is not None and r.get("timex_type") == "DATE")
    
    print("\n  [ScheduledTimeExtractor]")
    scheduled = ScheduledTimeExtractor.extract("dans 20 minutes")
    check("scheduled 'dans 20 minutes'", scheduled is not None)
    if scheduled:
        check("scheduled has iso", scheduled.iso is not None)
    
    scheduled = ScheduledTimeExtractor.extract("à 20h30")
    check("scheduled 'à 20h30'", scheduled is not None)
    
    scheduled = ScheduledTimeExtractor.extract("le 25 décembre à 20h")
    check("scheduled 'le 25 décembre'", scheduled is not None)
    
    scheduled = ScheduledTimeExtractor.extract("ce soir")
    check("scheduled 'ce soir'", scheduled is not None)
    
    print("\n  [RelationExtractor]")
    try:
        import spacy
        nlp = spacy.load(SPACY_MODEL)
        doc = nlp("le fils de Marie")
        relations = RelationExtractor.extract(doc)
        check("relation 'fils de Marie'", len(relations) > 0)
    except Exception as e:
        print(f"  ⚠️ spaCy non disponible: {e}")
    
    print("\n" + "=" * 68)
    print(f"  Résultats: {passed} OK, {failed} ECHEC")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntentPipeline v4.1 — NLU français complet")
    parser.add_argument("--test", action="store_true", help="Exécute les tests")
    parser.add_argument("--predict", type=str, help="Prédit une phrase")
    parser.add_argument("--interactive", action="store_true", help="Mode interactif")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    parser.add_argument("--no-corrector", action="store_true", help="Désactive le correcteur")
    args = parser.parse_args()
    
    if args.test:
        run_tests()
    
    elif args.predict:
        q_in, q_out = Queue(), Queue()
        pipeline = IntentPipeline(q_in, q_out, debug=args.debug)
        if args.no_corrector:
            pipeline.enable_corrector(False)
        pipeline.load()
        pipeline.start()
        q_in.put(args.predict)
        try:
            result = q_out.get(timeout=10.0)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Empty:
            print("Timeout")
        pipeline.stop()
    
    elif args.interactive:
        print("Mode interactif. Tapez 'quit' pour quitter.")
        q_in, q_out = Queue(), Queue()
        pipeline = IntentPipeline(q_in, q_out, debug=args.debug)
        if args.no_corrector:
            pipeline.enable_corrector(False)
        pipeline.load()
        pipeline.start()
        
        try:
            while True:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                q_in.put(user_input)
                try:
                    result = q_out.get(timeout=10.0)
                    if result:
                        intent = result[0]
                        print(f"\nIntent: {intent['intent']} ({intent['confidence']:.2f})")
                        if intent.get('scheduled_time'):
                            st = intent['scheduled_time']
                            print(f"  Programmé: {st.get('human', st.get('raw'))}")
                        if intent.get('relations'):
                            print(f"  Relations: {len(intent['relations'])}")
                        if intent.get('events'):
                            print(f"  Événements: {len(intent['events'])}")
                except Empty:
                    print("Timeout")
        except KeyboardInterrupt:
            print("\nAu revoir!")
        finally:
            pipeline.stop()
    
    else:
        parser.print_help()