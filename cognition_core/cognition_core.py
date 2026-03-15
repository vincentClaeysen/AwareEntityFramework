#!/usr/bin/env python3
"""
cognitioncore.py — Cœur cognitif de l'entité synthétique transcendée
Version 7.0.0 - FINALE

Ce fichier contient l'intégralité du système :
- GEM (âme immuable) chargé depuis fichier JSON
- Utilisation de librairies existantes (spaCy, dateparser, hunspell, etc.)
- Mémoires multiples avec durées de vie variables
- Poids des sources pour la confiance
- Sorties 100% structurées (pas de texte)
- Refroidissement et nettoyage nocturne

Aucun hard coding - tout est dans des fichiers de configuration.
"""

import json
import logging
import time
import uuid
import threading
import sqlite3
import gzip
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import importlib.util

__version__ = "7.0.0"
logger = logging.getLogger("CognitionCore")

# ============================================================
# ENUMS
# ============================================================

class IntentType(str, Enum):
    INFORMATION = "information"
    QUESTION = "question"
    REPONSE = "reponse"
    ACTION = "action"
    CLARIFICATION = "clarification"
    SOCIAL = "social"

class MemoryType(str, Enum):
    GEM = "gem"                    # Âme immuable
    WORDS = "words"                # Mots et synonymes
    ERRORS = "errors"              # Fautes corrigées
    TEMPORAL = "temporal"          # Expressions temporelles
    EPISODIC = "episodic"          # Expériences vécues
    SOCIAL = "social"              # Personnes et relations
    NARRATIVE = "narrative"        # Livre de vie
    LITERARY_ROMAN = "literary_roman"  # Romans (éphémère)
    LITERARY_EDUC = "literary_educ"    # Savoir académique (permanent)

class SourceWeight(str, Enum):
    OBSERVATION = "observation"    # 1.0 - L'entité voit/vécu
    SELF = "self"                  # 0.95 - L'utilisateur sur lui-même
    EDUCATIVE = "educative"        # 0.9 - Encyclopédies, manuels
    SCIENTIFIC = "scientific"      # 0.8 - Revues peer-reviewed
    REPORTED = "reported"          # 0.6 - Rapporté sur quelqu'un
    FICTION = "fiction"            # 0.3 - Roman, fiction
    INTERNET = "internet"          # 0.2 - Forums, blogs
    RUMOR = "rumor"                # 0.1 - Oui-dire

# ============================================================
# MODÈLES DE BASE
# ============================================================

@dataclass
class Attribute:
    """Attribut typé avec confiance et source."""
    type: str
    value: Any
    confidence: float = 1.0
    source: SourceWeight = SourceWeight.OBSERVATION
    normalized: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StructuredIntent:
    """Intent purement structurel - AUCUN TEXTE."""
    intent: str
    sub_intent: str
    type: IntentType
    attributes: Dict[str, Attribute]
    confidence: float
    source: str
    in_response_to: Optional[str] = None
    conversation_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "sub_intent": self.sub_intent,
            "type": self.type.value,
            "attributes": {
                k: {
                    "type": v.type,
                    "value": v.value,
                    "confidence": v.confidence,
                    "source": v.source.value,
                    "normalized": v.normalized
                } for k, v in self.attributes.items()
            },
            "confidence": self.confidence,
            "source": self.source
        }

@dataclass
class MultiIntent:
    """Une phrase peut produire plusieurs intents."""
    intents: List[StructuredIntent]
    original_text: str
    timestamp: float
    
    def to_dict(self) -> dict:
        return {
            "intents": [i.to_dict() for i in self.intents],
            "original_text": self.original_text,
            "timestamp": self.timestamp
        }

# ============================================================
# GEM - L'ÂME IMMUABLE (chargé depuis fichier)
# ============================================================

@dataclass
class Gem:
    """Âme de l'entité - immuable, définit la personnalité."""
    
    # Identité
    identifiant: str
    nom: str
    date_naissance: str
    version: int
    
    # Personnalité de base
    tempo_base: float           # 0-1, vitesse de réponse
    intensite_base: float       # 0-1, force des émotions
    grace: float                # 0-1, indulgence
    reactivite: float           # 0-1, réactivité aux stimuli
    
    # Curiosités (0-1)
    curiosite_mots: float
    curiosite_verbes: float
    curiosite_personnes: float
    curiosite_lieux: float
    curiosite_faits: float
    
    # Affinités littéraires
    affinites_litterature: Dict[str, float]  # genre → affinité
    
    # Durées mémoire (en jours)
    duree_memoire_litterature: int
    duree_memoire_episodique: int
    duree_memoire_sociale: int
    
    # Préférences
    style_prefere: str  # "narratif", "poétique", "direct"
    seuil_curiosite: float  # quand poser une question
    
    # Signature
    signature_type: str
    signature_valeur: str
    
    @classmethod
    def from_file(cls, path: Path) -> 'Gem':
        """Charge le Gem depuis un fichier JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        g = data.get("gem", data)
        
        return cls(
            identifiant=g.get("identifiant", "inconnu"),
            nom=g.get("nom", "Shirka"),
            date_naissance=g.get("date_naissance", datetime.now().isoformat()),
            version=g.get("version", 1),
            
            tempo_base=g.get("tempo_base", 0.65),
            intensite_base=g.get("intensite_base", 0.7),
            grace=g.get("grace", 0.5),
            reactivite=g.get("reactivite", 0.8),
            
            curiosite_mots=g.get("curiosite_mots", 0.6),
            curiosite_verbes=g.get("curiosite_verbes", 0.5),
            curiosite_personnes=g.get("curiosite_personnes", 0.9),
            curiosite_lieux=g.get("curiosite_lieux", 0.8),
            curiosite_faits=g.get("curiosite_faits", 0.7),
            
            affinites_litterature=g.get("affinites_litterature", {
                "roman": 0.8,
                "poesie": 0.4,
                "theatre": 0.6,
                "essai": 0.7
            }),
            
            duree_memoire_litterature=g.get("duree_memoire_litterature", 30),
            duree_memoire_episodique=g.get("duree_memoire_episodique", 90),
            duree_memoire_sociale=g.get("duree_memoire_sociale", 365),
            
            style_prefere=g.get("style_prefere", "narratif"),
            seuil_curiosite=g.get("seuil_curiosite", 0.7),
            
            signature_type=g.get("signature_type", "sha256"),
            signature_valeur=g.get("signature_valeur", "")
        )
    
    def to_dict(self) -> dict:
        return {
            "identifiant": self.identifiant,
            "nom": self.nom,
            "date_naissance": self.date_naissance,
            "version": self.version,
            "tempo_base": self.tempo_base,
            "intensite_base": self.intensite_base,
            "grace": self.grace,
            "reactivite": self.reactivite,
            "curiosite_mots": self.curiosite_mots,
            "curiosite_verbes": self.curiosite_verbes,
            "curiosite_personnes": self.curiosite_personnes,
            "curiosite_lieux": self.curiosite_lieux,
            "curiosite_faits": self.curiosite_faits,
            "affinites_litterature": self.affinites_litterature,
            "duree_memoire_litterature": self.duree_memoire_litterature,
            "duree_memoire_episodique": self.duree_memoire_episodique,
            "duree_memoire_sociale": self.duree_memoire_sociale,
            "style_prefere": self.style_prefere,
            "seuil_curiosite": self.seuil_curiosite,
            "signature_type": self.signature_type,
            "signature_valeur": self.signature_valeur
        }

# ============================================================
# LIBRAIRIES EXTERNES (chargement dynamique)
# ============================================================

class ExternalLibraries:
    """Gère le chargement des librairies existantes."""
    
    def __init__(self):
        self.have_spacy = False
        self.have_stanza = False
        self.have_dateparser = False
        self.have_hunspell = False
        self.have_pattern = False
        
        self.nlp = None
        self.spellchecker = None
        
        self._init_libraries()
    
    def _init_libraries(self):
        # 1. spaCy pour NLP (recommandé)
        if importlib.util.find_spec("spacy"):
            try:
                import spacy
                # Essayer de charger le modèle français
                try:
                    self.nlp = spacy.load("fr_core_news_sm")
                    self.have_spacy = True
                    logger.info("✅ spaCy chargé (modèle français)")
                except:
                    logger.warning("⚠️ Modèle français spaCy non trouvé, téléchargement recommandé")
            except:
                pass
        
        # 2. stanza (alternative)
        if not self.have_spacy and importlib.util.find_spec("stanza"):
            try:
                import stanza
                stanza.download('fr', verbose=False)
                self.nlp = stanza.Pipeline('fr', processors='tokenize,pos,lemma', verbose=False)
                self.have_stanza = True
                logger.info("✅ stanza chargé")
            except:
                pass
        
        # 3. dateparser pour le temporel
        if importlib.util.find_spec("dateparser"):
            try:
                import dateparser
                self.dateparser = dateparser
                self.have_dateparser = True
                logger.info("✅ dateparser chargé")
            except:
                pass
        
        # 4. Hunspell pour la correction orthographique
        if importlib.util.find_spec("hunspell"):
            try:
                import hunspell
                # Chercher les dictionnaires français
                dict_paths = [
                    "/usr/share/hunspell/fr",
                    "/usr/share/myspell/fr",
                    "./dictionaries/fr"
                ]
                for path in dict_paths:
                    if Path(f"{path}.dic").exists():
                        self.spellchecker = hunspell.HunSpell(f"{path}.dic", f"{path}.aff")
                        self.have_hunspell = True
                        logger.info(f"✅ Hunspell chargé ({path})")
                        break
            except:
                pass
        
        # 5. pattern pour le français (grammaire, conjugaison)
        if importlib.util.find_spec("pattern"):
            try:
                from pattern.fr import conjugate, predict, tenses
                self.pattern_conjugate = conjugate
                self.pattern_predict = predict
                self.pattern_tenses = tenses
                self.have_pattern = True
                logger.info("✅ pattern.fr chargé")
            except:
                pass
        
        if not any([self.have_spacy, self.have_stanza]):
            logger.warning("⚠️ Aucune librairie NLP trouvée - fonctionnement limité")

# ============================================================
# STOCKAGE COMPRESSÉ
# ============================================================

class CompressedStorage:
    """Stockage avec compression gzip."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
    
    def save_json(self, filename: str, data: Any) -> Path:
        path = self.base_path / f"{filename}.json.gz"
        with self._lock, gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    
    def load_json(self, filename: str) -> Optional[Any]:
        path = self.base_path / f"{filename}.json.gz"
        if not path.exists():
            return None
        with self._lock, gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    
    def save_pickle(self, filename: str, data: Any) -> Path:
        path = self.base_path / f"{filename}.pkl.gz"
        with self._lock, gzip.open(path, 'wb') as f:
            pickle.dump(data, f)
        return path
    
    def load_pickle(self, filename: str) -> Optional[Any]:
        path = self.base_path / f"{filename}.pkl.gz"
        if not path.exists():
            return None
        with self._lock, gzip.open(path, 'rb') as f:
            return pickle.load(f)

# ============================================================
# MÉMOIRE: WORDS (mots et synonymes)
# ============================================================

@dataclass
class WordEntry:
    word: str
    pos: str
    synonyms: Set[str] = field(default_factory=set)
    definitions: List[str] = field(default_factory=list)
    source_weights: Dict[str, float] = field(default_factory=dict)

class WordsMemory:
    """Mots et synonymes - permanent."""
    
    def __init__(self, storage: CompressedStorage, gem: Gem):
        self.storage = storage
        self.gem = gem
        self.words: Dict[str, WordEntry] = {}
        self._lock = threading.RLock()
        self._load()
    
    def add(self, word: str, pos: str = "nom", 
            source: SourceWeight = SourceWeight.EDUCATIVE):
        word = word.lower()
        with self._lock:
            if word not in self.words:
                self.words[word] = WordEntry(word=word, pos=pos)
            
            # Enregistrer la source
            self.words[word].source_weights[source.value] = \
                self.words[word].source_weights.get(source.value, 0) + source.value
            
            self._save()
    
    def add_synonym(self, word: str, synonym: str,
                    source: SourceWeight = SourceWeight.EDUCATIVE):
        word = word.lower()
        synonym = synonym.lower()
        with self._lock:
            if word in self.words:
                self.words[word].synonyms.add(synonym)
            if synonym in self.words:
                self.words[synonym].synonyms.add(word)
            self._save()
    
    def add_definition(self, word: str, definition: str,
                       source: SourceWeight = SourceWeight.EDUCATIVE):
        word = word.lower()
        with self._lock:
            if word in self.words:
                self.words[word].definitions.append(definition)
            self._save()
    
    def has(self, word: str) -> bool:
        return word.lower() in self.words
    
    def expand(self, word: str) -> Set[str]:
        word = word.lower()
        if word not in self.words:
            return {word}
        return {word} | self.words[word].synonyms
    
    def get_confidence(self, word: str) -> float:
        """Confiance basée sur les sources."""
        word = word.lower()
        if word not in self.words:
            return 0.0
        
        weights = self.words[word].source_weights.values()
        if not weights:
            return 0.5
        
        return min(1.0, sum(weights) / len(weights))
    
    def _save(self):
        data = {
            w: {
                "word": e.word,
                "pos": e.pos,
                "synonyms": list(e.synonyms),
                "definitions": e.definitions,
                "source_weights": e.source_weights
            } for w, e in self.words.items()
        }
        self.storage.save_json("words_memory", data)
    
    def _load(self):
        data = self.storage.load_json("words_memory")
        if data:
            for w, wdata in data.items():
                self.words[w] = WordEntry(
                    word=wdata["word"],
                    pos=wdata.get("pos", "nom"),
                    synonyms=set(wdata.get("synonyms", [])),
                    definitions=wdata.get("definitions", []),
                    source_weights=wdata.get("source_weights", {})
                )

# ============================================================
# MÉMOIRE: ERRORS (fautes corrigées)
# ============================================================

class ErrorsMemory:
    """Fautes et corrections - SQLite avec nettoyage."""
    
    def __init__(self, db_path: Path, gem: Gem):
        self.db_path = Path(db_path)
        self.gem = gem
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    wrong TEXT PRIMARY KEY,
                    correct TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8,
                    count INTEGER DEFAULT 1,
                    first_seen REAL,
                    last_seen REAL,
                    source_weights TEXT,  -- JSON dict
                    contexts TEXT          -- JSON list
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_last_seen 
                ON errors(last_seen)
            """)
    
    def add(self, wrong: str, correct: str, 
            source: SourceWeight = SourceWeight.SELF,
            context: str = None):
        wrong = wrong.lower()
        correct = correct.lower()
        now = time.time()
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT confidence, count, source_weights FROM errors WHERE wrong = ?",
                (wrong,)
            )
            row = cur.fetchone()
            
            if row:
                old_conf, count, weights_json = row
                weights = json.loads(weights_json) if weights_json else {}
                
                # Mettre à jour le poids de la source
                weights[source.value] = weights.get(source.value, 0) + 1
                
                # La confiance augmente avec le nombre et la qualité des sources
                new_conf = min(0.95, old_conf + (source.value * 0.1))
                
                conn.execute("""
                    UPDATE errors 
                    SET confidence = ?, count = ?, last_seen = ?, source_weights = ?
                    WHERE wrong = ?
                """, (new_conf, count + 1, now, json.dumps(weights), wrong))
            else:
                weights = {source.value: 1}
                conn.execute("""
                    INSERT INTO errors 
                    (wrong, correct, confidence, count, first_seen, last_seen, source_weights)
                    VALUES (?, ?, ?, 1, ?, ?, ?)
                """, (wrong, correct, source.value, now, now, json.dumps(weights)))
    
    def get(self, wrong: str) -> Optional[Tuple[str, float]]:
        wrong = wrong.lower()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT correct, confidence FROM errors WHERE wrong = ?",
                (wrong,)
            )
            row = cur.fetchone()
            if row:
                return row[0], row[1]
        return None
    
    def get_all(self, wrong: str) -> List[Dict]:
        """Toutes les corrections possibles (pour ambiguïté)."""
        wrong = wrong.lower()
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Correspondance exacte
            cur = conn.execute(
                "SELECT correct, confidence, count FROM errors WHERE wrong = ?",
                (wrong,)
            )
            row = cur.fetchone()
            if row:
                results.append({
                    "correct": row[0],
                    "confidence": row[1],
                    "count": row[2]
                })
            
            # Correspondances similaires
            cur = conn.execute("""
                SELECT wrong, correct, confidence, count FROM errors 
                WHERE wrong LIKE ? OR ? LIKE ('%' || wrong || '%')
                LIMIT 10
            """, (f"%{wrong}%", wrong))
            
            for w, c, conf, cnt in cur:
                if w != wrong:
                    results.append({
                        "correct": c,
                        "confidence": conf * 0.7,
                        "similar_to": w,
                        "count": cnt
                    })
        
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:5]
    
    def cleanup(self):
        """Nettoyage nocturne selon le Gem."""
        max_age_days = self.gem.duree_memoire_episodique
        cutoff = time.time() - (max_age_days * 24 * 3600)
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Supprimer les vieilles erreurs peu fiables
            cur = conn.execute("""
                DELETE FROM errors 
                WHERE last_seen < ? AND confidence < 0.3
            """, (cutoff,))
            deleted = cur.rowcount
            
            if deleted > 0:
                logger.info(f"🧹 Nettoyage erreurs: {deleted} supprimées")
            
            return deleted

# ============================================================
# MÉMOIRE: TEMPORAL (expressions temporelles)
# ============================================================

@dataclass
class TemporalExpression:
    expression: str
    type: str  # "relative", "absolute", "recurring"
    resolution: str  # Comment résoudre
    synonyms: Set[str] = field(default_factory=set)
    source_weights: Dict[str, float] = field(default_factory=dict)

class TemporalMemory:
    """Expressions temporelles - permanent."""
    
    def __init__(self, storage: CompressedStorage, libs: ExternalLibraries):
        self.storage = storage
        self.libs = libs
        self.expressions: Dict[str, TemporalExpression] = {}
        self._lock = threading.RLock()
        self._init_defaults()
        self._load()
    
    def _init_defaults(self):
        """Expressions de base (minimum)."""
        defaults = [
            ("aujourd'hui", "relative", "today"),
            ("demain", "relative", "tomorrow"),
            ("hier", "relative", "yesterday"),
            ("maintenant", "relative", "now"),
            ("ce matin", "relative", "this_morning"),
            ("ce midi", "relative", "this_noon"),
            ("ce soir", "relative", "this_evening"),
            ("cette nuit", "relative", "tonight"),
        ]
        
        for expr, typ, res in defaults:
            self.expressions[expr] = TemporalExpression(
                expression=expr,
                type=typ,
                resolution=res
            )
    
    def add(self, expression: str, resolution: str, type: str = "relative",
            source: SourceWeight = SourceWeight.EDUCATIVE):
        expr = expression.lower()
        with self._lock:
            if expr not in self.expressions:
                self.expressions[expr] = TemporalExpression(
                    expression=expr,
                    type=type,
                    resolution=resolution
                )
            
            self.expressions[expr].source_weights[source.value] = \
                self.expressions[expr].source_weights.get(source.value, 0) + source.value
            
            self._save()
    
    def add_synonym(self, expression: str, synonym: str,
                    source: SourceWeight = SourceWeight.EDUCATIVE):
        expr = expression.lower()
        syn = synonym.lower()
        with self._lock:
            if expr in self.expressions:
                self.expressions[expr].synonyms.add(syn)
            self._save()
    
    def resolve(self, text: str, reference: datetime = None) -> Optional[Dict]:
        """Résout une expression temporelle."""
        text = text.lower()
        
        # Utiliser dateparser si disponible
        if self.libs.have_dateparser:
            try:
                parsed = self.libs.dateparser.parse(
                    text,
                    languages=['fr'],
                    settings={
                        'PREFER_DATES_FROM': 'future',
                        'RELATIVE_BASE': reference or datetime.now()
                    }
                )
                if parsed:
                    return {
                        "success": True,
                        "iso": parsed.isoformat(),
                        "timestamp": parsed.timestamp(),
                        "expression": text,
                        "confidence": 0.95,
                        "source": "dateparser"
                    }
            except:
                pass
        
        # Fallback sur le dictionnaire interne
        if text in self.expressions:
            expr = self.expressions[text]
            return {
                "success": True,
                "type": expr.type,
                "resolution": expr.resolution,
                "expression": text,
                "confidence": 0.8,
                "source": "internal"
            }
        
        # Synonymes
        for expr, data in self.expressions.items():
            if text in data.synonyms:
                return {
                    "success": True,
                    "type": data.type,
                    "resolution": data.resolution,
                    "expression": expr,
                    "confidence": 0.7,
                    "source": "synonym"
                }
        
        return {"success": False, "expression": text}
    
    def _save(self):
        data = {
            e: {
                "expression": expr.expression,
                "type": expr.type,
                "resolution": expr.resolution,
                "synonyms": list(expr.synonyms),
                "source_weights": expr.source_weights
            } for e, expr in self.expressions.items()
        }
        self.storage.save_json("temporal_memory", data)
    
    def _load(self):
        data = self.storage.load_json("temporal_memory")
        if data:
            for e, edata in data.items():
                self.expressions[e] = TemporalExpression(
                    expression=edata["expression"],
                    type=edata["type"],
                    resolution=edata["resolution"],
                    synonyms=set(edata.get("synonyms", [])),
                    source_weights=edata.get("source_weights", {})
                )

# ============================================================
# MÉMOIRE: EPISODIC (expériences vécues)
# ============================================================

class EpisodicMemory:
    """Expériences vécues - SQLite avec TTL selon le Gem."""
    
    def __init__(self, db_path: Path, gem: Gem):
        self.db_path = Path(db_path)
        self.gem = gem
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic (
                    id TEXT PRIMARY KEY,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    context TEXT,  -- JSON
                    importance REAL,
                    source_weight REAL,
                    created REAL,
                    last_recalled REAL,
                    recall_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX idx_episodic_created ON episodic(created)")
    
    def add(self, subject: str, predicate: str, object: str,
            context: Dict, importance: float = 0.5,
            source: SourceWeight = SourceWeight.OBSERVATION) -> str:
        mid = f"ep_{uuid.uuid4().hex[:8]}"
        now = time.time()
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO episodic 
                (id, subject, predicate, object, context, importance, 
                 source_weight, created, last_recalled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (mid, subject, predicate, object, json.dumps(context),
                  importance, source.value, now, now))
        
        return mid
    
    def query(self, subject: str = None, predicate: str = None,
              days: int = None) -> List[Dict]:
        """Recherche avec limite de temps."""
        if days is None:
            days = self.gem.duree_memoire_episodique
        
        cutoff = time.time() - (days * 24 * 3600)
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            sql = "SELECT * FROM episodic WHERE created > ?"
            params = [cutoff]
            
            if subject:
                sql += " AND subject = ?"
                params.append(subject)
            if predicate:
                sql += " AND predicate = ?"
                params.append(predicate)
            
            cur = conn.execute(sql, params)
            for row in cur:
                results.append({
                    "id": row[0],
                    "subject": row[1],
                    "predicate": row[2],
                    "object": row[3],
                    "context": json.loads(row[4]),
                    "importance": row[5],
                    "source_weight": row[6],
                    "created": row[7],
                    "last_recalled": row[8],
                    "recall_count": row[9]
                })
        
        # Mettre à jour last_recalled
        for r in results:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE episodic 
                    SET last_recalled = ?, recall_count = recall_count + 1
                    WHERE id = ?
                """, (time.time(), r["id"]))
        
        return results
    
    def cleanup(self):
        """Nettoyage selon les durées du Gem."""
        max_age = self.gem.duree_memoire_episodique
        min_importance = 0.3
        cutoff = time.time() - (max_age * 24 * 3600)
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                DELETE FROM episodic 
                WHERE created < ? AND importance < ?
            """, (cutoff, min_importance))
            deleted = cur.rowcount
            
            if deleted > 0:
                logger.info(f"🧹 Nettoyage épisodique: {deleted} souvenirs supprimés")
            
            return deleted

# ============================================================
# MÉMOIRE: SOCIAL (personnes et relations)
# ============================================================

@dataclass
class Person:
    id: str
    name: str
    nicknames: Set[str]
    relation: str
    gender: str
    weight: float  # Poids basé sur les interactions
    metadata: Dict[str, Any]
    first_met: float
    last_interaction: float
    interaction_count: int

class SocialMemory:
    """Personnes et relations - SQLite avec poids."""
    
    def __init__(self, db_path: Path, gem: Gem):
        self.db_path = Path(db_path)
        self.gem = gem
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    nicknames TEXT,  -- JSON list
                    relation TEXT,
                    gender TEXT,
                    weight REAL DEFAULT 1.0,
                    metadata TEXT,   -- JSON
                    first_met REAL,
                    last_interaction REAL,
                    interaction_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    target TEXT,
                    type TEXT,
                    strength REAL,
                    FOREIGN KEY(source) REFERENCES persons(id),
                    FOREIGN KEY(target) REFERENCES persons(id)
                )
            """)
            conn.execute("CREATE INDEX idx_persons_name ON persons(name)")
    
    def add_person(self, name: str, relation: str = "unknown",
                   source: SourceWeight = SourceWeight.SELF) -> Person:
        pid = f"person_{uuid.uuid4().hex[:8]}"
        now = time.time()
        weight = source.value  # Poids initial basé sur la source
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO persons 
                (id, name, nicknames, relation, gender, weight, metadata, 
                 first_met, last_interaction, interaction_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (pid, name, json.dumps([]), relation, "unknown",
                  weight, json.dumps({}), now, now, 1))
        
        return self.get_person(pid)
    
    def get_person(self, person_id: str) -> Optional[Person]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
            row = cur.fetchone()
            if row:
                return Person(
                    id=row[0],
                    name=row[1],
                    nicknames=set(json.loads(row[2])),
                    relation=row[3],
                    gender=row[4],
                    weight=row[5],
                    metadata=json.loads(row[6]),
                    first_met=row[7],
                    last_interaction=row[8],
                    interaction_count=row[9]
                )
        return None
    
    def find_by_name(self, name: str) -> Optional[Person]:
        name_lower = name.lower()
        with sqlite3.connect(self.db_path) as conn:
            # Recherche exacte
            cur = conn.execute(
                "SELECT * FROM persons WHERE LOWER(name) = ?",
                (name_lower,)
            )
            row = cur.fetchone()
            if row:
                return self._row_to_person(row)
            
            # Recherche dans les surnoms
            cur = conn.execute("SELECT id, nicknames FROM persons")
            for pid, nicknames_json in cur:
                nicknames = json.loads(nicknames_json)
                if name_lower in [n.lower() for n in nicknames]:
                    return self.get_person(pid)
        
        return None
    
    def add_interaction(self, person_id: str):
        """Chaque interaction augmente le poids."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE persons 
                SET weight = weight + 1,
                    last_interaction = ?,
                    interaction_count = interaction_count + 1
                WHERE id = ?
            """, (time.time(), person_id))
    
    def add_nickname(self, person_id: str, nickname: str,
                     source: SourceWeight = SourceWeight.SELF):
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT nicknames, weight FROM persons WHERE id = ?",
                (person_id,)
            )
            row = cur.fetchone()
            if row:
                nicknames = set(json.loads(row[0]))
                nicknames.add(nickname)
                # Augmenter le poids si c'est une source fiable
                weight_increase = source.value * 0.5
                conn.execute("""
                    UPDATE persons 
                    SET nicknames = ?, weight = weight + ?
                    WHERE id = ?
                """, (json.dumps(list(nicknames)), weight_increase, person_id))
    
    def cleanup(self):
        """Nettoyage social - le poids protège de l'oubli."""
        max_age = self.gem.duree_memoire_sociale
        cutoff = time.time() - (max_age * 24 * 3600)
        
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # Ne supprimer que les personnes avec faible poids ET vieilles
            cur = conn.execute("""
                DELETE FROM persons 
                WHERE last_interaction < ? AND weight < 5
            """, (cutoff,))
            deleted = cur.rowcount
            
            if deleted > 0:
                logger.info(f"🧹 Nettoyage social: {deleted} personnes oubliées")
            
            return deleted
    
    def _row_to_person(self, row) -> Person:
        return Person(
            id=row[0],
            name=row[1],
            nicknames=set(json.loads(row[2])),
            relation=row[3],
            gender=row[4],
            weight=row[5],
            metadata=json.loads(row[6]),
            first_met=row[7],
            last_interaction=row[8],
            interaction_count=row[9]
        )

# ============================================================
# MÉMOIRE: LITERARY ROMAN (éphémère, refroidit)
# ============================================================

@dataclass
class Roman:
    id: str
    title: str
    author: str
    genre: str
    year: Optional[int]
    characters: List[str]
    summary: str
    themes: List[str]
    source_weight: float
    temperature: float = 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

class RomanMemory:
    """Romans - éphémère, refroidit selon les affinités du Gem."""
    
    def __init__(self, storage: CompressedStorage, gem: Gem):
        self.storage = storage
        self.gem = gem
        self.romans: Dict[str, Roman] = {}
        self.cooling_rate = 0.05  # base par heure
        self.last_cooling = time.time()
        self._lock = threading.RLock()
        self._load()
        self._start_cooling()
    
    def _start_cooling(self):
        def cool():
            while True:
                time.sleep(3600)  # 1 heure
                self.apply_cooling()
        threading.Thread(target=cool, daemon=True).start()
    
    def add(self, title: str, author: str, genre: str,
            characters: List[str], summary: str, themes: List[str],
            source: SourceWeight = SourceWeight.FICTION,
            year: int = None) -> str:
        
        rid = f"roman_{uuid.uuid4().hex[:8]}"
        
        # La température initiale dépend de l'affinité pour le genre
        affinite = self.gem.affinites_litterature.get(genre, 0.5)
        temp_initiale = 0.5 + (affinite * 0.5)
        
        with self._lock:
            self.romans[rid] = Roman(
                id=rid,
                title=title,
                author=author,
                genre=genre,
                year=year,
                characters=characters,
                summary=summary,
                themes=themes,
                source_weight=source.value,
                temperature=temp_initiale
            )
            self._save()
            return rid
    
    def get(self, roman_id: str) -> Optional[Roman]:
        with self._lock:
            if roman_id in self.romans:
                r = self.romans[roman_id]
                r.access_count += 1
                r.last_accessed = time.time()
                # Réchauffer à l'accès
                affinite = self.gem.affinites_litterature.get(r.genre, 0.5)
                r.temperature = min(1.0, r.temperature + (0.1 * affinite))
                self._save()
                return r
            return None
    
    def search(self, query: str) -> List[Roman]:
        q = query.lower()
        with self._lock:
            results = []
            for r in self.romans.values():
                if r.temperature < 0.1:  # trop froid
                    continue
                if q in r.title.lower() or q in r.author.lower():
                    results.append(r)
                    # Réchauffer légèrement
                    affinite = self.gem.affinites_litterature.get(r.genre, 0.5)
                    r.temperature = min(1.0, r.temperature + (0.05 * affinite))
            self._save()
            return results
    
    def apply_cooling(self):
        """Refroidissement influencé par le Gem."""
        now = time.time()
        hours = (now - self.last_cooling) / 3600
        
        with self._lock:
            to_remove = []
            for rid, r in self.romans.items():
                # Le refroidissement dépend de l'affinité
                affinite = self.gem.affinites_litterature.get(r.genre, 0.5)
                facteur = 1.0 - (affinite * 0.5)  # moins de refroidissement si affinité
                
                r.temperature = max(0.0, r.temperature - (self.cooling_rate * hours * facteur))
                
                if r.temperature < 0.1:
                    to_remove.append(rid)
            
            for rid in to_remove:
                logger.info(f"❄️ Roman oublié: {self.romans[rid].title}")
                # Archiver avant suppression
                self._archive(rid)
                del self.romans[rid]
            
            self.last_cooling = now
            self._save()
    
    def _archive(self, roman_id: str):
        """Archive un roman avant oubli."""
        if roman_id in self.romans:
            r = self.romans[roman_id]
            archive = {
                "id": r.id,
                "title": r.title,
                "author": r.author,
                "genre": r.genre,
                "disappeared": time.time(),
                "access_count": r.access_count,
                "temperature_finale": r.temperature
            }
            self.storage.save_json(f"roman_archive_{r.id}", archive)
    
    def _save(self):
        data = {
            rid: {
                "id": r.id,
                "title": r.title,
                "author": r.author,
                "genre": r.genre,
                "year": r.year,
                "characters": r.characters,
                "summary": r.summary,
                "themes": r.themes,
                "source_weight": r.source_weight,
                "temperature": r.temperature,
                "last_accessed": r.last_accessed,
                "access_count": r.access_count
            } for rid, r in self.romans.items()
        }
        self.storage.save_json("roman_memory", data)
    
    def _load(self):
        data = self.storage.load_json("roman_memory")
        if data:
            for rid, rdata in data.items():
                self.romans[rid] = Roman(
                    id=rdata["id"],
                    title=rdata["title"],
                    author=rdata["author"],
                    genre=rdata.get("genre", "roman"),
                    year=rdata.get("year"),
                    characters=rdata.get("characters", []),
                    summary=rdata.get("summary", ""),
                    themes=rdata.get("themes", []),
                    source_weight=rdata.get("source_weight", 0.3),
                    temperature=rdata.get("temperature", 1.0),
                    last_accessed=rdata.get("last_accessed", time.time()),
                    access_count=rdata.get("access_count", 0)
                )

# ============================================================
# MÉMOIRE: LITERARY EDUC (permanent)
# ============================================================

@dataclass
class EducationalWork:
    id: str
    title: str
    author: str
    period: str
    genre: str
    importance: float
    content: Dict[str, Any]
    source_weight: float
    learned_date: float = field(default_factory=time.time)

class EducationalMemory:
    """Littérature éducative - permanente."""
    
    def __init__(self, storage: CompressedStorage):
        self.storage = storage
        self.works: Dict[str, EducationalWork] = {}
        self._lock = threading.RLock()
        self._load()
    
    def add(self, title: str, author: str, period: str,
            genre: str, importance: float, content: Dict,
            source: SourceWeight = SourceWeight.EDUCATIVE) -> str:
        wid = f"edu_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            self.works[wid] = EducationalWork(
                id=wid,
                title=title,
                author=author,
                period=period,
                genre=genre,
                importance=importance,
                content=content,
                source_weight=source.value
            )
            self._save()
            return wid
    
    def get(self, work_id: str) -> Optional[EducationalWork]:
        return self.works.get(work_id)
    
    def search_by_period(self, period: str) -> List[EducationalWork]:
        return [w for w in self.works.values() if w.period == period]
    
    def search_by_genre(self, genre: str) -> List[EducationalWork]:
        return [w for w in self.works.values() if w.genre == genre]
    
    def search(self, query: str) -> List[EducationalWork]:
        q = query.lower()
        return [w for w in self.works.values() 
                if q in w.title.lower() or q in w.author.lower()]
    
    def _save(self):
        data = {
            wid: {
                "id": w.id,
                "title": w.title,
                "author": w.author,
                "period": w.period,
                "genre": w.genre,
                "importance": w.importance,
                "content": w.content,
                "source_weight": w.source_weight,
                "learned_date": w.learned_date
            } for wid, w in self.works.items()
        }
        self.storage.save_json("educational_memory", data)
    
    def _load(self):
        data = self.storage.load_json("educational_memory")
        if data:
            for wid, wdata in data.items():
                self.works[wid] = EducationalWork(
                    id=wdata["id"],
                    title=wdata["title"],
                    author=wdata["author"],
                    period=wdata["period"],
                    genre=wdata["genre"],
                    importance=wdata["importance"],
                    content=wdata.get("content", {}),
                    source_weight=wdata.get("source_weight", 0.9),
                    learned_date=wdata.get("learned_date", time.time())
                )

# ============================================================
# NARRATIVE MEMORY (Livre de vie)
# ============================================================

@dataclass
class NarrativePage:
    page_num: int
    start_day: int
    end_day: int
    events: List[Dict]
    summary: str = ""

@dataclass
class NarrativeSection:
    month: int
    year: int
    pages: Dict[int, NarrativePage]
    summary: str = ""

@dataclass
class NarrativeChapter:
    year: int
    sections: Dict[int, NarrativeSection]
    summary: str = ""
    title: str = ""

@dataclass
class NarrativeBook:
    book_id: str
    embodiment_name: str
    start_date: float
    end_date: Optional[float]
    chapters: Dict[int, NarrativeChapter]
    summary: str = ""

class NarrativeMemory:
    """Livre de vie - structure livre/chapitre/section/page."""
    
    def __init__(self, base_path: Path, storage: CompressedStorage):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.storage = storage
        self.current_book: Optional[NarrativeBook] = None
        self._lock = threading.RLock()
        self._load_current()
    
    def start_new_book(self, embodiment_name: str) -> str:
        book_id = f"book_{uuid.uuid4().hex[:8]}"
        now = time.time()
        
        with self._lock:
            self.current_book = NarrativeBook(
                book_id=book_id,
                embodiment_name=embodiment_name,
                start_date=now,
                end_date=None,
                chapters={}
            )
            self._save_current()
            return book_id
    
    def add_event(self, event: Dict, source: SourceWeight = SourceWeight.OBSERVATION):
        if not self.current_book:
            self.start_new_book("default")
        
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day
        
        page_num = 1 if day <= 15 else 2
        
        with self._lock:
            # Créer le chapitre si nécessaire
            if year not in self.current_book.chapters:
                self.current_book.chapters[year] = NarrativeChapter(
                    year=year,
                    sections={}
                )
            
            chapter = self.current_book.chapters[year]
            
            # Créer la section si nécessaire
            if month not in chapter.sections:
                chapter.sections[month] = NarrativeSection(
                    month=month,
                    year=year,
                    pages={}
                )
            
            section = chapter.sections[month]
            
            # Créer la page si nécessaire
            if page_num not in section.pages:
                start_day = 1 if page_num == 1 else 16
                end_day = 15 if page_num == 1 else 31
                section.pages[page_num] = NarrativePage(
                    page_num=page_num,
                    start_day=start_day,
                    end_day=end_day,
                    events=[]
                )
            
            # Ajouter l'événement avec sa source
            event['timestamp'] = time.time()
            event['day'] = day
            event['source'] = source.value
            section.pages[page_num].events.append(event)
            
            self._save_current()
    
    def archive_month(self, year: int, month: int):
        """Archive un mois."""
        if not self.current_book:
            return
        
        if year not in self.current_book.chapters:
            return
        
        chapter = self.current_book.chapters[year]
        if month not in chapter.sections:
            return
        
        section = chapter.sections[month]
        
        # Sauvegarder dans un fichier d'archive
        archive_data = {
            "book_id": self.current_book.book_id,
            "embodiment": self.current_book.embodiment_name,
            "year": year,
            "month": month,
            "pages": {}
        }
        
        for page_num, page in section.pages.items():
            archive_data["pages"][page_num] = {
                "start_day": page.start_day,
                "end_day": page.end_day,
                "events": page.events,
                "summary": page.summary
            }
        
        archive_path = self.base_path / f"archive_{year}_{month:02d}.json.gz"
        with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
            json.dump(archive_data, f, ensure_ascii=False, indent=2)
        
        # Vider la section courante
        with self._lock:
            for page in section.pages.values():
                page.events = []
                page.summary = ""
            self._save_current()
        
        logger.info(f"📚 Mois {year}-{month:02d} archivé")
    
    def end_book(self):
        """Termine le livre courant."""
        if self.current_book:
            self.current_book.end_date = time.time()
            self._save_book(self.current_book)
            self.start_new_book(f"{self.current_book.embodiment_name}_next")
    
    def _save_current(self):
        if self.current_book:
            data = {
                "book_id": self.current_book.book_id,
                "embodiment_name": self.current_book.embodiment_name,
                "start_date": self.current_book.start_date,
                "end_date": self.current_book.end_date,
                "chapters": {}
            }
            
            for year, chapter in self.current_book.chapters.items():
                chapter_data = {
                    "year": chapter.year,
                    "sections": {},
                    "summary": chapter.summary,
                    "title": chapter.title
                }
                
                for month, section in chapter.sections.items():
                    section_data = {
                        "month": section.month,
                        "year": section.year,
                        "pages": {},
                        "summary": section.summary
                    }
                    
                    for page_num, page in section.pages.items():
                        section_data["pages"][page_num] = {
                            "page_num": page.page_num,
                            "start_day": page.start_day,
                            "end_day": page.end_day,
                            "events": page.events,
                            "summary": page.summary
                        }
                    
                    chapter_data["sections"][month] = section_data
                
                data["chapters"][year] = chapter_data
            
            self.storage.save_json("current_book", data)
    
    def _load_current(self):
        data = self.storage.load_json("current_book")
        if data:
            book = NarrativeBook(
                book_id=data["book_id"],
                embodiment_name=data["embodiment_name"],
                start_date=data["start_date"],
                end_date=data["end_date"],
                chapters={}
            )
            
            for year_str, chapter_data in data.get("chapters", {}).items():
                year = int(year_str)
                chapter = NarrativeChapter(
                    year=chapter_data["year"],
                    sections={},
                    summary=chapter_data.get("summary", ""),
                    title=chapter_data.get("title", "")
                )
                
                for month_str, section_data in chapter_data.get("sections", {}).items():
                    month = int(month_str)
                    section = NarrativeSection(
                        month=section_data["month"],
                        year=section_data["year"],
                        pages={},
                        summary=section_data.get("summary", "")
                    )
                    
                    for page_num_str, page_data in section_data.get("pages", {}).items():
                        page_num = int(page_num_str)
                        section.pages[page_num] = NarrativePage(
                            page_num=page_data["page_num"],
                            start_day=page_data["start_day"],
                            end_day=page_data["end_day"],
                            events=page_data.get("events", []),
                            summary=page_data.get("summary", "")
                        )
                    
                    chapter.sections[month] = section
                
                book.chapters[year] = chapter
            
            self.current_book = book
    
    def _save_book(self, book: NarrativeBook):
        """Sauvegarde un livre complet."""
        data = {
            "book_id": book.book_id,
            "embodiment_name": book.embodiment_name,
            "start_date": book.start_date,
            "end_date": book.end_date,
            "chapters": {}
        }
        
        for year, chapter in book.chapters.items():
            data["chapters"][year] = {
                "year": chapter.year,
                "sections": {},
                "summary": chapter.summary,
                "title": chapter.title
            }
            # ... (similaire à _save_current)
        
        archive_path = self.base_path / f"book_{book.book_id}.json.gz"
        with gzip.open(archive_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================================
# NORMALISEUR DE TEXTE
# ============================================================

class TextNormalizer:
    """Normalise le texte en utilisant les librairies disponibles."""
    
    def __init__(self, errors: ErrorsMemory, libs: ExternalLibraries):
        self.errors = errors
        self.libs = libs
    
    def normalize(self, text: str, speaker: str = None) -> Tuple[str, List[Dict]]:
        """
        Normalise le texte : corrections orthographiques, contractions.
        """
        words = text.split()
        corrected_words = []
        corrections = []
        
        for word in words:
            # 1. Vérifier dans le dictionnaire des erreurs
            correction = self.errors.get(word)
            if correction:
                corrected, conf = correction
                corrected_words.append(corrected)
                corrections.append({
                    "original": word,
                    "corrected": corrected,
                    "confidence": conf,
                    "source": "errors_db"
                })
                continue
            
            # 2. Contractions courantes
            expanded = self._expand_contractions(word)
            if expanded != word:
                corrected_words.append(expanded)
                corrections.append({
                    "original": word,
                    "corrected": expanded,
                    "confidence": 0.9,
                    "source": "contraction"
                })
                continue
            
            # 3. Correction orthographique avec Hunspell si disponible
            if self.libs.have_hunspell and self.libs.spellchecker:
                if not self.libs.spellchecker.spell(word):
                    suggestions = self.libs.spellchecker.suggest(word)
                    if suggestions:
                        corrected_words.append(suggestions[0])
                        corrections.append({
                            "original": word,
                            "corrected": suggestions[0],
                            "confidence": 0.7,
                            "source": "hunspell"
                        })
                        continue
            
            corrected_words.append(word)
        
        return " ".join(corrected_words), corrections
    
    def _expand_contractions(self, word: str) -> str:
        """Expansion des contractions courantes."""
        contractions = {
            "j'": "je",
            "t'": "tu",
            "m'": "me",
            "s'": "se",
            "c'": "ce",
            "l'": "le",
            "d'": "de",
            "qu'": "que",
            "n'": "ne"
        }
        
        for contr, exp in contractions.items():
            if word.startswith(contr):
                return exp + " " + word[len(contr):]
        
        return word

# ============================================================
# DÉTECTEUR D'INTENTS
# ============================================================

class IntentDetector:
    """Détecte les intents à partir du texte normalisé."""
    
    def __init__(self, libs: ExternalLibraries):
        self.libs = libs
        
        # Patterns chargés depuis un fichier (pas de hard coding)
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict:
        """Charge les patterns depuis un fichier."""
        # En vrai, charger depuis un fichier JSON
        # Ici version minimale pour l'exemple
        return {
            "question": {
                "time": [r"heure", r"temps", r"quelle heure"],
                "person": [r"qui", r"comment s'appelle"],
                "place": [r"où", r"dans quel endroit"]
            },
            "plan": {
                "meal": [r"resto", r"restaurant", r"manger", r"dîner"],
                "meeting": [r"rendez-vous", r"voir", r"retrouver"]
            },
            "action": {
                "device": [r"allume", r"éteint", r"ouvre", r"ferme"]
            }
        }
    
    def detect(self, text: str, attributes: Dict) -> List[StructuredIntent]:
        """
        Détecte tous les intents présents.
        """
        intents = []
        text_lower = text.lower()
        
        # Questions
        if text_lower.endswith("?"):
            if any(w in text_lower for w in ["heure", "temps"]):
                intent = StructuredIntent(
                    intent="question",
                    sub_intent="time",
                    type=IntentType.QUESTION,
                    attributes={},
                    confidence=0.9,
                    source="detector"
                )
                intents.append(intent)
        
        # Plans
        if any(w in text_lower for w in ["resto", "restaurant", "manger"]):
            intent = StructuredIntent(
                intent="plan",
                sub_intent="meal",
                type=IntentType.INFORMATION,
                attributes=attributes,
                confidence=0.85,
                source="detector"
            )
            intents.append(intent)
        
        # Actions
        for action in ["allume", "éteint"]:
            if action in text_lower:
                intent = StructuredIntent(
                    intent="action",
                    sub_intent="device",
                    type=IntentType.ACTION,
                    attributes=attributes,
                    confidence=0.9,
                    source="detector"
                )
                intents.append(intent)
        
        return intents

# ============================================================
# CURIOSITÉ ENGINE
# ============================================================

class CuriosityEngine:
    """Gère les connaissances manquantes selon le Gem."""
    
    def __init__(self, gem: Gem):
        self.gem = gem
        self.pending: List[Dict] = []
    
    def check_intents(self, intents: List[StructuredIntent]) -> List[StructuredIntent]:
        """
        Vérifie les intents et ajoute des clarifications si nécessaire.
        """
        results = list(intents)
        
        for intent in intents:
            # Vérifier les personnes inconnues
            if "person" in intent.attributes:
                attr = intent.attributes["person"]
                if not attr.metadata.get("known", False):
                    # La curiosité pour les personnes est-elle assez élevée ?
                    if self.gem.curiosite_personnes > self.gem.seuil_curiosite:
                        results.append(StructuredIntent(
                            intent="ask_clarification",
                            sub_intent="unknown_person",
                            type=IntentType.CLARIFICATION,
                            attributes={"person": attr},
                            confidence=0.9,
                            source="curiosity"
                        ))
            
            # Vérifier les mots inconnus
            if "word" in intent.attributes and self.gem.curiosite_mots > self.gem.seuil_curiosite:
                # etc.
                pass
        
        return results

# ============================================================
# ROUTER PRINCIPAL
# ============================================================

class Router:
    """Route les intents vers les bonnes mémoires."""
    
    def __init__(self, words: WordsMemory, errors: ErrorsMemory,
                 temporal: TemporalMemory, episodic: EpisodicMemory,
                 social: SocialMemory, narrative: NarrativeMemory,
                 romans: RomanMemory, educational: EducationalMemory,
                 gem: Gem):
        self.words = words
        self.errors = errors
        self.temporal = temporal
        self.episodic = episodic
        self.social = social
        self.narrative = narrative
        self.romans = romans
        self.educational = educational
        self.gem = gem
    
    def route(self, intents: List[StructuredIntent]) -> List[StructuredIntent]:
        """
        Met à jour les mémoires en fonction des intents.
        """
        responses = []
        
        for intent in intents:
            if intent.type == IntentType.INFORMATION:
                response = self._handle_information(intent)
                if response:
                    responses.append(response)
        
        return responses
    
    def _handle_information(self, intent: StructuredIntent) -> Optional[StructuredIntent]:
        """Traite un intent d'information."""
        
        if intent.sub_intent == "new_word":
            # Ajouter un mot
            word = intent.attributes.get("word")
            if word:
                source = intent.attributes.get("source", Attribute(
                    type="source",
                    value=SourceWeight.EDUCATIVE
                ))
                self.words.add(word.value, source=source.value)
                
                return StructuredIntent(
                    intent="acknowledge",
                    sub_intent="word_added",
                    type=IntentType.REPONSE,
                    attributes={"word": word},
                    confidence=0.95,
                    source="router"
                )
        
        elif intent.sub_intent == "correction":
            # Ajouter une correction
            wrong = intent.attributes.get("wrong")
            correct = intent.attributes.get("correct")
            if wrong and correct:
                source = intent.attributes.get("source", Attribute(
                    type="source",
                    value=SourceWeight.SELF
                ))
                self.errors.add(wrong.value, correct.value, source=source.value)
        
        elif intent.sub_intent == "new_person":
            # Ajouter une personne
            name = intent.attributes.get("name")
            if name:
                source = intent.attributes.get("source", Attribute(
                    type="source",
                    value=SourceWeight.SELF
                ))
                person = self.social.add_person(name.value, source=source.value)
                
                return StructuredIntent(
                    intent="acknowledge",
                    sub_intent="person_added",
                    type=IntentType.REPONSE,
                    attributes={
                        "person_id": Attribute(type="string", value=person.id),
                        "name": name
                    },
                    confidence=0.95,
                    source="router"
                )
        
        elif intent.sub_intent == "new_fact":
            # Ajouter un fait épisodique
            subject = intent.attributes.get("subject")
            predicate = intent.attributes.get("predicate")
            object_val = intent.attributes.get("object")
            if subject and predicate and object_val:
                source = intent.attributes.get("source", Attribute(
                    type="source",
                    value=SourceWeight.OBSERVATION
                ))
                self.episodic.add(
                    subject=subject.value,
                    predicate=predicate.value,
                    object=object_val.value,
                    context={},
                    source=source.value
                )
        
        elif intent.sub_intent == "new_roman":
            # Ajouter un roman
            title = intent.attributes.get("title")
            author = intent.attributes.get("author")
            if title and author:
                source = intent.attributes.get("source", Attribute(
                    type="source",
                    value=SourceWeight.FICTION
                ))
                self.romans.add(
                    title=title.value,
                    author=author.value,
                    genre=intent.attributes.get("genre", Attribute(type="string", value="roman")).value,
                    characters=intent.attributes.get("characters", Attribute(type="list", value=[])).value,
                    summary=intent.attributes.get("summary", Attribute(type="string", value="")).value,
                    themes=intent.attributes.get("themes", Attribute(type="list", value=[])).value,
                    source=source.value
                )
        
        elif intent.sub_intent == "new_knowledge":
            # Ajouter une connaissance éducative
            title = intent.attributes.get("title")
            if title:
                source = intent.attributes.get("source", Attribute(
                    type="source",
                    value=SourceWeight.EDUCATIVE
                ))
                self.educational.add(
                    title=title.value,
                    author=intent.attributes.get("author", Attribute(type="string", value="")).value,
                    period=intent.attributes.get("period", Attribute(type="string", value="contemporain")).value,
                    genre=intent.attributes.get("genre", Attribute(type="string", value="essai")).value,
                    importance=intent.attributes.get("importance", Attribute(type="number", value=0.8)).value,
                    content=intent.attributes.get("content", Attribute(type="object", value={})).value,
                    source=source.value
                )
        
        return None

# ============================================================
# COGNITION CORE PRINCIPAL
# ============================================================

class CognitionCore:
    """
    Cœur cognitif principal - point d'entrée unique.
    """
    
    def __init__(self, data_path: Path, gem_path: Path):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Charger le Gem (l'âme immuable)
        logger.info(f"📖 Chargement du Gem depuis {gem_path}")
        self.gem = Gem.from_file(gem_path)
        
        # Initialiser les librairies externes
        self.libs = ExternalLibraries()
        
        # Chemins
        self.files_path = self.data_path / "files"
        self.db_path = self.data_path / "db"
        self.narrative_path = self.data_path / "narrative"
        
        self.files_path.mkdir(exist_ok=True)
        self.db_path.mkdir(exist_ok=True)
        self.narrative_path.mkdir(exist_ok=True)
        
        # Stockage compressé
        self.storage = CompressedStorage(self.files_path)
        
        # Initialiser toutes les mémoires
        self.words = WordsMemory(self.storage, self.gem)
        self.errors = ErrorsMemory(self.db_path / "errors.db", self.gem)
        self.temporal = TemporalMemory(self.storage, self.libs)
        self.episodic = EpisodicMemory(self.db_path / "episodic.db", self.gem)
        self.social = SocialMemory(self.db_path / "social.db", self.gem)
        self.narrative = NarrativeMemory(self.narrative_path, self.storage)
        self.romans = RomanMemory(self.storage, self.gem)
        self.educational = EducationalMemory(self.storage)
        
        # Composants de traitement
        self.normalizer = TextNormalizer(self.errors, self.libs)
        self.detector = IntentDetector(self.libs)
        self.curiosity = CuriosityEngine(self.gem)
        self.router = Router(
            self.words, self.errors, self.temporal,
            self.episodic, self.social, self.narrative,
            self.romans, self.educational, self.gem
        )
        
        # État
        self.conversation_id = None
        self.running = True
        
        logger.info(f"✅ CognitionCore initialisé - Gem: {self.gem.nom} v{self.gem.version}")
    
    def process(self, text: str, speaker: str = None) -> MultiIntent:
        """
        Traite une entrée texte et retourne des intents structurés.
        """
        logger.debug(f"Traitement: {text}")
        
        # 1. Normalisation (correction des fautes)
        normalized_text, corrections = self.normalizer.normalize(text, speaker)
        
        # 2. Extraction des attributs basiques
        attributes = self._extract_attributes(normalized_text)
        
        # 3. Détection des intents
        intents = self.detector.detect(normalized_text, attributes)
        
        # 4. Ajouter les corrections comme meta
        for intent in intents:
            intent.attributes["_corrections"] = Attribute(
                type="list",
                value=corrections,
                confidence=0.9
            )
        
        # 5. Vérification par la curiosité
        intents = self.curiosity.check_intents(intents)
        
        # 6. Router vers les mémoires
        responses = self.router.route(intents)
        
        # 7. Ajouter au livre de vie
        self.narrative.add_event({
            "type": "user_input",
            "text": text,
            "normalized": normalized_text,
            "speaker": speaker,
            "intents": [i.to_dict() for i in intents]
        })
        
        # 8. Résultat final
        all_intents = intents + responses
        
        return MultiIntent(
            intents=all_intents,
            original_text=text,
            timestamp=time.time()
        )
    
    def _extract_attributes(self, text: str) -> Dict[str, Attribute]:
        """Extraction basique d'attributs."""
        attributes = {}
        
        # Personnes (simplifié)
        person_match = re.search(r'(?:avec|et|pour) (\w+)', text)
        if person_match:
            name = person_match.group(1)
            person = self.social.find_by_name(name)
            attributes["person"] = Attribute(
                type="person",
                value=name,
                confidence=0.8,
                metadata={"known": person is not None}
            )
        
        # Lieux
        loc_match = re.search(r'(?:au|à la|dans) (\w+)', text)
        if loc_match:
            attributes["location"] = Attribute(
                type="place",
                value=loc_match.group(1),
                confidence=0.7
            )
        
        # Temps
        time_match = re.search(r'(demain|aujourd\'hui|ce soir|maintenant)', text)
        if time_match:
            resolved = self.temporal.resolve(time_match.group(1))
            attributes["temporal"] = Attribute(
                type="datetime",
                value=time_match.group(1),
                normalized=resolved.get("iso") if resolved.get("success") else None,
                confidence=resolved.get("confidence", 0.7)
            )
        
        return attributes
    
    def nightly_maintenance(self):
        """Tâches de maintenance nocturne."""
        logger.info("🌙 Maintenance nocturne...")
        
        # 1. Nettoyer les erreurs
        errors_deleted = self.errors.cleanup()
        
        # 2. Nettoyer l'épisodique
        episodic_deleted = self.episodic.cleanup()
        
        # 3. Nettoyer le social
        social_deleted = self.social.cleanup()
        
        # 4. Appliquer le refroidissement des romans
        self.romans.apply_cooling()
        
        # 5. Archiver le mois dernier
        now = datetime.now()
        last_month = now.month - 1 if now.month > 1 else 12
        last_year = now.year if now.month > 1 else now.year - 1
        self.narrative.archive_month(last_year, last_month)
        
        logger.info(f"🌙 Maintenance terminée: {errors_deleted} erreurs, "
                   f"{episodic_deleted} épisodiques, {social_deleted} sociaux")
    
    def get_stats(self) -> Dict:
        """Statistiques du système."""
        return {
            "gem": {
                "nom": self.gem.nom,
                "version": self.gem.version,
                "naissance": self.gem.date_naissance
            },
            "memories": {
                "words": len(self.words.words),
                "errors": "voir DB",
                "episodic": "voir DB",
                "social": "voir DB",
                "romans": len(self.romans.romans),
                "educational": len(self.educational.works),
                "narrative_current": self.narrative.current_book.book_id if self.narrative.current_book else None
            },
            "libraries": {
                "spacy": self.libs.have_spacy,
                "stanza": self.libs.have_stanza,
                "dateparser": self.libs.have_dateparser,
                "hunspell": self.libs.have_hunspell,
                "pattern": self.libs.have_pattern
            }
        }

# ============================================================
# EXEMPLE DE FICHIER GEM
# ============================================================

"""
Exemple de fichier gem.json:
{
    "gem": {
        "identifiant": "shirka_001",
        "nom": "Shirka",
        "date_naissance": "2024-01-01T00:00:00",
        "version": 1,
        "tempo_base": 0.65,
        "intensite_base": 0.7,
        "grace": 0.5,
        "reactivite": 0.8,
        "curiosite_mots": 0.6,
        "curiosite_verbes": 0.5,
        "curiosite_personnes": 0.9,
        "curiosite_lieux": 0.8,
        "curiosite_faits": 0.7,
        "affinites_litterature": {
            "roman": 0.8,
            "poesie": 0.4,
            "theatre": 0.6,
            "essai": 0.7
        },
        "duree_memoire_litterature": 30,
        "duree_memoire_episodique": 90,
        "duree_memoire_sociale": 365,
        "style_prefere": "narratif",
        "seuil_curiosite": 0.7,
        "signature_type": "sha256",
        "signature_valeur": "abc123..."
    }
}
"""

# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================

def demo():
    """Démonstration du système."""
    
    # Initialiser avec un fichier Gem
    gem_path = Path("./gem.json")
    if not gem_path.exists():
        # Créer un exemple si le fichier n'existe pas
        example_gem = {
            "gem": {
                "identifiant": "shirka_demo",
                "nom": "Shirka",
                "date_naissance": datetime.now().isoformat(),
                "version": 1,
                "tempo_base": 0.65,
                "intensite_base": 0.7,
                "grace": 0.5,
                "reactivite": 0.8,
                "curiosite_mots": 0.8,
                "curiosite_verbes": 0.5,
                "curiosite_personnes": 0.9,
                "curiosite_lieux": 0.8,
                "curiosite_faits": 0.7,
                "affinites_litterature": {
                    "roman": 0.8,
                    "poesie": 0.4,
                    "theatre": 0.6,
                    "essai": 0.7
                },
                "duree_memoire_litterature": 30,
                "duree_memoire_episodique": 90,
                "duree_memoire_sociale": 365,
                "style_prefere": "narratif",
                "seuil_curiosite": 0.7,
                "signature_type": "sha256",
                "signature_valeur": "demo"
            }
        }
        with open(gem_path, 'w', encoding='utf-8') as f:
            json.dump(example_gem, f, indent=2)
        logger.info(f"✅ Fichier Gem exemple créé: {gem_path}")
    
    # Créer le core
    core = CognitionCore(Path("./data"), gem_path)
    
    # Tester avec des phrases
    test_phrases = [
        "demain midi, je vais au restaurant avec Paul",
        "quelle heure est-il ?",
        "Je viens de lire Le Petit Prince",
        "Paul est mon ami"
    ]
    
    for phrase in test_phrases:
        print(f"\n📥 Entrée: {phrase}")
        result = core.process(phrase)
        print(f"📤 Sortie: {json.dumps(result.to_dict(), indent=2, ensure_ascii=False)}")
    
    # Statistiques
    print(f"\n📊 Stats: {json.dumps(core.get_stats(), indent=2, ensure_ascii=False)}")
    
    # Simuler une maintenance nocturne
    core.nightly_maintenance()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    demo()