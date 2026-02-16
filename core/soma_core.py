import psutil
import time
import json
import zenoh
import logging
import threading
import signal
import sys
import os
import collections
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime, timezone
import subprocess
import statistics

# --- Ajout pour le PubScheduler ---
from pub_scheduler import PubScheduler

# --- Configuration des logs ---
logger = logging.getLogger("SomaCore")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
logger.addHandler(handler)

# --- Spécifique Windows pour WMI ---
IF_WINDOWS = os.name == 'nt'
if IF_WINDOWS:
    try:
        import pythoncom
        import wmi
    except ImportError:
        pass

# --- Constantes ---
WINDOW_SIZE = 20
SALVE_SIZE = 3
WARMUP_SAMPLES = 10
LIFE_NOISE_PERIOD = 1.0
FREQ_MIN = 1.0
FREQ_MAX = 200.0
PERIODE_VALIDITE = 1.0
PRECISION_DECIMALES = 2

# Constantes pour la gestion des capteurs
BOOTSTRAP_LECTURES = 5
SEUIL_LENTEUR = 3.0
SEUIL_VARIABILITE = 0.5
SEUIL_TIMEOUT_WARNING = 0.8
FREQ_MIN_ABSOLUE = 0.1
EXCEPTIONS_CONSECUTIVES_MAX = 3
FACTEUR_TIMEOUT_NORMAL = 1.5
FACTEUR_TIMEOUT_INSTABLE = 2.5

# Fichier d'override
OVERRIDE_FILE = "capteurs_override.json"

# Topic santé (diagnostic)
HEALTH_TOPIC = "soma/health"
HEALTH_FREQ = 1.0  # 1 Hz

# Noms des zones pour les logs
ZONE_NAMES = {
    'lt_plateau': '❄️ CRITIQUE FROID',
    'lt_exp2': '🥶 Froid extrême',
    'lt_exp1': '😨 Froid modéré',
    'lt_log': '😐 Froid léger',
    'comfort': '😊 Confort',
    'gt_log': '😐 Chaud léger',
    'gt_exp1': '😓 Chaud modéré',
    'gt_exp2': '🥵 Chaud extrême',
    'gt_plateau': '🔥 CRITIQUE CHAUD'
}

@dataclass
class SamplingProfile:
    name: str
    frequency: float
    description: str = ""

@dataclass
class CapteurConfig:
    """Configuration dynamique d'un capteur"""
    nom: str
    nerf_alias: str
    profil_origine: str
    freq_cible: float
    freq_effective: float
    periode_effective: float
    temps_lecture_max: float
    temps_lecture_moyen: float
    variabilite: float
    timeout: float
    seuil_max: float
    seuil_min: Optional[float] = None
    degrade: bool = False
    instable: bool = False
    suspendu: bool = False
    en_douleur: bool = False
    exceptions_consecutives: int = 0
    raison_suspension: str = ""
    override: bool = False

@dataclass
class ProfilAcquisition:
    """Profil d'acquisition avec sa liste de capteurs"""
    nom: str
    freq_cible: float
    capteurs: List[str] = field(default_factory=list)
    freq_effective: float = 0.0
    temps_max: float = 0.0

@dataclass
class Zone:
    """Représente une zone de la courbe d'accélération"""
    name: str
    start: float
    end: float
    curve_type: str
    f_start: float
    f_end: float
    metric_name: str = ""
    
    def contains(self, value: float) -> bool:
        if self.curve_type == 'comfort':
            return self.start <= value <= self.end
        elif 'lt' in self.name:
            if self.name == 'lt_log':
                return value > self.end
            elif self.name == 'lt_exp1':
                return self.end < value <= self.start
            elif self.name == 'lt_exp2':
                return self.end < value <= self.start
            elif self.name == 'lt_plateau':
                return value <= self.start
        else:
            if self.name == 'gt_log':
                return 0 <= value < self.end
            elif self.name in ['gt_exp1', 'gt_exp2']:
                return self.start <= value < self.end
            elif self.name == 'gt_plateau':
                return value >= self.start
        return False
    
    def compute_frequency(self, value: float) -> float:
        if self.curve_type == 'plateau' or self.curve_type == 'comfort':
            return self.f_start
        
        if 'lt' in self.name:
            if self.name == 'lt_log':
                pos = 1.0
            elif self.name == 'lt_exp1':
                pos = (self.start - value) / (self.start - self.end)
            elif self.name == 'lt_exp2':
                pos = (self.start - value) / (self.start - self.end)
            else:
                pos = 1.0
        else:
            if self.name == 'gt_log':
                pos = value / self.end
            else:
                pos = (value - self.start) / (self.end - self.start)
        
        pos = max(0.0, min(1.0, pos))
        
        if self.curve_type == 'log':
            if pos <= 0:
                return self.f_start
            return self.f_start + math.log1p(pos) / math.log(2) * (self.f_end - self.f_start)
        
        elif self.curve_type == 'exp':
            if pos <= 0:
                return self.f_start
            return self.f_start + (math.exp(pos) - 1) / (math.e - 1) * (self.f_end - self.f_start)
        
        elif self.curve_type == 'exp_aggressive':
            if pos <= 0:
                return self.f_start
            aggro_factor = 2.5
            exp_factor = math.exp(aggro_factor) - 1
            return self.f_start + (math.exp(pos * aggro_factor) - 1) / exp_factor * (self.f_end - self.f_start)
        
        return self.f_start

@dataclass
class AlertRule:
    name: str
    alias: str
    flux_topic: str
    gt: Optional[List[float]] = None
    lt: Optional[List[float]] = None
    sampling_profile: str = "slow"
    output_freq_min: float = FREQ_MIN
    output_freq_max: float = FREQ_MAX
    absolute_delta_threshold: float = 0.0
    aggro_factor: float = 2.5

class BatteryMonitor:
    """Moniteur de batterie avec estimation en minutes"""
    def __init__(self, battery_capacity_wh: float = 50.0):
        self.battery_capacity_wh = battery_capacity_wh
        self.history = collections.deque(maxlen=60)
        self.last_level = None
        self.last_timestamp = None
        
    def update(self, level: float, charging: bool, timestamp: float) -> Dict:
        self.history.append((timestamp, level, charging))
        
        result = {
            "level": round(level, 1),
            "charging": charging,
            "timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "charge_rate": 0.0,
            "discharge_rate": 0.0,
            "minutes_to_full": None,
            "minutes_to_empty": None,
            "confidence": "low"
        }
        
        if len(self.history) < 2:
            return result
        
        window_size = min(len(self.history), max(5, len(self.history) // 3))
        recent = list(self.history)[-window_size:]
        
        if window_size >= 30:
            result["confidence"] = "high"
        elif window_size >= 15:
            result["confidence"] = "medium"
        
        if len(recent) >= 2:
            dt = recent[-1][0] - recent[0][0]
            if dt > 0:
                dlevel = recent[-1][1] - recent[0][1]
                rate_per_second = dlevel / dt
                rate_per_minute = rate_per_second * 60
                
                if charging and rate_per_second > 0:
                    result["charge_rate"] = round(rate_per_minute, 2)
                    remaining_to_full = 100.0 - level
                    if rate_per_second > 0:
                        minutes_to_full = remaining_to_full / rate_per_minute
                        result["minutes_to_full"] = round(minutes_to_full, 1)
                
                elif not charging and rate_per_second < 0:
                    result["discharge_rate"] = round(abs(rate_per_minute), 2)
                    if rate_per_second < 0:
                        minutes_to_empty = level / abs(rate_per_minute)
                        result["minutes_to_empty"] = round(minutes_to_empty, 1)
        
        self.last_level = level
        self.last_timestamp = timestamp
        return result
    
    def reset(self):
        self.history.clear()
        self.last_level = None
        self.last_timestamp = None
        logger.debug("BatteryMonitor réinitialisé")

class FrequencyMapper:
    """Mappe les valeurs vers des fréquences selon une courbe d'accélération"""
    def __init__(self, rule: AlertRule):
        self.rule = rule
        self.f_min = rule.output_freq_min
        self.f_max = rule.output_freq_max
        self.f_low = self.f_min + (self.f_max - self.f_min) / 3
        self.f_mid = self.f_min + 2 * (self.f_max - self.f_min) / 3
        
        self.zones: List[Zone] = []
        self._build_zones()
        self._cache: Dict[float, Tuple[float, str]] = {}
        
        logger.debug(f"📊 Mapper créé pour {rule.name}: {len(self.zones)} zones")
    
    def _build_zones(self):
        rule = self.rule
        comfort_start = float('-inf')
        comfort_end = float('inf')
        
        if rule.gt and len(rule.gt) >= 3:
            s1, s2, s3 = rule.gt
            if not (s1 < s2 < s3):
                s1, s2, s3 = sorted(rule.gt)
            
            if s1 > 0:
                self.zones.append(Zone("gt_log", 0.0, s1, "log", self.f_min, self.f_low, rule.name))
            self.zones.append(Zone("gt_exp1", s1, s2, "exp", self.f_low, self.f_mid, rule.name))
            self.zones.append(Zone("gt_exp2", s2, s3, "exp_aggressive", self.f_mid, self.f_max, rule.name))
            self.zones.append(Zone("gt_plateau", s3, float('inf'), "plateau", self.f_max, self.f_max, rule.name))
            comfort_end = s1
        
        if rule.lt and len(rule.lt) >= 3:
            l3, l2, l1 = rule.lt
            if not (l3 > l2 > l1):
                l3, l2, l1 = sorted(rule.lt, reverse=True)
            
            self.zones.append(Zone("lt_plateau", l1, float('-inf'), "plateau", self.f_max, self.f_max, rule.name))
            self.zones.append(Zone("lt_exp2", l2, l1, "exp_aggressive", self.f_mid, self.f_max, rule.name))
            self.zones.append(Zone("lt_exp1", l3, l2, "exp", self.f_low, self.f_mid, rule.name))
            self.zones.append(Zone("lt_log", float('inf'), l3, "log", self.f_min, self.f_low, rule.name))
            comfort_start = l3
        
        if comfort_start != float('-inf') or comfort_end != float('inf'):
            self.zones.append(Zone("comfort", comfort_start, comfort_end, "comfort", self.f_min, self.f_min, rule.name))
    
    def get_frequency(self, value: float) -> Tuple[float, str]:
        rounded_val = round(value, PRECISION_DECIMALES)
        if rounded_val in self._cache:
            return self._cache[rounded_val]
        
        for zone in self.zones:
            if zone.contains(value):
                freq = round(zone.compute_frequency(value), 3)
                self._cache[rounded_val] = (freq, zone.name)
                return freq, zone.name
        
        return self.f_min, "unknown"
    
    def get_period_ms(self, value: float) -> Tuple[float, str]:
        freq, zone_name = self.get_frequency(value)
        period_ms = 1000.0 / freq if freq > 0 else 1000.0
        rounded_ms = max(10, round(period_ms / 10) * 10)
        return rounded_ms / 1000.0, zone_name
    
    def reset(self):
        self._cache.clear()
        logger.debug(f"FrequencyMapper pour {self.rule.name} réinitialisé")

class MetricCollector:
    def __init__(self, nom: str):
        self.nom = nom
        self.supported_metrics = set()
    
    def can_collect(self, metric_name: str) -> bool:
        return metric_name in self.supported_metrics
    
    def collect(self, metric_names: List[str]) -> Dict[str, float]:
        return {}

class SystemMetricCollector(MetricCollector):
    def __init__(self):
        super().__init__("system")
        self.supported_metrics = {"cpu", "memory", "temperature", "energy"}
        self._wmi_interface = None
        if IF_WINDOWS:
            try:
                pythoncom.CoInitialize()
                self._wmi_interface = wmi.WMI(namespace="root\\wmi")
            except:
                pass
    
    def _get_temperature(self) -> float:
        if IF_WINDOWS and self._wmi_interface:
            try:
                cmd = "Get-WmiObject -Namespace root/wmi -Class MSAcpi_ThermalZoneTemperature | Select-Object -ExpandProperty CurrentTemperature"
                output = subprocess.check_output(
                    ["powershell", "-Command", cmd],
                    stderr=subprocess.DEVNULL,
                    shell=True
                ).decode().strip()
                if output:
                    raw_k = float(output.split()[0])
                    return round((raw_k / 10.0) - 273.15, 2)
            except Exception:
                pass
        else:
            try:
                temps = psutil.sensors_temperatures()
                if temps and list(temps.values()):
                    return list(temps.values())[0][0].current
            except Exception:
                pass
        return 20.0
    
    def collect(self, metric_names: List[str]) -> Dict[str, float]:
        metrics = {}
        
        if "cpu" in metric_names:
            metrics["cpu"] = psutil.cpu_percent(interval=None)
        
        if "memory" in metric_names:
            metrics["memory"] = psutil.virtual_memory().percent
        
        if "temperature" in metric_names:
            metrics["temperature"] = self._get_temperature()
        
        if "energy" in metric_names:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    metrics["energy"] = battery.percent
                    metrics["_energy_charging"] = battery.power_plugged
                else:
                    metrics["energy"] = 100.0
                    metrics["_energy_charging"] = False
            except:
                metrics["energy"] = 100.0
                metrics["_energy_charging"] = False
        
        return metrics

class IMUMetricCollector(MetricCollector):
    def __init__(self):
        super().__init__("imu")
        self.supported_metrics = {
            "imu_pitch_forward", "imu_pitch_backward",
            "imu_roll_left", "imu_roll_right",
            "imu_yaw_left", "imu_yaw_right",
            "imu_front_accel", "imu_back_accel"
        }

    def collect(self, metric_names: List[str]) -> Dict[str, float]:
        metrics = {}
        for name in metric_names:
            if name in self.supported_metrics:
                metrics[name] = 0.0
        return metrics

class MetricNerve:
    def __init__(self, rule: AlertRule, battery_monitor: Optional[BatteryMonitor] = None):
        self.rule = rule
        self.battery_monitor = battery_monitor
        self.history = collections.deque(maxlen=WINDOW_SIZE)
        self.timestamps = collections.deque(maxlen=WINDOW_SIZE)
        self.last_published_value = None
        self.warmup_complete = False
        self.current_stress = 0.0
        self.current_period = PERIODE_VALIDITE
        self.frequency_mapper = FrequencyMapper(rule)
        self._new_data_event = threading.Event()
        self._lock = threading.Lock()
        self.last_freq = rule.output_freq_min
        self.last_zone = "unknown"
        self.last_battery_info = None

    def push(self, val: float, timestamp: float, extra_data: Dict = None):
        with self._lock:
            self.history.append(val)
            self.timestamps.append(timestamp)
            if extra_data:
                self.last_battery_info = extra_data
            if not self.warmup_complete and len(self.history) >= WARMUP_SAMPLES:
                self.warmup_complete = True
            self._new_data_event.set()

    def wait_for_data(self, timeout=1.0):
        return self._new_data_event.wait(timeout)
    
    def clear_event(self):
        self._new_data_event.clear()

    def get_latest(self):
        with self._lock:
            if len(self.history) > 0:
                return self.history[-1], self.timestamps[-1], self.last_battery_info
            return None, None, None

    def get_stress(self, val: float, charging: bool = False) -> float:
        x = 0.0
        if self.rule.gt and len(self.rule.gt) >= 3 and val > self.rule.gt[0]:
            x = min(1.0, max(0.0, (val - self.rule.gt[0]) / (self.rule.gt[-1] - self.rule.gt[0])))
        elif self.rule.lt and len(self.rule.lt) >= 3 and val < self.rule.lt[0]:
            if not (self.rule.name == "energy" and charging):
                x = min(1.0, max(0.0, (self.rule.lt[0] - val) / (self.rule.lt[0] - self.rule.lt[-1])))
        self.current_stress = x
        return x
    
    def get_period_for_value(self, value: float) -> Tuple[float, str]:
        return self.frequency_mapper.get_period_ms(value)
    
    def reset(self):
        with self._lock:
            self.history.clear()
            self.timestamps.clear()
            self.warmup_complete = False
            self.current_stress = 0.0
            self.current_period = PERIODE_VALIDITE
            self.last_battery_info = None
            self.frequency_mapper.reset()
            self._new_data_event.clear()
        logger.debug(f"Nerf {self.rule.name} réinitialisé")

class HealthMonitor:
    """Moniteur de santé publiant sur /soma/health à 1 Hz"""
    
    def __init__(self, session: zenoh.Session, capteurs_config: Dict[str, CapteurConfig]):
        self.session = session
        self.capteurs_config = capteurs_config
        self.pub = session.declare_publisher(HEALTH_TOPIC)
        self.running = True
        self._lock = threading.Lock()
        
    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        logger.info("✅ HealthMonitor démarré (1 Hz)")
        
    def _run(self):
        while self.running:
            try:
                rapport = self._generer_rapport()
                self.pub.put(json.dumps(rapport))
            except Exception as e:
                logger.error(f"Erreur HealthMonitor: {e}")
            time.sleep(1.0 / HEALTH_FREQ)
    
    def _generer_rapport(self) -> Dict:
        with self._lock:
            rapport = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "capteurs": {}
            }
            
            for nom, cfg in self.capteurs_config.items():
                etat = "OK"
                if cfg.en_douleur:
                    etat = "DOULEUR"
                elif cfg.suspendu:
                    etat = "SUSPENDU"
                elif cfg.degrade:
                    etat = "DEGRADE"
                elif cfg.instable:
                    etat = "INSTABLE"
                
                rapport["capteurs"][nom] = {
                    "etat": etat,
                    "freq_effective": round(cfg.freq_effective, 2),
                    "temps_max_ms": round(cfg.temps_lecture_max * 1000, 1),
                    "variabilite": round(cfg.variabilite, 2),
                    "raison": cfg.raison_suspension if cfg.suspendu else ""
                }
            
            # Statistiques globales
            total = len(self.capteurs_config)
            douleur = sum(1 for c in self.capteurs_config.values() if c.en_douleur)
            suspendus = sum(1 for c in self.capteurs_config.values() if c.suspendu)
            degrades = sum(1 for c in self.capteurs_config.values() if c.degrade)
            
            rapport["global"] = {
                "total": total,
                "douleur": douleur,
                "suspendus": suspendus,
                "degrades": degrades,
                "sante": round((total - douleur - suspendus) / total * 100, 1) if total > 0 else 100
            }
            
            return rapport
    
    def stop(self):
        self.running = False

class SomaCore:
    def __init__(self, zenoh_config: str, config_path: str):
        self.running = True
        self.zenoh_config = zenoh_config
        self.config_path = config_path
        
        # Structure pour les profils et capteurs
        self.profils: Dict[str, ProfilAcquisition] = {}
        self.capteurs_config: Dict[str, CapteurConfig] = {}
        self.capteurs_actifs: List[str] = []
        
        # Charger les overrides existants
        self.overrides = self._charger_overrides()
        
        # Initialisation Zenoh
        self._init_zenoh()
        
        self._clock_lock = threading.Lock()
        self._salve_times = []
        self._phase_sync_event = threading.Event()
        self._charging = False
        self._last_charging = False
        self._temperature = 20.0
        
        self._battery_monitor = BatteryMonitor()

        # Charger la configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # Charger les profils d'échantillonnage
        self.sampling_profiles = {
            name: SamplingProfile(name=name, **profile_data)
            for name, profile_data in config["sampling_profiles"].items()
        }
        self.default_sampling_profile = config["default_sampling_profile"]
        logger.info(f"📊 Profils: {list(self.sampling_profiles.keys())}")

        # Créer les AlertRule
        self.rules = []
        for metric_name, metric_config in config["metrics"].items():
            if metric_name == "ego":
                continue
                
            if "output_freq_min" in metric_config:
                metric_config["output_freq_min"] = max(1.0, metric_config["output_freq_min"])
            
            rule = AlertRule(
                name=metric_name,
                alias=metric_config["alias"],
                flux_topic=metric_config["flux_topic"],
                gt=metric_config.get("threshold_GT"),
                lt=metric_config.get("threshold_LT"),
                sampling_profile=metric_config.get("sampling_profile", self.default_sampling_profile),
                output_freq_min=metric_config.get("output_freq_min", FREQ_MIN),
                output_freq_max=metric_config.get("output_freq_max", FREQ_MAX),
                absolute_delta_threshold=metric_config.get("absolute_delta_threshold", 0.0),
                aggro_factor=metric_config.get("aggro_factor", 2.5)
            )
            self.rules.append(rule)
            logger.debug(f"Règle: {metric_name} (profil={rule.sampling_profile})")

        # Initialiser les collecteurs
        self.collectors = [SystemMetricCollector(), IMUMetricCollector()]

        # Initialiser le scheduler et les nerfs
        self._init_scheduler_and_nerves()
        
        # Lancer le bootstrap des capteurs
        self._bootstrap_capteurs()
        
        # Démarrer le HealthMonitor
        self.health_monitor = HealthMonitor(self.session, self.capteurs_config)
        self.health_monitor.start()
        
        # S'abonner au topic de reset
        self._subscribe_to_reset()

        # Démarrer les threads d'acquisition
        self._demarrer_acquisition()

        # Démarrer les threads de traitement
        for metric_name in self.nerves.keys():
            threading.Thread(
                target=self._nerve_process_loop,
                args=(metric_name,),
                daemon=True
            ).start()

        logger.info("⏳ Warmup...")
        self._warmup_phase()
        
        signal.signal(signal.SIGINT, lambda s, f: self.stop())
    
    def _charger_overrides(self) -> Dict:
        try:
            if os.path.exists(OVERRIDE_FILE):
                with open(OVERRIDE_FILE, 'r') as f:
                    overrides = json.load(f)
                logger.info(f"📁 Overrides chargés: {len(overrides)} capteurs")
                return overrides
        except Exception as e:
            logger.error(f"Erreur chargement overrides: {e}")
        return {}
    
    def _sauvegarder_overrides(self):
        try:
            with open(OVERRIDE_FILE, 'w') as f:
                json.dump(self.overrides, f, indent=2)
            logger.debug("📁 Overrides sauvegardés")
        except Exception as e:
            logger.error(f"Erreur sauvegarde overrides: {e}")
    
    def _calculer_seuil_max(self, rule: AlertRule) -> float:
        """Calcule la valeur maximale de seuil (la plus critique)"""
        if rule.gt:
            return rule.gt[-1]
        elif rule.lt:
            return rule.lt[-1]
        return 1.0
    
    def _calculer_seuil_min(self, rule: AlertRule) -> Optional[float]:
        """Calcule la valeur minimale de seuil (la plus critique pour LT)"""
        if rule.lt:
            return rule.lt[0]
        return None
    
    def _bootstrap_capteurs(self):
        logger.info("🔧 Démarrage du bootstrap des capteurs...")
        
        # Étape 1 : Regrouper les capteurs par profil d'origine
        capteurs_par_profil = {}
        for rule in self.rules:
            profil = rule.sampling_profile
            if profil not in capteurs_par_profil:
                capteurs_par_profil[profil] = []
            capteurs_par_profil[profil].append((rule.name, rule))
        
        # Étape 2 : Pour chaque profil, mesurer les temps des capteurs
        for nom_profil, capteurs in capteurs_par_profil.items():
            profil_config = self.sampling_profiles[nom_profil]
            freq_cible = profil_config.frequency
            periode_cible = 1.0 / freq_cible
            
            logger.info(f"📊 Bootstrap du profil {nom_profil} ({freq_cible} Hz)")
            
            temps_par_capteur = {}
            for capteur_nom, rule in capteurs:
                collecteur = next((c for c in self.collectors if c.can_collect(capteur_nom)), None)
                if not collecteur:
                    logger.warning(f"⚠️ Aucun collecteur pour {capteur_nom}")
                    continue
                
                temps_lectures = []
                try:
                    for i in range(BOOTSTRAP_LECTURES):
                        debut = time.perf_counter()
                        val = collecteur.collect([capteur_nom]).get(capteur_nom, 0.0)
                        duree = time.perf_counter() - debut
                        temps_lectures.append(duree)
                        time.sleep(periode_cible * 0.1)
                    
                    t_min = min(temps_lectures)
                    t_max = max(temps_lectures)
                    t_moyen = sum(temps_lectures) / len(temps_lectures)
                    variabilite = (t_max - t_min) / t_moyen if t_moyen > 0 else 0
                    
                    temps_par_capteur[capteur_nom] = {
                        'min': t_min,
                        'max': t_max,
                        'moyen': t_moyen,
                        'variabilite': variabilite,
                        'lectures': temps_lectures,
                        'rule': rule
                    }
                    
                    logger.debug(f"  {capteur_nom}: min={t_min*1000:.2f}ms, "
                               f"max={t_max*1000:.2f}ms, "
                               f"var={variabilite:.2f}")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur bootstrap {capteur_nom}: {e}")
                    self._gerer_exception_capteur(capteur_nom, str(e))
            
            if not temps_par_capteur:
                continue
            
            temps_max_liste = [t['max'] for t in temps_par_capteur.values()]
            mediane_max = statistics.median(temps_max_liste) if temps_max_liste else 0
            
            for capteur_nom, stats in temps_par_capteur.items():
                rule = stats['rule']
                t_max = stats['max']
                variabilite = stats['variabilite']
                
                if capteur_nom in self.overrides:
                    logger.info(f"  ↪ {capteur_nom} utilise override: {self.overrides[capteur_nom]}")
                    continue
                
                est_lent = t_max > mediane_max * SEUIL_LENTEUR
                est_instable = variabilite > SEUIL_VARIABILITE
                
                freq_max_possible = 1.0 / t_max if t_max > 0 else freq_cible
                
                if est_lent or est_instable:
                    facteur_timeout = FACTEUR_TIMEOUT_INSTABLE if est_instable else FACTEUR_TIMEOUT_NORMAL
                    timeout = t_max * facteur_timeout
                    
                    freq_effective = min(freq_cible, freq_max_possible)
                    if freq_effective < FREQ_MIN_ABSOLUE:
                        freq_effective = FREQ_MIN_ABSOLUE
                        logger.warning(f"⚠️ {capteur_nom} fréquence très basse: {freq_effective} Hz")
                    
                    self.capteurs_config[capteur_nom] = CapteurConfig(
                        nom=capteur_nom,
                        nerf_alias=rule.alias,
                        profil_origine=nom_profil,
                        freq_cible=freq_cible,
                        freq_effective=freq_effective,
                        periode_effective=1.0/freq_effective,
                        temps_lecture_max=t_max,
                        temps_lecture_moyen=stats['moyen'],
                        variabilite=variabilite,
                        timeout=timeout,
                        seuil_max=self._calculer_seuil_max(rule),
                        seuil_min=self._calculer_seuil_min(rule),
                        degrade=est_lent,
                        instable=est_instable
                    )
                    
                    logger.warning(f"⚠️ Capteur {capteur_nom}: lent={est_lent}, instable={est_instable}")
                    logger.info(f"   → Profil dédié à {freq_effective:.2f} Hz")
                    
                else:
                    self.capteurs_config[capteur_nom] = CapteurConfig(
                        nom=capteur_nom,
                        nerf_alias=rule.alias,
                        profil_origine=nom_profil,
                        freq_cible=freq_cible,
                        freq_effective=freq_cible,
                        periode_effective=1.0/freq_cible,
                        temps_lecture_max=t_max,
                        temps_lecture_moyen=stats['moyen'],
                        variabilite=variabilite,
                        timeout=t_max * FACTEUR_TIMEOUT_NORMAL,
                        seuil_max=self._calculer_seuil_max(rule),
                        seuil_min=self._calculer_seuil_min(rule)
                    )
        
        self.capteurs_actifs = list(self.capteurs_config.keys())
        logger.info(f"✅ Bootstrap terminé: {len(self.capteurs_actifs)} capteurs actifs")
    
    def _activer_canal_douleur(self, capteur: str, raison: str):
        """Active le canal douleur pour un capteur suspendu"""
        config = self.capteurs_config.get(capteur)
        if not config:
            return
        
        config.suspendu = True
        config.en_douleur = True
        config.raison_suspension = raison
        
        # Arrêter les publications sur le nerf normal
        self.scheduler.remove_nerf(config.nerf_alias)
        
        # Créer le canal douleur
        pain_alias = f"pain_{capteur}"
        pain_topic = f"pain/{capteur}"
        
        # Ajouter au scheduler avec fréquence max
        self.scheduler.add_nerf(pain_alias, 1.0 / config.freq_cible)
        
        # Publier la valeur max en continu
        self.scheduler.update_payload(pain_alias, {
            "v": config.seuil_max,
            "raison": raison,
            "default": True,
            "capteur": capteur,
            "ts": time.time()
        })
        
        logger.critical(f"🚨 CANAL DOULEUR activé pour {capteur}: {raison}")
    
    def _gerer_exception_capteur(self, capteur: str, erreur: str):
        if capteur not in self.capteurs_config:
            return
        
        config = self.capteurs_config[capteur]
        config.exceptions_consecutives += 1
        
        logger.error(f"❌ Exception {capteur} (#{config.exceptions_consecutives}): {erreur}")
        
        if config.exceptions_consecutives >= EXCEPTIONS_CONSECUTIVES_MAX:
            self._activer_canal_douleur(capteur, f"exceptions ({EXCEPTIONS_CONSECUTIVES_MAX})")
    
    def _surveiller_temps_lecture(self, capteur: str, duree: float):
        if capteur not in self.capteurs_config:
            return
        
        config = self.capteurs_config[capteur]
        periode = config.periode_effective
        
        if duree > periode * SEUIL_TIMEOUT_WARNING:
            logger.warning(f"⚠️ {capteur}: lecture lente ({duree*1000:.2f}ms > {periode*1000:.1f}ms)")
        
        if duree > periode:
            logger.error(f"⛔ {capteur}: TIMEOUT ({duree*1000:.2f}ms > {periode*1000:.1f}ms)")
            
            # Réduire la fréquence par 2
            nouvelle_freq = config.freq_effective / 2
            if nouvelle_freq < FREQ_MIN_ABSOLUE:
                # On a atteint la fréquence minimale, on passe en douleur
                self._activer_canal_douleur(capteur, "timeout critique")
                return
            
            ancienne_freq = config.freq_effective
            config.freq_effective = nouvelle_freq
            config.periode_effective = 1.0 / nouvelle_freq
            
            logger.warning(f"🔄 {capteur}: réduction fréquence {ancienne_freq:.2f} → {nouvelle_freq:.2f} Hz")
            
            self.overrides[capteur] = {
                "freq_forcee": nouvelle_freq,
                "raison": "timeout",
                "timestamp": time.time()
            }
            self._sauvegarder_overrides()
    
    def _init_zenoh(self):
        self.conf = zenoh.Config()
        self.conf.from_file(self.zenoh_config)
        self.session = zenoh.open(self.conf)
        logger.info("✅ Session Zenoh ouverte")
        
    def _close_zenoh(self):
        try:
            self.session.close()
            logger.info("✅ Session Zenoh fermée")
        except Exception as e:
            logger.error(f"Erreur fermeture session Zenoh: {e}")
    
    def _init_scheduler_and_nerves(self):
        def publish_callback(alias: str, payload: dict):
            for rule in self.rules:
                if rule.alias == alias:
                    try:
                        pub = self.session.declare_publisher(rule.flux_topic)
                        pub.put(json.dumps(payload))
                    except Exception as e:
                        logger.error(f"Erreur publication {alias}: {e}")
                    return
            # Si c'est un canal douleur (pain/...)
            if alias.startswith("pain_"):
                try:
                    capteur = alias.replace("pain_", "")
                    pub = self.session.declare_publisher(f"pain/{capteur}")
                    pub.put(json.dumps(payload))
                except Exception as e:
                    logger.error(f"Erreur publication douleur {alias}: {e}")
                return
            logger.warning(f"Aucun topic trouvé pour l'alias {alias}")

        self.scheduler = PubScheduler(
            publish_callback=publish_callback,
            base_period=0.01,
            name="SomaCore_Scheduler"
        )
        self.scheduler.start()

        self.nerves = {}
        for rule in self.rules:
            if rule.name == "energy":
                nerve = MetricNerve(rule, self._battery_monitor)
            else:
                nerve = MetricNerve(rule)
            
            self.nerves[rule.name] = nerve
            self.scheduler.add_nerf(rule.alias, 1.0 / rule.output_freq_min)
    
    def _subscribe_to_reset(self):
        try:
            self.reset_sub = self.session.declare_subscriber(
                "circa/reset", 
                self._on_reset_signal
            )
            logger.info("✅ Abonné à circa/reset")
        except Exception as e:
            logger.error(f"Erreur abonnement circa/reset: {e}")
    
    def _on_reset_signal(self, sample):
        try:
            payload = sample.payload.to_string()
            data = json.loads(payload)
            logger.warning(f"🔄 SIGNAL DE RESET REÇU: {data}")
            threading.Thread(target=self._perform_reset, daemon=True).start()
        except Exception as e:
            logger.error(f"Erreur traitement reset signal: {e}")
    
    def _perform_reset(self):
        logger.warning("🔄 DÉBUT DU RESET SomaCore")
        
        if hasattr(self, 'scheduler'):
            logger.info("Arrêt du scheduler...")
            self.scheduler.stop()
        
        logger.info("Réinitialisation des nerfs...")
        for nerve in self.nerves.values():
            nerve.reset()
        
        logger.info("Réinitialisation du BatteryMonitor...")
        self._battery_monitor.reset()
        
        self._charging = False
        self._last_charging = False
        self._temperature = 20.0
        
        logger.info("Fermeture de la session Zenoh...")
        self._close_zenoh()
        time.sleep(1.0)
        logger.info("Réouverture de la session Zenoh...")
        self._init_zenoh()
        
        # On garde les overrides mais on enlève les suspensions (on réessaie)
        self.overrides = {k: v for k, v in self.overrides.items() if not v.get("suspendu", False)}
        self._sauvegarder_overrides()
        
        self._bootstrap_capteurs()
        
        logger.info("Recréation du scheduler et des nerfs...")
        self._init_scheduler_and_nerves()
        
        self._subscribe_to_reset()
        
        self._demarrer_acquisition()
        
        # Redémarrer HealthMonitor
        if hasattr(self, 'health_monitor'):
            self.health_monitor.stop()
        self.health_monitor = HealthMonitor(self.session, self.capteurs_config)
        self.health_monitor.start()
        
        logger.info("Phase de warmup après reset...")
        self._warmup_phase()
        
        logger.warning("✅ RESET SomaCore TERMINÉ")
    
    def _demarrer_acquisition(self):
        capteurs_par_freq = {}
        for capteur in self.capteurs_actifs:
            if capteur not in self.capteurs_config:
                continue
            config = self.capteurs_config[capteur]
            if config.suspendu:
                continue  # Ne pas lancer de thread pour les capteurs suspendus
            freq = config.freq_effective
            if freq not in capteurs_par_freq:
                capteurs_par_freq[freq] = []
            capteurs_par_freq[freq].append(capteur)
        
        for freq, capteurs in capteurs_par_freq.items():
            logger.info(f"🔄 Acquisition [{freq} Hz]: {capteurs}")
            threading.Thread(
                target=self._acquisition_loop,
                args=(freq, capteurs),
                daemon=True
            ).start()
    
    def _acquisition_loop(self, frequency: float, capteurs: List[str]):
        period = 1.0 / frequency
        logger.info(f"▶️ Acquisition loop {frequency} Hz démarrée pour {len(capteurs)} capteurs")
        
        while self.running:
            start_time = time.time()
            current_time = start_time
            
            for capteur in capteurs:
                if capteur not in self.capteurs_actifs:
                    continue
                
                config = self.capteurs_config.get(capteur)
                if not config or config.suspendu:
                    continue
                
                collecteur = next((c for c in self.collectors if c.can_collect(capteur)), None)
                if not collecteur:
                    continue
                
                try:
                    debut_lecture = time.perf_counter()
                    
                    metrics = collecteur.collect([capteur])
                    valeur = metrics.get(capteur, 0.0)
                    
                    duree_lecture = time.perf_counter() - debut_lecture
                    
                    self._surveiller_temps_lecture(capteur, duree_lecture)
                    
                    if capteur == "energy" and "_energy_charging" in metrics:
                        charging = metrics["_energy_charging"]
                        battery_info = self._battery_monitor.update(valeur, charging, current_time)
                        if "energy" in self.nerves:
                            self.nerves["energy"].push(valeur, current_time, battery_info)
                        
                        if charging != self._last_charging:
                            self._charging = charging
                            self._last_charging = charging
                            logger.warning(f"⚡ Charging {'ACTIVÉ' if charging else 'DÉSACTIVÉ'} | Niveau: {valeur:.1f}%")
                        else:
                            self._charging = charging
                    
                    elif capteur in self.nerves:
                        self.nerves[capteur].push(valeur, current_time)
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lecture {capteur}: {e}")
                    self._gerer_exception_capteur(capteur, str(e))
            
            elapsed = time.time() - start_time
            sleep_time = max(0.0001, period - elapsed)
            time.sleep(sleep_time)

    def _nerve_process_loop(self, metric_name: str):
        nerve = self.nerves[metric_name]
        logger.info(f"🔧 Traitement [{metric_name}] démarré")
        
        while self.running:
            if not nerve.wait_for_data(timeout=2.0):
                continue
            
            nerve.clear_event()
            
            val, timestamp, extra_data = nerve.get_latest()
            if val is None or not nerve.warmup_complete:
                continue
            
            stress = nerve.get_stress(val, self._charging)
            period, zone_name = nerve.get_period_for_value(val)
            
            payload_data = {
                "v": round(val, 2),
                "stress": round(stress, 3),
                "ts": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                "freq": round(1.0 / period, 3),
                "period_ms": round(period * 1000, 1),
                "zone": zone_name
            }
            
            if metric_name == "energy" and extra_data:
                payload_data.update({
                    "charging": extra_data["charging"],
                    "level": extra_data["level"],
                    "charge_rate": extra_data["charge_rate"],
                    "discharge_rate": extra_data["discharge_rate"],
                    "minutes_to_full": extra_data["minutes_to_full"],
                    "minutes_to_empty": extra_data["minutes_to_empty"],
                    "battery_confidence": extra_data["confidence"]
                })
            
            alias = nerve.rule.alias
            self.scheduler.update_payload(alias, payload_data)
            
            current_period_ms = period * 1000
            last_period_ms = nerve.current_period * 1000 if hasattr(nerve, 'current_period') else 0
            
            if abs(current_period_ms - last_period_ms) > max(10, last_period_ms * 0.1):
                self.scheduler.update_period(alias, period)
                nerve.current_period = period
                
                old_freq = 1.0 / nerve.current_period if hasattr(nerve, 'current_period') else nerve.rule.output_freq_min
                new_freq = 1.0 / period
                
                try:
                    unit = ""
                    if metric_name == 'temperature':
                        unit = "°C"
                    elif metric_name == 'energy':
                        unit = "%"
                    
                    zone_display = ZONE_NAMES.get(zone_name, zone_name)
                    
                    logger.info(
                        f"🔄 FRÉQUENCE [{metric_name}] "
                        f"Valeur: {val}{unit} | "
                        f"Zone: {zone_display} | "
                        f"{old_freq:.2f}Hz → {new_freq:.2f}Hz "
                        f"({last_period_ms:.0f}ms → {current_period_ms:.0f}ms)"
                    )
                except Exception as e:
                    logger.debug(f"🔄 FRÉQUENCE {metric_name}: {old_freq:.2f}Hz → {new_freq:.2f}Hz")

    def _warmup_phase(self):
        logger.info("🔥 Warmup...")
        time.sleep(WARMUP_SAMPLES * 0.1)
        logger.info("✅ Warmup terminé")

    def stop(self):
        logger.info("🛑 Arrêt...")
        self.running = False
        
        if hasattr(self, 'health_monitor'):
            self.health_monitor.stop()
        
        if hasattr(self, 'scheduler'):
            self.scheduler.stop()
        
        self._close_zenoh()
        logger.info("✅ SomaCore éteint")

# ================== MAIN ==================
if __name__ == "__main__":
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        logger.info(f"SomaCore dir: {script_dir}")

        zenoh_config_file = "zenoh_config.json5"
        zenoh_filepath = os.path.join(script_dir, '..\\config', zenoh_config_file)
        if not os.path.exists(zenoh_filepath):
            logger.error(f"❌ Zenoh config manquante: {zenoh_filepath}")
            sys.exit(1)

        config_file = "soma_rules.json"
        config_filepath = os.path.join(script_dir, config_file)
        if not os.path.exists(config_filepath):
            logger.error(f"❌ Config manquante: {config_filepath}")
            sys.exit(1)
        
        core = SomaCore(zenoh_filepath, config_filepath)
        logger.info("✅ SomaCore démarré - Ctrl+C pour arrêter")
        
        while True:
            time.sleep(1)
            
    except (KeyboardInterrupt, SystemExit):
        logger.info("👋 Arrêt demandé")
    except Exception as e:
        logger.error(f"💥 Erreur: {e}", exc_info=True)