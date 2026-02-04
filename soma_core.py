import psutil
import time
import json
import zenoh
import logging
import threading
import signal
import os
import collections
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime, timezone
from scipy.stats import linregress

# --- Configuration des logs ---
logger = logging.getLogger("SomaCore")
logger.setLevel(logging.INFO)  # Changé à INFO pour réduire le flood
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
FREQ_MIN = 0.01
FREQ_MAX = 200.0
PERIODE_VALIDITE = 1.0
TEMP_SEUIL_BAS = 15
TEMP_SEUIL_HAUT = 30

# ===== BATTERY MANAGEMENT =====
class BatteryMonitor:
    """
    Moniteur de batterie avec estimation temps de charge/décharge
    L'estimation s'affine progressivement avec plus de données
    """
    def __init__(self, battery_capacity_wh: float = 50.0):
        self.battery_capacity_wh = battery_capacity_wh  # Capacité en Wh
        self.history = collections.deque(maxlen=20)  # Historique étendu pour meilleur affinement
        self.last_level = None
        self.last_timestamp = None
        
    def update(self, level: float, charging: bool, timestamp: float):
        """
        Met à jour l'état de la batterie et calcule les estimations
        
        L'estimation du temps restant s'affine au fur et à mesure :
        - 2-5 échantillons : estimation grossière
        - 5-10 échantillons : estimation moyenne
        - 10-20 échantillons : estimation précise
        
        Args:
            level: Niveau batterie (%)
            charging: État de charge
            timestamp: Timestamp actuel
        
        Returns:
            dict: Informations enrichies sur la batterie
        """
        # Ajouter à l'historique
        self.history.append((timestamp, level, charging))
        
        # Calculer le taux de variation
        charge_rate = 0.0  # %/s
        discharge_rate = 0.0  # %/s
        quality = "unknown"
        time_remaining = None
        confidence = "low"  # Niveau de confiance de l'estimation
        
        if len(self.history) >= 2:
            # Adapter la fenêtre selon le nombre d'échantillons disponibles
            # Plus on a de données, plus la fenêtre est grande = estimation stable
            window_size = min(len(self.history), max(5, len(self.history) // 2))
            recent = list(self.history)[-window_size:]
            
            # Déterminer la confiance selon le nombre d'échantillons
            if window_size >= 15:
                confidence = "high"
            elif window_size >= 8:
                confidence = "medium"
            else:
                confidence = "low"
            
            if len(recent) >= 2:
                dt = recent[-1][0] - recent[0][0]  # Delta temps
                dlevel = recent[-1][1] - recent[0][1]  # Delta niveau
                
                if dt > 0:
                    rate = dlevel / dt  # %/s
                    
                    if charging:
                        charge_rate = rate
                        
                        # Estimer temps de charge restant
                        if rate > 0:
                            remaining_percent = 100.0 - level
                            time_remaining = remaining_percent / rate  # secondes
                            
                            # Déterminer qualité de charge
                            charge_rate_per_hour = rate * 3600  # %/h
                            if charge_rate_per_hour > 50:
                                quality = "rapid"  # >50%/h
                            elif charge_rate_per_hour > 20:
                                quality = "fast"   # 20-50%/h
                            elif charge_rate_per_hour > 5:
                                quality = "normal" # 5-20%/h
                            else:
                                quality = "slow"   # <5%/h
                    else:
                        discharge_rate = -rate  # Positif pour décharge
                        
                        # Estimer temps de décharge restant
                        if rate < 0:
                            time_remaining = level / abs(rate)  # secondes
        
        self.last_level = level
        self.last_timestamp = timestamp
        
        return {
            "level": level,
            "charging": charging,
            "timestamp": timestamp,
            "charge_rate": charge_rate,  # %/s
            "discharge_rate": discharge_rate,  # %/s
            "time_remaining": time_remaining,  # secondes (None si indéterminé)
            "charge_quality": quality if charging else None,
            "confidence": confidence,  # Niveau de confiance de l'estimation
            "sample_count": len(self.history)  # Nombre d'échantillons utilisés
        }

@dataclass
class SamplingProfile:
    name: str
    frequency: float
    description: str = ""

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
    mixing_coefficients: Optional[Dict[str, float]] = None  # Pour ego
    normalization: str = "weighted_rms"  # "weighted_rms", "max", "mean"

class MetricCollector:
    """Collecteur de métriques de base"""
    def __init__(self):
        self.supported_metrics = set()
    
    def can_collect(self, metric_name: str) -> bool:
        """Vérifie si ce collecteur peut collecter cette métrique"""
        return metric_name in self.supported_metrics
    
    def collect(self, metric_names: List[str]) -> Dict[str, float]:
        """Collecte uniquement les métriques demandées"""
        return {}

class SystemMetricCollector(MetricCollector):
    def __init__(self):
        super().__init__()
        self.supported_metrics = {"cpu", "memory", "temperature", "energy", "charging", "energy_discharge"}
        self._wmi_interface = None
        if IF_WINDOWS:
            try:
                pythoncom.CoInitialize()
                self._wmi_interface = wmi.WMI(namespace="root\\wmi")
            except:
                pass
    
    def collect(self, metric_names: List[str]) -> Dict[str, float]:
        """Collecte uniquement les métriques système demandées"""
        metrics = {}
        
        # CPU
        if "cpu" in metric_names:
            metrics["cpu"] = psutil.cpu_percent(interval=None)
        
        # Memory
        if "memory" in metric_names:
            metrics["memory"] = psutil.virtual_memory().percent
        
        # Température
        if "temperature" in metric_names:
            temp = 0.0
            if IF_WINDOWS and self._wmi_interface:
                try:
                    t_data = self._wmi_interface.MSAcpi_ThermalZoneTemperature()
                    if t_data:
                        temp = (t_data[0].CurrentTemperature / 10.0) - 273.15
                except:
                    pass
            else:
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        temp = list(temps.values())[0][0].current
                except:
                    pass
            metrics["temperature"] = temp
        
        # Batterie (collectée même si pas demandée pour l'état charging)
        if "energy" in metric_names or "charging" in metric_names or "energy_discharge" in metric_names:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    if "energy" in metric_names:
                        metrics["energy"] = battery.percent
                    metrics["charging"] = battery.power_plugged
                    
                    # ✅ NOUVEAU : energy_discharge = taux de décharge
                    # Métrique virtuelle calculée depuis energy
                    if "energy_discharge" in metric_names:
                        # energy_discharge représente la "pression" de décharge
                        # C'est l'inverse du niveau de batterie
                        # Plus la batterie est basse, plus le stress est élevé
                        discharge_pressure = 100.0 - battery.percent
                        metrics["energy_discharge"] = discharge_pressure
            except:
                metrics["charging"] = False
                if "energy" in metric_names:
                    metrics["energy"] = 0.0
                if "energy_discharge" in metric_names:
                    metrics["energy_discharge"] = 100.0  # Max stress si pas de batterie
            
        return metrics

class IMUMetricCollector(MetricCollector):
    def __init__(self):
        super().__init__()
        self.supported_metrics = {
            "imu_pitch_forward", "imu_pitch_backward",
            "imu_roll_left", "imu_roll_right",
            "imu_yaw_left", "imu_yaw_right",
            "imu_front_accel", "imu_back_accel"
        }

    def collect(self, metric_names: List[str]) -> Dict[str, float]:
        """Collecte uniquement les métriques IMU demandées"""
        metrics = {}
        
        # TODO: Implémenter la vraie lecture IMU
        # Pour l'instant, retourne 0 pour toutes les métriques demandées
        for name in metric_names:
            if name in self.supported_metrics:
                metrics[name] = 0.0
        
        return metrics

class MetricNerve:
    def __init__(self, rule: AlertRule):
        self.rule = rule
        self.history = collections.deque(maxlen=WINDOW_SIZE)
        self.timestamps = collections.deque(maxlen=WINDOW_SIZE)
        self.last_published_value = None
        self.warmup_complete = False
        self.current_stress = 0.0
        self.current_period = PERIODE_VALIDITE
        self._new_data_event = threading.Event()  # ✅ AJOUT : Signale nouvelle donnée
        self._lock = threading.Lock()

    def push(self, val: float, timestamp: float):
        with self._lock:
            self.history.append(val)
            self.timestamps.append(timestamp)
            if not self.warmup_complete and len(self.history) >= WARMUP_SAMPLES:
                self.warmup_complete = True
            self._new_data_event.set()  # ✅ AJOUT : Signaler nouvelle donnée

    def wait_for_data(self, timeout=1.0):
        """Attend qu'une nouvelle donnée arrive"""
        return self._new_data_event.wait(timeout)
    
    def clear_event(self):
        """Réinitialise l'event après traitement"""
        self._new_data_event.clear()

    def get_latest(self):
        """Récupère la dernière valeur de manière thread-safe"""
        with self._lock:
            if len(self.history) > 0:
                return self.history[-1], self.timestamps[-1]
            return None, None

    def get_stress(self, val: float, charging: bool = False) -> float:
        """Calcule le stress (0 si dans la plage valide, sinon [0,1])."""
        x = 0.0
        if self.rule.gt and len(self.rule.gt) >= 3 and val > self.rule.gt[0]:
            x = min(1.0, max(0.0, (val - self.rule.gt[0]) / (self.rule.gt[-1] - self.rule.gt[0])))
        elif self.rule.lt and len(self.rule.lt) >= 3 and val < self.rule.lt[0]:
            # ✅ CORRECTION : Gestion charging pour energy
            if not (self.rule.name == "energy" and charging):
                x = min(1.0, max(0.0, (self.rule.lt[0] - val) / (self.rule.lt[0] - self.rule.lt[-1])))
        self.current_stress = x
        return x

def freq_from_value_gt(value: float, seuils: List[float], f_base: float = FREQ_MIN, f_max: float = FREQ_MAX) -> float:
    """Calcule la fréquence pour un seuil GT (logarithmique puis exponentiel)."""
    if len(seuils) < 3:
        return f_base
    s1, s2, s3 = seuils
    if value <= s1:
        return f_base
    elif value <= s2:
        ratio = (value - s1) / (s2 - s1)
        freq = f_base + math.log1p(ratio) / math.log1p(1.0) * (f_max/2 - f_base)
        return freq
    elif value <= s3:
        ratio = (value - s2) / (s3 - s2)
        freq = f_max/2 + (math.exp(ratio) - 1) / (math.e - 1) * (f_max/2)
        return freq
    else:
        return f_max

def freq_from_value_lt(value: float, seuils: List[float], f_base: float = FREQ_MIN, f_max: float = FREQ_MAX) -> float:
    """Calcule la fréquence pour un seuil LT (logarithmique puis exponentiel)."""
    if len(seuils) < 3:
        return f_base
    # ✅ CORRECTION: seuils LT dans l'ordre décroissant [l3, l2, l1]
    l3, l2, l1 = seuils
    if value >= l3:
        return f_base
    elif value >= l2:
        ratio = (l3 - value) / (l3 - l2)
        freq = f_base + math.log1p(ratio) / math.log1p(1.0) * (f_max/2 - f_base)
        return freq
    elif value >= l1:
        ratio = (l2 - value) / (l2 - l1)
        freq = f_max/2 + (math.exp(ratio) - 1) / (math.e - 1) * (f_max/2)
        return freq
    else:
        return f_max

def freq_from_value_unified(value: float, 
                           seuils_gt: Optional[List[float]], 
                           seuils_lt: Optional[List[float]], 
                           f_base: float = FREQ_MIN, 
                           f_max: float = FREQ_MAX) -> float:
    """
    Calcule la fréquence sur une courbe unifiée pour métriques avec GT ET LT.
    
    Cas d'usage typique: TEMPÉRATURE
    - LT [10, 5, 0] : Froid critique → fréquence monte
    - Zone confort [10-45°C] : fréquence basse (f_base)
    - GT [45, 65, 80] : Chaud critique → fréquence monte
    
    Résultat: Courbe en "U" 
    
    Args:
        value: Valeur actuelle
        seuils_gt: [s1, s2, s3] dans l'ordre CROISSANT où value > s1 active GT
        seuils_lt: [l3, l2, l1] dans l'ordre DÉCROISSANT où value < l3 active LT
        f_base: Fréquence de base (zone confort)
        f_max: Fréquence maximale (zones critiques)
    
    Returns:
        Fréquence calculée (f_base à f_max)
    """
    # Si seulement GT ou seulement LT, utiliser fonction classique
    if seuils_gt and not seuils_lt:
        return freq_from_value_gt(value, seuils_gt, f_base, f_max)
    if seuils_lt and not seuils_gt:
        return freq_from_value_lt(value, seuils_lt, f_base, f_max)
    
    # Cas unifié: GT ET LT
    if not seuils_gt or len(seuils_gt) < 3:
        return f_base
    if not seuils_lt or len(seuils_lt) < 3:
        return f_base
    
    gt_s1, gt_s2, gt_s3 = seuils_gt  # Ordre croissant: [45, 65, 80]
    lt_l3, lt_l2, lt_l1 = seuils_lt  # ✅ Ordre décroissant: [10, 5, 0]
    
    # Zone GT (chaud pour température)
    if value >= gt_s1:
        if value <= gt_s2:
            ratio = (value - gt_s1) / (gt_s2 - gt_s1)
            freq = f_base + math.log1p(ratio) / math.log1p(1.0) * (f_max/2 - f_base)
            return freq
        elif value <= gt_s3:
            ratio = (value - gt_s2) / (gt_s3 - gt_s2)
            freq = f_max/2 + (math.exp(ratio) - 1) / (math.e - 1) * (f_max/2)
            return freq
        else:
            return f_max
    
    # Zone LT (froid pour température)
    elif value <= lt_l3:
        if value >= lt_l2:
            ratio = (lt_l3 - value) / (lt_l3 - lt_l2)
            freq = f_base + math.log1p(ratio) / math.log1p(1.0) * (f_max/2 - f_base)
            return freq
        elif value >= lt_l1:
            ratio = (lt_l2 - value) / (lt_l2 - lt_l1)
            freq = f_max/2 + (math.exp(ratio) - 1) / (math.e - 1) * (f_max/2)
            return freq
        else:
            return f_max
    
    # Zone confort (entre LT et GT)
    else:
        return f_base

def freq_from_value_lt_battery(value: float, 
                               seuils: List[float], 
                               f_base: float = 1.0,  # ← Même paramètre que GT
                               f_max: float = FREQ_MAX) -> float:
    """
    Calcule la fréquence pour batterie (LT) avec la MÊME courbe que GT.
    
    IDENTIQUE à freq_from_value_gt, juste inversée pour LT.
    
    Args:
        value: Niveau batterie (%)
        seuils: [l3, l2, l1] dans l'ordre DÉCROISSANT (ex: [60, 30, 15])
        f_base: Fréquence de base (1 Hz pour batterie)
        f_max: Fréquence maximale
    
    Returns:
        Fréquence dans [f_base, f_max]
    
    Exemple avec seuils = [60, 30, 15], f_base=1.0, f_max=10.0:
        battery ≥ 60% → freq = 1.0 Hz (f_base)
        battery = 45% → freq ≈ 3.5 Hz (zone log)
        battery = 20% → freq ≈ 7 Hz (zone exp)
        battery ≤ 15% → freq = 10 Hz (f_max)
    """
    if len(seuils) < 3:
        return f_base
    
    # ✅ CORRECTION: seuils LT sont dans l'ordre décroissant
    l3, l2, l1 = seuils  # [60, 30, 15] → l3=60, l2=30, l1=15
    
    if value >= l3:
        # Au-dessus du seuil haut (≥60%) → fréquence de base
        return f_base
    elif value >= l2:
        # Zone log: 60% → 30% (progression douce)
        ratio = (l3 - value) / (l3 - l2)
        freq = f_base + math.log1p(ratio) / math.log1p(1.0) * (f_max/2 - f_base)
        return freq
    elif value >= l1:
        # Zone exp: 30% → 15% (progression agressive)
        ratio = (l2 - value) / (l2 - l1)
        freq = f_max/2 + (math.exp(ratio) - 1) / (math.e - 1) * (f_max/2)
        return freq
    else:
        # En-dessous du seuil bas (<15%) → fréquence max
        return f_max

class PulseGenerator(threading.Thread):
    def __init__(self, rule: AlertRule, session: zenoh.Session):
        super().__init__(daemon=True)
        self.rule = rule
        self.pub = session.declare_publisher(rule.flux_topic)
        self.target_period = 1.0 / rule.output_freq_min
        self.payload = None
        self.running = True
        self._lock = threading.Lock()

    def update(self, period: float, msg: Optional[str]):
        with self._lock:
            self.target_period = period
            self.payload = msg

    def run(self):
        while self.running:
            with self._lock:
                p, msg = self.target_period, self.payload
            if msg:
                try:
                    self.pub.put(msg)
                    time.sleep(max(0.001, p))
                except Exception as e:
                    logger.error(f"Erreur publication pour {self.rule.name}: {e}")
                    time.sleep(1.0)
            else:
                void_payload = json.dumps({"state": "void", "ts": datetime.now(timezone.utc).isoformat()})
                try:
                    self.pub.put(void_payload)
                except Exception as e:
                    logger.error(f"Erreur publication void pour {self.rule.name}: {e}")
                time.sleep(LIFE_NOISE_PERIOD)

    def stop(self):
        self.running = False

class SomaCore:
    def __init__(self, config_path: str):
        self.running = True
        
         # Initialisation Zenoh
        self.conf = zenoh.Config()
        self.zenohCfgFile = "zenoh_config.json5"
        if os.path.exists(self.zenohCfgFile):
            self.conf.from_file(self.zenohCfgFile)
        self.session = zenoh.open(self.conf)

        self._local_period = PERIODE_VALIDITE
        self._local_tick = 0
        self._clock_lock = threading.Lock()
        self._salve_times = []
        self._phase_sync_event = threading.Event()
        self._charging = False  # État charging
        self._last_charging = False  # Pour détecter changement
        self._temperature = 0.0  # Température courante
        self._temp_critical = False  # État critique température
        
        # ✅ NOUVEAU : Moniteur de batterie
        self._battery_monitor = BatteryMonitor(battery_capacity_wh=50.0)  # Ajuster selon batterie réelle
        
        # ✅ NOUVEAU : Events spéciaux
        self._charging_changed_event = threading.Event()
        self._temp_critical_event = threading.Event()

        # Charger la configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # Charger les profils d'échantillonnage
        self.sampling_profiles = {
            name: SamplingProfile(name=name, **profile_data)
            for name, profile_data in config["sampling_profiles"].items()
        }
        self.default_sampling_profile = config["default_sampling_profile"]
        logger.info(f"📊 Profils d'échantillonnage: {list(self.sampling_profiles.keys())}")

        # Créer les AlertRule
        self.rules = []
        for metric_name, metric_config in config["metrics"].items():
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
                mixing_coefficients=metric_config.get("mixing_coefficients"),
                normalization=metric_config.get("normalization", "weighted_rms")
            )
            self.rules.append(rule)
            logger.debug(f"Règle créée: {metric_name} (profil={rule.sampling_profile})")

        # Initialiser les collecteurs
        self.collectors = [SystemMetricCollector(), IMUMetricCollector()]

        # Créer les nerfs et générateurs (sauf ego)
        self.nerves = {r.name: MetricNerve(r) for r in self.rules if r.name != "ego"}
        self.generators = {r.name: PulseGenerator(r, self.session) for r in self.rules if r.name != "ego"}

        # Démarrer les générateurs
        for gen in self.generators.values():
            gen.start()

        # Gérer le nerf ego séparément
        ego_rule = next((r for r in self.rules if r.name == "ego"), None)
        self.ego_nerve = MetricNerve(ego_rule) if ego_rule else None
        self.ego_gen = PulseGenerator(ego_rule, self.session) if ego_rule else None
        if self.ego_gen:
            self.ego_gen.start()

        # Regrouper les métriques par profil
        self.metrics_by_profile = {}
        for rule in self.rules:
            if rule.name == "ego":
                continue
            profile_name = rule.sampling_profile
            if profile_name not in self.metrics_by_profile:
                self.metrics_by_profile[profile_name] = []
            self.metrics_by_profile[profile_name].append(rule.name)

        # ✅ CORRECTION : Démarrer 1 thread d'acquisition par profil
        for profile_name, metric_names in self.metrics_by_profile.items():
            profile = self.sampling_profiles[profile_name]
            logger.info(f"🔄 Thread acquisition [{profile_name}]: {metric_names} @ {profile.frequency}Hz")
            threading.Thread(
                target=self._sampling_loop,
                args=(profile.frequency, metric_names),
                daemon=True
            ).start()

        # ✅ CORRECTION : Démarrer 1 thread de traitement par nerf (TOUS les profils)
        for metric_name in self.nerves.keys():
            threading.Thread(
                target=self._nerve_process_loop,
                args=(metric_name,),
                daemon=True
            ).start()

        # ✅ CORRECTION : Démarrer le thread ego
        if self.ego_gen:
            threading.Thread(target=self._ego_process_loop, daemon=True).start()
            logger.info("🧠 Thread ego démarré")

        logger.info("⏳ Phase d'éveil...")
        self._warmup_phase()
        
        # Démarrer le thread somatic (pour synchro horloge globale)
        self.session.declare_subscriber("clock/somatic", self._on_global_tick)
        threading.Thread(target=self._somatic_loop, daemon=True).start()
        
        signal.signal(signal.SIGINT, lambda s, f: self.stop())

    def _sampling_loop(self, frequency: float, metric_names: List[str]):
        """✅ CORRIGÉ : Boucle de collecte pour un profil donné - collecte UNIQUEMENT ses métriques."""
        period = 1.0 / frequency
        logger.info(f"▶️  Acquisition [{', '.join(metric_names[:3])}{'...' if len(metric_names) > 3 else ''}] @ {frequency}Hz")
        
        while self.running:
            start_time = time.time()
            current_time = start_time
            
            # ✅ CORRECTION : Chaque collecteur ne collecte QUE les métriques demandées
            metrics = {}
            for collector in self.collectors:
                # Demander uniquement les métriques supportées par ce collecteur
                requested = [name for name in metric_names if collector.can_collect(name)]
                if requested:
                    collector_metrics = collector.collect(requested)
                    metrics.update(collector_metrics)
            
            # ✅ NOUVEAU : Détection événements spéciaux
            
            # Event 1: Changement état charging (publication simple)
            if "charging" in metrics and "energy" in metrics:
                new_charging = metrics["charging"]
                battery_level = metrics["energy"]
                
                # Mettre à jour le moniteur de batterie
                battery_info = self._battery_monitor.update(
                    level=battery_level,
                    charging=new_charging,
                    timestamp=current_time
                )
                
                # Détecter changement d'état
                if new_charging != self._last_charging:
                    self._charging = new_charging
                    self._last_charging = new_charging
                    self._charging_changed_event.set()
                    
                    logger.warning(f"⚡ ÉVÉNEMENT: Charging {'ACTIVÉ' if new_charging else 'DÉSACTIVÉ'} | Niveau: {battery_level:.1f}%")
                else:
                    self._charging = new_charging
                
                # Publication energy/charging UNIQUEMENT si en charge
                if new_charging:
                    charging_payload = {
                        "charging": True,
                        "level": round(battery_level, 1),
                        "timestamp": datetime.fromtimestamp(current_time, timezone.utc).isoformat(),
                        "charge_rate": round(battery_info["charge_rate"] * 3600, 2) if battery_info["charge_rate"] else 0.0,  # %/h
                        "time_remaining_seconds": round(battery_info["time_remaining"]) if battery_info["time_remaining"] else None,
                        "charge_quality": battery_info["charge_quality"]
                    }
                    
                    try:
                        self.pub_charging.put(json.dumps(charging_payload))
                    except Exception as e:
                        logger.error(f"Erreur publication energy/charging: {e}")
            
            elif "charging" in metrics:
                # Fallback si seulement charging disponible
                self._charging = metrics["charging"]
            
            # Event 2: Température critique
            if "temperature" in metrics:
                temp = metrics["temperature"]
                self._temperature = temp
                # Seuil critique à 80°C
                was_critical = self._temp_critical
                self._temp_critical = temp > 80.0
                if self._temp_critical and not was_critical:
                    self._temp_critical_event.set()
                    logger.error(f"🔥 ÉVÉNEMENT CRITIQUE: Température {temp:.1f}°C > 80°C !")
                elif not self._temp_critical and was_critical:
                    logger.info(f"❄️  Température revenue à la normale: {temp:.1f}°C")
            
            # Pousser dans les nerfs correspondants
            for name in metric_names:
                if name in metrics and name in self.nerves:
                    self.nerves[name].push(metrics[name], current_time)
            
            # Respecter la période d'échantillonnage
            elapsed = time.time() - start_time
            sleep_time = max(0.0001, period - elapsed)
            time.sleep(sleep_time)

    def _nerve_process_loop(self, metric_name: str):
        """✅ CORRIGÉ : Boucle de traitement avec attente de nouvelles données."""
        nerve = self.nerves[metric_name]
        generator = self.generators[metric_name]
        logger.info(f"🔧 Thread traitement [{metric_name}] démarré")
        
        while self.running:
            # ✅ CORRECTION : Attendre une nouvelle donnée au lieu de boucler
            if not nerve.wait_for_data(timeout=2.0):
                continue
            
            nerve.clear_event()
            
            val, timestamp = nerve.get_latest()
            if val is None or not nerve.warmup_complete:
                continue
            
            # Calculer le stress
            stress = nerve.get_stress(val, self._charging)

            # ✅ NOUVEAU : Calcul de fréquence selon le type de métrique
            
            # CAS 1: TEMPÉRATURE (GT + LT) → Courbe unifiée en "U"
            if metric_name == "temperature" and nerve.rule.gt and nerve.rule.lt:
                new_freq = freq_from_value_unified(
                    val, 
                    nerve.rule.gt, 
                    nerve.rule.lt, 
                    nerve.rule.output_freq_min, 
                    nerve.rule.output_freq_max
                )
            
            # CAS 2: BATTERIE (LT uniquement) → Même courbe que GT avec f_base=1Hz
            # Note: energy utilise LT mais seulement pour calcul stress ego
            # La publication se fait via energy_discharge avec GT
            elif metric_name == "energy" and nerve.rule.lt:
                if not self._charging:  # Seulement si pas en charge
                    new_freq = freq_from_value_lt_battery(
                        val, 
                        nerve.rule.lt, 
                        f_base=1.0,  # Base à 1 Hz (au lieu de output_freq_min)
                        f_max=nerve.rule.output_freq_max
                    )
                else:
                    new_freq = 1.0  # En charge → 1 Hz fixe
            
            # CAS 2b: ENERGY_DISCHARGE (GT) → Publié seulement si NOT charging
            elif metric_name == "energy_discharge" and nerve.rule.gt:
                if not self._charging:  # Seulement si sur batterie
                    new_freq = freq_from_value_gt(
                        val,
                        nerve.rule.gt,
                        nerve.rule.output_freq_min,
                        nerve.rule.output_freq_max
                    )
                    
                    # Enrichir le payload avec infos de décharge
                    battery_info = self._battery_monitor.history[-1] if self._battery_monitor.history else None
                    if battery_info:
                        discharge_rate = self._battery_monitor.update(
                            100.0 - val,  # Reconvertir en level
                            False,
                            timestamp
                        )
                        
                        payload = json.dumps({
                            "v": round(val, 2),
                            "stress": round(stress, 3),
                            "ts": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                            "freq": round(new_freq, 3),
                            "battery_level": round(100.0 - val, 2),  # Niveau réel
                            "discharge_rate": round(discharge_rate["discharge_rate"] * 3600, 2) if discharge_rate["discharge_rate"] else 0.0,  # %/h
                            "time_remaining_seconds": round(discharge_rate["time_remaining"]) if discharge_rate["time_remaining"] else None,
                            "confidence": discharge_rate.get("confidence", "low")
                        })
                        generator.update(new_period, payload)
                        continue  # Skip le payload générique ci-dessous
                else:
                    # En charge → ne pas publier energy_discharge
                    continue
            
            # CAS 3: Seuils GT uniquement
            elif nerve.rule.gt and len(nerve.rule.gt) >= 3 and val > nerve.rule.gt[0]:
                new_freq = freq_from_value_gt(
                    val, 
                    nerve.rule.gt, 
                    nerve.rule.output_freq_min, 
                    nerve.rule.output_freq_max
                )
            
            # CAS 4: Seuils LT uniquement (autres métriques)
            elif nerve.rule.lt and len(nerve.rule.lt) >= 3 and val < nerve.rule.lt[0]:
                new_freq = freq_from_value_lt(
                    val, 
                    nerve.rule.lt, 
                    nerve.rule.output_freq_min, 
                    nerve.rule.output_freq_max
                )
            
            # CAS 5: Pas d'alerte
            else:
                new_freq = nerve.rule.output_freq_min

            new_period = 1.0 / new_freq

            # Publier
            payload = json.dumps({
                "v": round(val, 2),
                "stress": round(stress, 3),
                "ts": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                "freq": round(new_freq, 3)
            })
            generator.update(new_period, payload)

    def _ego_process_loop(self):
        """✅ CORRIGÉ : Boucle de traitement pour le nerf ego avec coefficients de mixage."""
        if not self.ego_gen:
            return
        
        ego_rule = next((r for r in self.rules if r.name == "ego"), None)
        if not ego_rule:
            return
        
        # Coefficients de mixage (par défaut tous à 1.0)
        mixing_coeffs = ego_rule.mixing_coefficients or {}
        normalization = ego_rule.normalization
        
        logger.info(f"🧠 Thread ego démarré (normalisation: {normalization})")
        if mixing_coeffs:
            logger.info(f"   Coefficients de mixage: {mixing_coeffs}")
        
        while self.running:
            weighted_stresses = []
            stress_details = {}
            
            for name, nerve in self.nerves.items():
                if nerve.warmup_complete and len(nerve.history) > 0:
                    stress = nerve.current_stress
                    
                    # Appliquer le coefficient de mixage
                    coeff = mixing_coeffs.get(name, 1.0)
                    weighted_stress = stress * coeff
                    
                    weighted_stresses.append(weighted_stress ** 2)
                    stress_details[name] = {
                        "raw": round(stress, 3),
                        "coeff": coeff,
                        "weighted": round(weighted_stress, 3)
                    }
            
            # Calcul selon la méthode de normalisation
            if normalization == "weighted_rms":
                # RMS pondéré (par défaut)
                global_stress = math.sqrt(sum(weighted_stresses)) if weighted_stresses else 0.0
            elif normalization == "max":
                # Maximum des stress pondérés
                global_stress = max([math.sqrt(s) for s in weighted_stresses]) if weighted_stresses else 0.0
            elif normalization == "mean":
                # Moyenne des stress pondérés
                global_stress = sum([math.sqrt(s) for s in weighted_stresses]) / len(weighted_stresses) if weighted_stresses else 0.0
            else:
                # Par défaut: weighted_rms
                global_stress = math.sqrt(sum(weighted_stresses)) if weighted_stresses else 0.0
            
            global_stress = min(global_stress, 1.0)

            if self.ego_nerve:
                self.ego_nerve.current_stress = global_stress
            
            # ✅ NOUVEAU : Ajouter info événements spéciaux dans le payload
            payload = json.dumps({
                "v": round(global_stress, 3),
                "ts": datetime.now(timezone.utc).isoformat(),
                "normalization": normalization,
                "details": stress_details,
                "events": {
                    "charging": self._charging,
                    "temp_critical": self._temp_critical,
                    "temperature": round(self._temperature, 1)
                }
            })
            self.ego_gen.update(1.0, payload)
            time.sleep(1.0)  # Fréquence fixe 1 Hz pour ego

    def _on_global_tick(self, sample):
        """Synchronisation avec l'horloge globale."""
        now = time.time()
        try:
            p_str = sample.payload.decode() if hasattr(sample.payload, "decode") else sample.payload.to_string()
            data = json.loads(p_str)
            with self._clock_lock:
                self._salve_times.append(now)
                if len(self._salve_times) >= SALVE_SIZE:
                    delta = (self._salve_times[-1] - self._salve_times[0]) / (len(self._salve_times)-1)
                    self._local_period = max(0.1, min(delta, 5.0))
                    self._local_tick = data.get("tick", self._local_tick)
                    self._salve_times.pop(0)
                self._phase_sync_event.set()
        except Exception as e:
            logger.error(f"Erreur _on_global_tick: {e}")

    def _somatic_loop(self):
        """Boucle somatic pour synchronisation globale."""
        try:
            while self.running:
                synced = self._phase_sync_event.wait(timeout=self._local_period * 2.0)
                self._phase_sync_event.clear()
                if not self.running:
                    break
                self._local_tick += 1
                # Plus besoin de process() ici, tout est fait dans les threads dédiés
        except Exception as e:
            logger.error(f"Erreur _somatic_loop: {e}")

    def _warmup_phase(self):
        """Phase de warmup initiale."""
        logger.info("🔥 Warmup en cours...")
        time.sleep(WARMUP_SAMPLES * 0.1)  # Laisser les threads collecter
        logger.info("✅ Warmup terminé")

    def stop(self):
        """Arrêt propre du système."""
        logger.info("🛑 Arrêt en cours...")
        self.running = False
        self._phase_sync_event.set()
        
        for generator in self.generators.values():
            generator.stop()
        if self.ego_gen:
            self.ego_gen.stop()
        
        self.session.close()
        logger.info("✅ SomaCore éteint")

# ================== MAIN ==================
if __name__ == "__main__":
    try:
        config_file = "soma_rules.json"
        if not os.path.exists(config_file):
            logger.error(f"❌ Config manquante : {config_file}")
        else:
            core = SomaCore(config_file)
            logger.info("✅ SomaCore démarré - Appuyez sur Ctrl+C pour arrêter")
            while True:
                time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("👋 Arrêt demandé")
    except Exception as e:
        logger.error(f"💥 Erreur Fatale: {e}", exc_info=True)