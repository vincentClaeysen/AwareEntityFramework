#!/usr/bin/env python3
"""
SomaCore v2.3.0 – Bio-inspired sensor acquisition organ
Complete implementation with 3 Zenoh hubs, in‑memory override, and config validation.
"""

import os
import sys
import json
import time
import math
import uuid
import threading
import logging
import collections
import statistics
import queue
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field

import psutil
import zenoh

# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

@dataclass
class AlertRule:
    name: str
    alias: str
    flux_topic: str
    pain_topic: Optional[str] = None
    gt: Optional[List[float]] = None
    lt: Optional[List[float]] = None
    sampling_profile: str = "slow"
    output_freq_min: float = 1.0
    output_freq_max: float = 200.0
    silence_below_threshold: bool = True
    absolute_delta_threshold: float = 0.0
    aggro_factor: float = 2.5
    weight: float = 1.0
    description: str = ""

@dataclass
class SamplingProfile:
    name: str
    frequency: float
    description: str = ""

@dataclass
class SensorConfig:
    name: str
    nerve_alias: str
    source_profile: str
    target_freq: float
    effective_freq: float
    effective_period: float
    max_read_time: float
    avg_read_time: float
    variability: float
    timeout: float
    max_threshold: float
    min_threshold: Optional[float] = None
    degraded: bool = False
    unstable: bool = False
    suspended: bool = False
    in_pain: bool = False
    consecutive_exceptions: int = 0
    suspension_reason: str = ""
    override: bool = False
    rule: Optional[AlertRule] = None


# ============================================================================
# STRESS LOOKUP TABLE (O(1) stress calculation)
# ============================================================================

class StressLookupTable:
    """
    Pre‑computed lookup table for stress calculation (O(1) access with linear interpolation).
    """
    
    def __init__(self, rule: AlertRule, steps: int = 1000):
        self.rule = rule
        self.steps = steps
        self.min_val, self.max_val = self._get_bounds()
        self.table = self._build_table()
    
    def _get_bounds(self) -> Tuple[float, float]:
        if self.rule.gt:
            return 0.0, self.rule.gt[-1] * 1.2
        elif self.rule.lt:
            return 0.0, self.rule.lt[0] * 1.2
        else:
            return 0.0, 1.0
    
    def _build_table(self) -> List[float]:
        min_val, max_val = self.min_val, self.max_val
        table = []
        for i in range(self.steps + 1):
            val = min_val + (max_val - min_val) * i / self.steps
            table.append(self._compute_raw(val))
        return table
    
    def _compute_raw(self, value: float) -> float:
        if self.rule.gt:
            s1, s2, s3 = self.rule.gt
            if value >= s3:
                return 1.0
            elif value >= s2:
                ratio = (value - s2) / (s3 - s2)
                return 0.9 + 0.1 * ratio
            elif value >= s1:
                ratio = (value - s1) / (s2 - s1)
                return 0.8 + 0.1 * ratio
            else:
                return min(0.8, 0.8 * value / s1)
        elif self.rule.lt:
            l3, l2, l1 = self.rule.lt
            if value <= l1:
                return 1.0
            elif value <= l2:
                ratio = (l2 - value) / (l2 - l1)
                return 0.9 + 0.1 * ratio
            elif value <= l3:
                ratio = (l3 - value) / (l3 - l2)
                return 0.8 + 0.1 * ratio
            else:
                return max(0.0, 0.8 * (1.0 - value / l3))
        return 0.0
    
    def get_stress(self, value: float) -> float:
        if value <= self.min_val:
            return self.table[0]
        if value >= self.max_val:
            return self.table[-1]
        idx = (value - self.min_val) / (self.max_val - self.min_val) * self.steps
        i = int(idx)
        frac = idx - i
        if i >= self.steps:
            return self.table[-1]
        return self.table[i] * (1 - frac) + self.table[i + 1] * frac


class StressCalculator:
    _tables: Dict[int, StressLookupTable] = {}
    
    @classmethod
    def get_table(cls, rule: AlertRule) -> StressLookupTable:
        rule_id = id(rule)
        if rule_id not in cls._tables:
            cls._tables[rule_id] = StressLookupTable(rule)
        return cls._tables[rule_id]
    
    @classmethod
    def compute(cls, value: float, rule: AlertRule) -> float:
        return cls.get_table(rule).get_stress(value)
    
    @classmethod
    def clear_tables(cls):
        cls._tables.clear()


# ============================================================================
# OVERRIDE MANAGER (in‑memory only)
# ============================================================================

class OverrideManager:
    """In‑memory override storage – no persistence."""
    
    def __init__(self, component_name: str):
        self.component = component_name
        self.data = {
            "version": 0,
            "timestamp": None,
            "config": {}
        }
    
    def get(self, key: str, default=None):
        return self.data["config"].get(key, default)
    
    def set(self, key: str, value: Any):
        self.data["config"][key] = value
        self.data["version"] += 1
        self.data["timestamp"] = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    
    def update(self, config_dict: Dict):
        self.data["config"].update(config_dict)
        self.data["version"] += 1
        self.data["timestamp"] = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    
    def get_all(self) -> Dict:
        return self.data["config"].copy()
    
    def get_version(self) -> int:
        return self.data["version"]
    
    def get_timestamp(self) -> Optional[str]:
        return self.data["timestamp"]


# ============================================================================
# PUB SCHEDULER
# ============================================================================

class PubScheduler(threading.Thread):
    """Publication scheduler with silence support (frequency = 0)."""
    
    def __init__(self, publish_callback: Callable[[str, Any], None],
                 base_period: float = 0.01, name: str = "PubScheduler"):
        super().__init__(daemon=True, name=name)
        self.base_period = base_period
        self.publish = publish_callback
        self.running = True
        self._lock = threading.RLock()
        
        self.nerfs: Dict[str, List[float]] = {}      # alias -> [counter, step]
        self.active_flags: Dict[str, bool] = {}      # True if active
        self.base_periods: Dict[str, float] = {}     # base period (before sleep modulation)
        self.registry: Dict[str, Any] = {}
        self.pending_steps: Dict[str, float] = {}
        self.stats = {'cycles': 0, 'publications': 0, 'errors': 0}

    def _period_to_step(self, period: float) -> float:
        if period <= 0:
            return 0.0
        return self.base_period / max(period, self.base_period)

    def add_nerve(self, alias: str, target_period: float, active: bool = True):
        step = self._period_to_step(target_period)
        with self._lock:
            self.nerfs[alias] = [0.0, step]
            self.base_periods[alias] = target_period
            self.active_flags[alias] = active and (target_period > 0)
            self.registry.setdefault(alias, None)

    def update_payload(self, alias: str, payload: Any):
        with self._lock:
            self.registry[alias] = payload

    def update_period(self, alias: str, new_period: float):
        new_step = self._period_to_step(new_period)
        with self._lock:
            self.pending_steps[alias] = new_step
            self.base_periods[alias] = new_period
            if new_period <= 0:
                self.active_flags[alias] = False

    def set_activity_factor(self, factor: float):
        with self._lock:
            for alias, base_period in self.base_periods.items():
                new_period = base_period / max(factor, 0.1)
                self.pending_steps[alias] = self._period_to_step(new_period)

    def set_active(self, alias: str, active: bool):
        with self._lock:
            self.active_flags[alias] = active

    def remove_nerve(self, alias: str):
        with self._lock:
            self.nerfs.pop(alias, None)
            self.active_flags.pop(alias, None)
            self.base_periods.pop(alias, None)
            self.registry.pop(alias, None)
            self.pending_steps.pop(alias, None)

    def reset(self):
        with self._lock:
            self.nerfs.clear()
            self.active_flags.clear()
            self.base_periods.clear()
            self.registry.clear()
            self.pending_steps.clear()
            self.stats = {'cycles': 0, 'publications': 0, 'errors': 0}

    def run(self):
        while self.running:
            cycle_start = time.perf_counter()
            with self._lock:
                nerves_items = list(self.nerfs.items())
                active_flags = self.active_flags.copy()
            
            for alias, state in nerves_items:
                if not active_flags.get(alias, False):
                    continue
                state[0] += state[1]
                if state[0] >= 1.0:
                    state[0] = 0.0
                    with self._lock:
                        payload = self.registry.get(alias)
                        if alias in self.pending_steps:
                            state[1] = self.pending_steps.pop(alias)
                    if payload is not None:
                        try:
                            self.publish(alias, payload)
                            with self._lock:
                                self.stats['publications'] += 1
                        except Exception as e:
                            with self._lock:
                                self.stats['errors'] += 1
            
            elapsed = time.perf_counter() - cycle_start
            sleep_time = self.base_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            with self._lock:
                self.stats['cycles'] += 1

    def stop(self):
        self.running = False
        time.sleep(self.base_period * 2)


# ============================================================================
# FREQUENCY MAPPER
# ============================================================================

class FrequencyMapper:
    """Maps a value to a frequency (0 = silence)."""
    
    def __init__(self, rule: AlertRule):
        self.rule = rule
        self.f_min = rule.output_freq_min
        self.f_max = rule.output_freq_max
        self.silence_below = rule.silence_below_threshold
        self.gt = rule.gt if rule.gt and len(rule.gt) >= 3 else None
        self.lt = rule.lt if rule.lt and len(rule.lt) >= 3 else None
        self._cache = {}
    
    def get_frequency(self, value: float) -> float:
        rounded = round(value, 3)
        if rounded in self._cache:
            return self._cache[rounded]
        
        if self.gt:
            s1, s2, s3 = self.gt
            if value > s1:
                if value <= s2:
                    ratio = (value - s1) / (s2 - s1)
                    freq = self.f_min + (self.f_max/2 - self.f_min) * math.log1p(ratio) / math.log(2)
                elif value <= s3:
                    ratio = (value - s2) / (s3 - s2)
                    freq = self.f_max/2 + (self.f_max/2) * (math.exp(ratio) - 1) / (math.e - 1)
                else:
                    freq = self.f_max
                self._cache[rounded] = freq
                return freq
        
        if self.lt:
            l3, l2, l1 = self.lt
            if value < l3:
                if value >= l2:
                    ratio = (l3 - value) / (l3 - l2)
                    freq = self.f_min + (self.f_max/2 - self.f_min) * math.log1p(ratio) / math.log(2)
                elif value >= l1:
                    ratio = (l2 - value) / (l2 - l1)
                    freq = self.f_max/2 + (self.f_max/2) * (math.exp(ratio) - 1) / (math.e - 1)
                else:
                    freq = self.f_max
                self._cache[rounded] = freq
                return freq
        
        self._cache[rounded] = 0.0
        return 0.0


# ============================================================================
# BATTERY MONITOR
# ============================================================================

class BatteryMonitor:
    """Battery monitor with rate estimation."""
    
    def __init__(self):
        self.history = collections.deque(maxlen=60)
        self._lock = threading.RLock()
    
    def update(self, level: float, charging: bool, timestamp: float) -> Dict:
        with self._lock:
            self.history.append((timestamp, level, charging))
            result = {
                "level": round(level, 2),
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
            window = min(len(self.history), max(5, len(self.history)//3))
            recent = list(self.history)[-window:]
            if window >= 30:
                result["confidence"] = "high"
            elif window >= 15:
                result["confidence"] = "medium"
            if len(recent) >= 2:
                dt = recent[-1][0] - recent[0][0]
                if dt > 0:
                    dlevel = recent[-1][1] - recent[0][1]
                    rate_per_second = dlevel / dt
                    rate_per_minute = rate_per_second * 60
                    if charging and rate_per_second > 0:
                        result["charge_rate"] = round(rate_per_minute, 2)
                        remaining = 100.0 - level
                        result["minutes_to_full"] = round(remaining / rate_per_minute, 1)
                    elif not charging and rate_per_second < 0:
                        result["discharge_rate"] = round(abs(rate_per_minute), 2)
                        result["minutes_to_empty"] = round(level / abs(rate_per_minute), 1)
            return result
    
    def reset(self):
        with self._lock:
            self.history.clear()


# ============================================================================
# METRIC COLLECTORS
# ============================================================================

class MetricCollector:
    def __init__(self, name: str):
        self.name = name
        self.supported = set()
    
    def can_collect(self, metric: str) -> bool:
        return metric in self.supported
    
    def collect(self, metrics: List[str]) -> Dict[str, float]:
        return {}


class SystemMetricCollector(MetricCollector):
    """Collector for global system metrics."""
    
    def __init__(self):
        super().__init__("system")
        self.supported = {"cpu", "memory", "temperature", "energy"}
    
    def _get_temperature(self) -> float:
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps and list(temps.values()):
                    return list(temps.values())[0][0].current
        except:
            pass
        return 20.0
    
    def collect(self, metrics: List[str]) -> Dict[str, float]:
        result = {}
        if "cpu" in metrics:
            result["cpu"] = psutil.cpu_percent(interval=None)
        if "memory" in metrics:
            result["memory"] = psutil.virtual_memory().percent
        if "temperature" in metrics:
            result["temperature"] = self._get_temperature()
        if "energy" in metrics:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    result["energy"] = battery.percent
                    result["_energy_charging"] = battery.power_plugged
                else:
                    result["energy"] = 100.0
                    result["_energy_charging"] = False
            except:
                result["energy"] = 100.0
                result["_energy_charging"] = False
        return result


class SelfMetricCollector(MetricCollector):
    """Collector for process‑local (self) metrics."""
    
    def __init__(self):
        super().__init__("self")
        self.supported = {"self_cpu", "self_memory", "self_threads", "self_fds"}
        self.process = psutil.Process()
        self.cpu_measurements = []
        self._lock = threading.RLock()
    
    def _get_cpu(self) -> float:
        try:
            cpu = self.process.cpu_percent(interval=0.1)
            with self._lock:
                self.cpu_measurements.append(cpu)
                if len(self.cpu_measurements) > 5:
                    self.cpu_measurements.pop(0)
                return sum(self.cpu_measurements) / len(self.cpu_measurements) if self.cpu_measurements else 0.0
        except:
            return 0.0
    
    def _get_memory(self) -> float:
        try:
            return self.process.memory_percent()
        except:
            return 0.0
    
    def _get_threads(self) -> int:
        try:
            return self.process.num_threads()
        except:
            return 0
    
    def _get_fds(self) -> int:
        try:
            if hasattr(self.process, 'num_fds'):
                return self.process.num_fds()
            return 0
        except:
            return 0
    
    def collect(self, metrics: List[str]) -> Dict[str, float]:
        result = {}
        for m in metrics:
            if m == "self_cpu":
                result[m] = self._get_cpu()
            elif m == "self_memory":
                result[m] = self._get_memory()
            elif m == "self_threads":
                result[m] = float(self._get_threads())
            elif m == "self_fds":
                result[m] = float(self._get_fds())
        return result


# ============================================================================
# SENSOR ORCHESTRATOR (with history and trends)
# ============================================================================

class SensorOrchestrator:
    """Manages sensor configuration, health, and history."""
    
    def __init__(self, tech_config: Dict, collectors: List):
        self.config = tech_config
        self.collectors = collectors
        self.sensors: Dict[str, SensorConfig] = {}
        self.active: List[str] = []
        self._cache: Dict[str, float] = {}
        self._history: Dict[str, collections.deque] = {}
        self._trends: Dict[str, Dict] = {}
        self._lock = threading.RLock()
    
    def bootstrap(self, rules: List[AlertRule], profiles: Dict) -> Dict[str, SensorConfig]:
        readings = self.config.get('bootstrap', {}).get('readings', 5)
        slow_th = self.config.get('bootstrap', {}).get('slowness_threshold', 3.0)
        var_th = self.config.get('bootstrap', {}).get('variability_threshold', 0.5)
        normal_to = self.config.get('bootstrap', {}).get('normal_timeout_factor', 1.5)
        unstable_to = self.config.get('bootstrap', {}).get('unstable_timeout_factor', 2.5)
        min_freq = self.config.get('acquisition', {}).get('min_absolute_frequency', 0.1)
        
        by_profile = {}
        for rule in rules:
            by_profile.setdefault(rule.sampling_profile, []).append(rule)
        
        for prof_name, rule_list in by_profile.items():
            profile = profiles.get(prof_name)
            if not profile:
                continue
            target_freq = profile.frequency
            target_period = 1.0 / target_freq
            
            measurements = {}
            for rule in rule_list:
                coll = next((c for c in self.collectors if c.can_collect(rule.name)), None)
                if not coll:
                    continue
                times = []
                try:
                    for _ in range(readings):
                        start = time.perf_counter()
                        coll.collect([rule.name])
                        times.append(time.perf_counter() - start)
                        time.sleep(target_period * 0.1)
                    t_max = max(times)
                    t_avg = sum(times)/len(times)
                    var = (t_max - min(times)) / t_avg if t_avg>0 else 0
                    measurements[rule.name] = {
                        'max': t_max, 'avg': t_avg, 'var': var, 'rule': rule
                    }
                except:
                    pass
            
            if not measurements:
                continue
            
            max_times = [m['max'] for m in measurements.values()]
            median_max = statistics.median(max_times) if max_times else 0
            
            for name, stats in measurements.items():
                rule = stats['rule']
                t_max = stats['max']
                var = stats['var']
                
                is_slow = t_max > median_max * slow_th
                is_unstable = var > var_th
                timeout = t_max * (unstable_to if is_unstable else normal_to)
                max_possible = 1.0 / t_max if t_max>0 else target_freq
                eff_freq = min(target_freq, max_possible)
                if eff_freq < min_freq:
                    eff_freq = min_freq
                
                max_th = rule.gt[-1] if rule.gt else (rule.lt[-1] if rule.lt else 1.0)
                min_th = rule.lt[0] if rule.lt else None
                
                self.sensors[name] = SensorConfig(
                    name=name,
                    nerve_alias=rule.alias,
                    source_profile=prof_name,
                    target_freq=target_freq,
                    effective_freq=eff_freq,
                    effective_period=1.0/eff_freq,
                    max_read_time=t_max,
                    avg_read_time=stats['avg'],
                    variability=var,
                    timeout=timeout,
                    max_threshold=max_th,
                    min_threshold=min_th,
                    degraded=is_slow,
                    unstable=is_unstable,
                    rule=rule
                )
        
        self.active = list(self.sensors.keys())
        return self.sensors
    
    def update_cache(self, name: str, value: float):
        with self._lock:
            self._cache[name] = value
            if name not in self._history:
                self._history[name] = collections.deque(maxlen=30)
            self._history[name].append({
                "timestamp": time.time(),
                "value": value
            })
            if len(self._history[name]) >= 10:
                self._trends[name] = self._compute_trend(name)
    
    def _compute_trend(self, name: str) -> Dict:
        history = list(self._history[name])
        if len(history) < 2:
            return {"dir": "_", "speed": 0.0, "confidence": 0.0}
        start = history[0]
        end = history[-1]
        dt = end["timestamp"] - start["timestamp"]
        if dt == 0:
            return {"dir": "_", "speed": 0.0, "confidence": 0.0}
        dv = end["value"] - start["value"]
        speed = abs(dv / dt)
        direction = "_" if abs(dv) < 0.01 else ("+" if dv > 0 else "-")
        confidence = min(1.0, len(history) / 30.0)
        return {"dir": direction, "speed": round(speed, 4), "confidence": round(confidence, 2)}
    
    def get_cached_value(self, name: str) -> Optional[float]:
        with self._lock:
            return self._cache.get(name)
    
    def get_cached_trend(self, name: str) -> Dict:
        with self._lock:
            return self._trends.get(name, {"dir": "_", "speed": 0.0, "confidence": 0.0})
    
    def handle_exception(self, name: str) -> bool:
        with self._lock:
            if name not in self.sensors:
                return False
            cfg = self.sensors[name]
            cfg.consecutive_exceptions += 1
            max_ex = self.config.get('acquisition', {}).get('max_consecutive_exceptions', 3)
            return cfg.consecutive_exceptions >= max_ex
    
    def check_read_time(self, name: str, duration: float) -> Optional[float]:
        with self._lock:
            if name not in self.sensors:
                return None
            cfg = self.sensors[name]
            period = cfg.effective_period
            warn = self.config.get('acquisition', {}).get('timeout_warning_ratio', 0.8)
            if duration > period * warn:
                pass
            if duration > period:
                new_freq = cfg.effective_freq / 2
                min_freq = self.config.get('acquisition', {}).get('min_absolute_frequency', 0.1)
                if new_freq < min_freq:
                    return -1
                cfg.effective_freq = new_freq
                cfg.effective_period = 1.0 / new_freq
                return new_freq
        return None
    
    def suspend_sensor(self, name: str, reason: str):
        with self._lock:
            if name in self.active:
                self.active.remove(name)
                if name in self.sensors:
                    self.sensors[name].suspended = True
                    self.sensors[name].suspension_reason = reason
    
    def get_sensor(self, name: str) -> Optional[SensorConfig]:
        with self._lock:
            return self.sensors.get(name)
    
    def get_active(self) -> List[str]:
        with self._lock:
            return self.active.copy()


# ============================================================================
# PAIN SIGNAL
# ============================================================================

class PainSignal:
    """Pain signal with automatic heartbeat and frequency modulation."""
    
    HEARTBEAT_FREQ = 0.1
    MIN_FREQ = 1.0
    MAX_FREQ = 50.0
    THRESHOLD = 0.8
    
    def __init__(self, domain: str, metric: str, scheduler, zenoh, threshold=0.8):
        self.domain = domain
        self.metric = metric
        self.scheduler = scheduler
        self.zenoh = zenoh
        self.threshold = threshold
        self.topic = f"pain/{domain}/{metric}"
        self.nerve_alias = f"pain_{domain}_{metric}".replace('/', '_')
        self.active = False
        self.last_stress = 0.0
        self.last_value = 0.0
        self.last_metadata = {}
        self.scheduler.add_nerve(self.nerve_alias, 1.0 / self.HEARTBEAT_FREQ, active=True)
        self._update_heartbeat_payload()
    
    def update(self, stress: float, value: float, metadata=None):
        self.last_stress = stress
        self.last_value = value
        if metadata:
            self.last_metadata.update(metadata)
        transition = None
        if stress >= self.threshold:
            if not self.active:
                self.active = True
                transition = "pain_onset"
            intensity = (stress - self.threshold) / (1.0 - self.threshold)
            freq = self.MIN_FREQ + intensity * (self.MAX_FREQ - self.MIN_FREQ)
            self.scheduler.update_period(self.nerve_alias, 1.0 / freq)
        else:
            if self.active:
                self.active = False
                transition = "pain_offset"
            self.scheduler.update_period(self.nerve_alias, 1.0 / self.HEARTBEAT_FREQ)
        if transition:
            self.last_metadata["transition"] = transition
            self.last_metadata["transition_time"] = time.time()
        if self.active:
            self._publish()
        else:
            self._update_heartbeat_payload()
    
    def _publish(self):
        payload = {
            "v": round(self.last_value, 3),
            "stress": round(self.last_stress, 3),
            "active": self.active,
            "freq": self._current_freq(),
            "timestamp": time.time(),
            **self.last_metadata
        }
        self.scheduler.update_payload(self.nerve_alias, payload)
    
    def _update_heartbeat_payload(self):
        payload = {
            "v": round(self.last_value, 3),
            "stress": round(self.last_stress, 3),
            "active": False,
            "freq": self.HEARTBEAT_FREQ,
            "heartbeat": True,
            "timestamp": time.time(),
            **self.last_metadata
        }
        self.scheduler.update_payload(self.nerve_alias, payload)
    
    def _current_freq(self) -> float:
        if self.active:
            return 1.0 / self.scheduler.base_periods.get(self.nerve_alias, 1.0)
        return self.HEARTBEAT_FREQ
    
    def stop(self):
        self.scheduler.remove_nerve(self.nerve_alias)


# ============================================================================
# ORGAN FAILURE SIGNAL (with dedicated spike thread)
# ============================================================================

class OrganFailureSignal:
    """Organ failure signal with non‑blocking spike thread."""
    
    HEARTBEAT_FREQ = 1.0
    
    def __init__(self, component: str, scheduler, zenoh):
        self.component = component
        self.scheduler = scheduler
        self.zenoh = zenoh
        self.topic = f"nerve/organ/{component}"
        self.heartbeat_alias = f"organ_{component}_heartbeat"
        self.failing = False
        self.reason = {}
        self._spike_queue = queue.Queue()
        self._spike_thread = threading.Thread(target=self._spike_emitter, daemon=True)
        self._spike_thread.start()
    
    def _spike_emitter(self):
        while True:
            try:
                spike_data = self._spike_queue.get(timeout=1.0)
                if spike_data is None:
                    break
                event, reason, count = spike_data
                for i in range(count):
                    payload = {
                        "event": event,
                        "reason": reason,
                        "spike": i+1,
                        "total": count,
                        "timestamp": time.time()
                    }
                    self.zenoh.put(self.topic, json.dumps(payload))
                    time.sleep(0.01)
            except queue.Empty:
                continue
    
    def enter(self, reason: Dict):
        self.failing = True
        self.reason = reason
        self._spike_queue.put(("failure_enter", reason, 10))
        self.scheduler.add_nerve(self.heartbeat_alias, 1.0 / self.HEARTBEAT_FREQ, active=True)
        self._update_heartbeat()
    
    def update_reason(self, reason: Dict):
        self.reason.update(reason)
        self._update_heartbeat()
    
    def _update_heartbeat(self):
        payload = {
            "event": "failure_heartbeat",
            "failing": True,
            "reason": self.reason,
            "timestamp": time.time()
        }
        self.scheduler.update_payload(self.heartbeat_alias, payload)
    
    def exit(self):
        if not self.failing:
            return
        self.failing = False
        self.scheduler.remove_nerve(self.heartbeat_alias)
        self._spike_queue.put(("failure_exit", self.reason, 10))
    
    def cleanup(self):
        self._spike_queue.put(None)
        self._spike_thread.join(timeout=2.0)


# ============================================================================
# NEURAL SIGNALING SYSTEM
# ============================================================================

class NeuralSignalingSystem:
    """Unified neural signaling system."""
    
    def __init__(self, component: str, nerve_session, nerve_scheduler):
        self.component = component
        self.zenoh = nerve_session
        self.scheduler = nerve_scheduler
        self.pain_signals: Dict[Tuple[str, str], PainSignal] = {}
        self.organ_failure: Optional[OrganFailureSignal] = None
    
    def emit_pain(self, domain: str, metric: str, stress: float, value: float, metadata=None):
        key = (domain, metric)
        if key not in self.pain_signals:
            self.pain_signals[key] = PainSignal(domain, metric, self.scheduler, self.zenoh)
        self.pain_signals[key].update(stress, value, metadata)
    
    def stop_pain(self, domain: str, metric: str):
        key = (domain, metric)
        if key in self.pain_signals:
            self.pain_signals[key].stop()
            del self.pain_signals[key]
    
    def emit_sensor_fault(self, sensor: str, reason: str, severity: str) -> int:
        severity_map = {"info": 1, "warning": 5, "error": 20, "critical": 100}
        spikes = severity_map.get(severity, 1)
        topic = f"nerve/diagnostics/{self.component}/sensor/{sensor}"
        for i in range(spikes):
            payload = {
                "event": "sensor_fault",
                "sensor": sensor,
                "severity": severity,
                "reason": reason,
                "spike": i+1,
                "total": spikes,
                "timestamp": time.time()
            }
            self.zenoh.put(topic, json.dumps(payload))
            time.sleep(0.01)
        return spikes
    
    def emit_sensor_recovery(self, sensor: str):
        topic = f"nerve/diagnostics/{self.component}/sensor/{sensor}"
        payload = {
            "event": "sensor_recovery",
            "sensor": sensor,
            "timestamp": time.time()
        }
        self.zenoh.put(topic, json.dumps(payload))
    
    def emit_self_fault(self, fault_type: str, reason: str, severity: str) -> int:
        severity_map = {"warning": 5, "error": 20, "critical": 100}
        spikes = severity_map.get(severity, 5)
        topic = f"nerve/diagnostics/{self.component}/self/{fault_type}"
        for i in range(spikes):
            payload = {
                "event": "self_fault",
                "fault_type": fault_type,
                "severity": severity,
                "reason": reason,
                "spike": i+1,
                "total": spikes,
                "timestamp": time.time()
            }
            self.zenoh.put(topic, json.dumps(payload))
            time.sleep(0.01)
        return spikes
    
    def emit_organ_failure(self, reason: Dict):
        if not self.organ_failure:
            self.organ_failure = OrganFailureSignal(self.component, self.scheduler, self.zenoh)
        self.organ_failure.enter(reason)
    
    def update_organ_failure(self, reason: Dict):
        if self.organ_failure and self.organ_failure.failing:
            self.organ_failure.update_reason(reason)
    
    def emit_organ_recovery(self):
        if self.organ_failure:
            self.organ_failure.exit()
            self.organ_failure = None
    
    def cleanup(self):
        for key in list(self.pain_signals.keys()):
            self.stop_pain(*key)
        if self.organ_failure:
            self.organ_failure.exit()
            self.organ_failure = None


# ============================================================================
# ACQUISITION MANAGER
# ============================================================================

class AcquisitionManager:
    """Manages acquisition threads."""
    
    def __init__(self, tech_config: Dict, orch: SensorOrchestrator,
                 collectors: List, neural: NeuralSignalingSystem,
                 battery: BatteryMonitor, running_flag: Callable[[], bool]):
        self.config = tech_config
        self.orch = orch
        self.collectors = collectors
        self.neural = neural
        self.battery = battery
        self.running = running_flag
        self.charging = False
        self._lock = threading.RLock()
    
    def set_charging(self, chg: bool):
        with self._lock:
            self.charging = chg
    
    def start(self):
        by_freq = {}
        for s in self.orch.get_active():
            cfg = self.orch.get_sensor(s)
            if not cfg or cfg.suspended:
                continue
            by_freq.setdefault(cfg.effective_freq, []).append(s)
        for freq, sensors in by_freq.items():
            t = threading.Thread(target=self._loop, args=(freq, sensors), daemon=True)
            t.start()
    
    def _loop(self, freq: float, sensors: List[str]):
        period = 1.0 / freq
        while self.running():
            start = time.time()
            now = start
            for s in sensors:
                if s not in self.orch.get_active():
                    continue
                cfg = self.orch.get_sensor(s)
                if not cfg or cfg.suspended:
                    continue
                coll = next((c for c in self.collectors if c.can_collect(s)), None)
                if not coll:
                    continue
                try:
                    read_start = time.perf_counter()
                    vals = coll.collect([s])
                    val = vals.get(s, 0.0)
                    duration = time.perf_counter() - read_start
                    self.orch.update_cache(s, val)
                    res = self.orch.check_read_time(s, duration)
                    if res == -1:
                        self.orch.suspend_sensor(s, "timeout critical")
                        self.neural.emit_sensor_fault(s, "timeout critical", "error")
                        continue
                    stress = StressCalculator.compute(val, cfg.rule)
                    domain = "soma" if not s.startswith("self_") else "soma_core"
                    self.neural.emit_pain(
                        domain=domain,
                        metric=s,
                        stress=stress,
                        value=val,
                        metadata={"read_time_ms": duration*1000}
                    )
                    if s == "energy" and "_energy_charging" in vals:
                        chg = vals["_energy_charging"]
                        bat_info = self.battery.update(val, chg, now)
                        self.set_charging(chg)
                except Exception as e:
                    if self.orch.handle_exception(s):
                        self.orch.suspend_sensor(s, "exceptions")
                        self.neural.emit_sensor_fault(s, f"exceptions: {type(e).__name__}", "error")
            elapsed = time.time() - start
            time.sleep(max(0, period - elapsed))


# ============================================================================
# SELF HEALTH MANAGER
# ============================================================================

class SelfHealthManager:
    """Monitors self metrics with trend detection."""
    
    def __init__(self, neural: NeuralSignalingSystem, collector: SelfMetricCollector):
        self.neural = neural
        self.collector = collector
        self.running = True
        self.history = collections.deque(maxlen=60)
        self.trends = {"cpu": 0.0, "memory": 0.0}
    
    def _compute_trend(self, metric: str, window: int = 30) -> float:
        if len(self.history) < window:
            return 0.0
        recent = list(self.history)[-window:]
        n = len(recent)
        sum_x = sum(range(n))
        sum_y = sum(r[metric] for r in recent)
        sum_xy = sum(i * r[metric] for i, r in enumerate(recent))
        sum_xx = sum(i * i for i in range(n))
        if n * sum_xx - sum_x * sum_x == 0:
            return 0.0
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope
    
    def monitoring_loop(self):
        while self.running:
            metrics = self.collector.collect(["self_cpu", "self_memory", "self_threads", "self_fds"])
            now = time.time()
            self.history.append({
                "timestamp": now,
                "cpu": metrics.get("self_cpu", 0.0),
                "memory": metrics.get("self_memory", 0.0)
            })
            if len(self.history) >= 30:
                self.trends["cpu"] = self._compute_trend("cpu", 30)
                self.trends["memory"] = self._compute_trend("memory", 30)
            for name, value in metrics.items():
                if name == "self_cpu":
                    stress = StressCalculator.compute(value, AlertRule(gt=[80,150,200]))  # simplified
                    trend_val = self.trends.get("cpu", 0.0)
                elif name == "self_memory":
                    stress = StressCalculator.compute(value, AlertRule(gt=[20,40,60]))
                    trend_val = self.trends.get("memory", 0.0)
                else:
                    stress = min(1.0, value / 100.0)
                    trend_val = 0.0
                metadata = {"source": "self_monitor", "trend": round(trend_val, 4)}
                self.neural.emit_pain(
                    domain="soma_core",
                    metric=name,
                    stress=stress,
                    value=value,
                    metadata=metadata
                )
            if len(self.history) >= 30:
                mem_trend = self.trends["memory"]
                if mem_trend > 0.005:
                    proj = mem_trend * 60
                    sev = "warning" if mem_trend < 0.01 else "error"
                    self.neural.emit_self_fault(
                        fault_type="memory_leak",
                        reason=f"Memory inc {mem_trend*100:.2f}%/s, proj +{proj*100:.1f}% in 60s",
                        severity=sev
                    )
            time.sleep(1.0)
    
    def stop(self):
        self.running = False


# ============================================================================
# HEALTH MONITOR
# ============================================================================

class HealthMonitor:
    """Publishes organ health at 1 Hz (lymphatic channel)."""
    
    def __init__(self, component: str, version: str,
                 get_incoming: Callable, get_outgoing: Callable, get_sensors: Callable,
                 get_self_metrics: Optional[Callable] = None):
        self.component = component
        self.version = version
        self.health_version = "2.3"
        self.boot_count = 0
        self.start_time = time.time()
        self.get_incoming = get_incoming
        self.get_outgoing = get_outgoing
        self.get_sensors = get_sensors
        self.get_self_metrics = get_self_metrics
    
    def get_payload(self) -> Dict:
        payload = {
            "component": self.component,
            "version": self.version,
            "health_version": self.health_version,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            "uptime": int(time.time() - self.start_time),
            "boot_count": self.boot_count,
            "status": "OK",
            "incoming": self.get_incoming(),
            "outgoing": self.get_outgoing()
        }
        if self.get_self_metrics:
            self_metrics = self.get_self_metrics()
            if self_metrics:
                payload["self"] = self_metrics
        sensors = self.get_sensors()
        if sensors:
            payload["sensors"] = sensors
        return payload


# ============================================================================
# ORGAN HEALTH CHECKER
# ============================================================================

class OrganHealthChecker:
    """Periodically checks organ health and triggers failure if needed."""
    
    def __init__(self, orchestrator: SensorOrchestrator, neural: NeuralSignalingSystem,
                 self_monitor: Optional[SelfHealthManager] = None, interval: float = 5.0):
        self.orch = orchestrator
        self.neural = neural
        self.self_monitor = self_monitor
        self.interval = interval
        self.running = True
    
    def check_loop(self):
        while self.running:
            self._check()
            time.sleep(self.interval)
    
    def _check(self):
        sensors = self.orch.sensors
        total = len(sensors)
        if total == 0:
            return
        suspended = sum(1 for s in sensors.values() if s.suspended)
        in_pain = len([p for p in self.neural.pain_signals.values() if p.active])
        fault_ratio = suspended / total if total > 0 else 0
        pain_ratio = in_pain / total if total > 0 else 0
        cpu_crit = False
        mem_crit = False
        if self.self_monitor and len(self.self_monitor.history) > 0:
            last = self.self_monitor.history[-1]
            cpu_crit = last["cpu"] > 90
            mem_crit = last["memory"] > 95
        reason = {
            "sensor_faults": suspended,
            "active_pains": in_pain,
            "total_sensors": total,
            "fault_ratio": round(fault_ratio, 3),
            "pain_ratio": round(pain_ratio, 3),
            "cpu_critical": cpu_crit,
            "mem_critical": mem_crit
        }
        should_fail = fault_ratio > 0.5 or pain_ratio > 0.8 or (cpu_crit and mem_crit)
        is_failing = self.neural.organ_failure and self.neural.organ_failure.failing
        if should_fail and not is_failing:
            self.neural.emit_organ_failure(reason)
        elif not should_fail and is_failing:
            self.neural.emit_organ_recovery()
        elif should_fail and is_failing:
            self.neural.update_organ_failure(reason)
    
    def stop(self):
        self.running = False


# ============================================================================
# SOMA CORE MAIN
# ============================================================================

class SomaCore:
    """Main somatic perception organ."""
    
    def __init__(self, zenoh_config: str, rules_file: str, self_file: str, tech_file: str):
        self.name = "soma_core"
        self.version = "2.3.0"
        self.running = True
        self.pending_reset = False
        self.config_received = threading.Event()
        
        # Load static configurations (defaults)
        with open(rules_file, 'r') as f:
            self.rules_data = json.load(f)
        with open(self_file, 'r') as f:
            self.self_data = json.load(f)
        with open(tech_file, 'r') as f:
            self.tech_config = json.load(f)
        
        # In‑memory override (no persistence)
        self.override = OverrideManager(self.name)
        
        # Base Zenoh config
        base_conf = zenoh.Config()
        base_conf.from_file(zenoh_config)
        
        # Three dedicated Zenoh sessions
        self.nerve_session = zenoh.open(base_conf.clone())   # urgent signals
        self.hormonal_session = zenoh.open(base_conf.clone()) # not used by SomaCore
        self.meta_session = zenoh.open(base_conf.clone())     # config, health, validation
        
        # Dedicated schedulers
        self.nerve_scheduler = PubScheduler(
            publish_callback=self._publish_nerve,
            base_period=self.tech_config.get('scheduler', {}).get('base_period', 0.01),
            name=f"{self.name}_nerve_sched"
        )
        self.nerve_scheduler.start()
        
        self.meta_scheduler = PubScheduler(
            publish_callback=self._publish_meta,
            base_period=0.1,   # slower for meta channel
            name=f"{self.name}_meta_sched"
        )
        self.meta_scheduler.start()
        
        # Neural signaling system (uses nerve session)
        self.neural = NeuralSignalingSystem(self.name, self.nerve_session, self.nerve_scheduler)
        
        # Collectors
        self.system_collector = SystemMetricCollector()
        self.self_collector = SelfMetricCollector()
        
        # Profiles and rules
        self.profiles = self._load_profiles()
        self.rules = self._load_rules(self.rules_data, "metrics")
        self.self_rules = self._load_rules(self.self_data, "metrics", prefix="self_")
        all_rules = self.rules + self.self_rules
        
        # Battery monitor
        self.battery = BatteryMonitor()
        
        # Sensor orchestrator
        self.orch = SensorOrchestrator(self.tech_config, [self.system_collector, self.self_collector])
        self.sensors = self.orch.bootstrap(all_rules, self.profiles)
        
        # Acquisition manager
        self.acq = AcquisitionManager(
            self.tech_config, self.orch, [self.system_collector, self.self_collector],
            self.neural, self.battery, lambda: self.running
        )
        self.acq.start()
        
        # Self health manager
        self.self_monitor = SelfHealthManager(self.neural, self.self_collector)
        threading.Thread(target=self.self_monitor.monitoring_loop, daemon=True).start()
        
        # Organ health checker
        self.health_checker = OrganHealthChecker(self.orch, self.neural, self.self_monitor)
        threading.Thread(target=self.health_checker.check_loop, daemon=True).start()
        
        # Register nerves in schedulers
        self._register_nerves()
        
        # Health monitor
        self.health = HealthMonitor(
            component=self.name,
            version=self.version,
            get_incoming=self._get_incoming_stats,
            get_outgoing=self._get_outgoing_stats,
            get_sensors=self._get_sensor_summary,
            get_self_metrics=self._get_self_metrics
        )
        self.health.boot_count = self.override.get("boot_count", 0) + 1
        self.override.set("boot_count", self.health.boot_count)
        
        # Subscribe to configuration (retention)
        self.meta_session.declare_subscriber(f"config/{self.name}", self._on_config_update)
        
        # Subscribe to validation requests
        self.meta_session.declare_subscriber(
            f"config/validate/request/{self.name}",
            self._on_validate_request
        )
        
        # Subscribe to sleep signals
        self.meta_session.declare_subscriber("circa/phase", self._on_phase)
        self.meta_session.declare_subscriber("circa/sleep_weight", self._on_sleep_weight)
        
        # Wait a bit for config (if none received, use defaults)
        if not self.config_received.wait(timeout=2.0):
            print("⚠️ No configuration received, using defaults")
        
        # Start health loop
        threading.Thread(target=self._health_loop, daemon=True).start()
        
        print(f"✅ {self.name} v{self.version} started")
    
    def _load_profiles(self) -> Dict[str, SamplingProfile]:
        prof = {}
        for n, d in {**self.rules_data.get("sampling_profiles", {}), 
                    **self.self_data.get("sampling_profiles", {})}.items():
            prof[n] = SamplingProfile(n, d["frequency"], d.get("description", ""))
        return prof
    
    def _load_rules(self, data: Dict, section: str, prefix: str = "") -> List[AlertRule]:
        rules = []
        default = data.get("default_sampling_profile", "slow")
        for n, c in data.get(section, {}).items():
            rule = AlertRule(
                name=f"{prefix}{n}",
                alias=c["alias"],
                flux_topic=c["flux_topic"],
                pain_topic=c.get("pain_topic"),
                gt=c.get("threshold_GT"),
                lt=c.get("threshold_LT"),
                sampling_profile=c.get("sampling_profile", default),
                output_freq_min=c.get("output_freq_min", 1.0),
                output_freq_max=c.get("output_freq_max", 200.0),
                silence_below_threshold=c.get("silence_below_threshold", True),
                weight=c.get("weight", 1.0),
                description=c.get("description", "")
            )
            rules.append(rule)
        return rules
    
    def _register_nerves(self):
        for name, cfg in self.sensors.items():
            self.nerve_scheduler.add_nerve(cfg.nerve_alias, cfg.effective_period)
    
    def _publish_nerve(self, alias: str, payload: Any):
        for rule in self.rules + self.self_rules:
            if rule.alias == alias:
                self.nerve_session.put(rule.flux_topic, json.dumps(payload))
                return
    
    def _publish_meta(self, alias: str, payload: Any):
        if alias.startswith("health_"):
            self.meta_session.put(f"health/{self.name}", json.dumps(payload))
    
    def _on_config_update(self, sample):
        """Receive new configuration (retention topic)."""
        data = json.loads(sample.payload)
        new_config = data.get("config", {})
        version = data.get("version")
        print(f"📥 Received config v{version} for {self.name}")
        self.override.update(new_config)
        self._apply_config_updates(new_config)
        self.config_received.set()
    
    def _on_validate_request(self, sample):
        """Handle validation request from PersonalityCore."""
        req = json.loads(sample.payload)
        results = {}
        rejected = {}
        all_accepted = True
        for key, value in req["config"].items():
            # Basic validation (can be extended)
            if self._is_valid(key, value):
                results[key] = {"status": "accepted"}
            else:
                all_accepted = False
                results[key] = {"status": "rejected", "reason": "Invalid value"}
                rejected[key] = "Invalid value"
        response = {
            "request_id": req["request_id"],
            "accepted": all_accepted,
            "results": results,
            "rejected": rejected,
            "timestamp": datetime.now().isoformat()
        }
        self.meta_session.put(f"config/validate/response/{self.name}", json.dumps(response))
    
    def _is_valid(self, key: str, value: Any) -> bool:
        """Basic validation placeholder."""
        return True
    
    def _apply_config_updates(self, updates: Dict):
        """Apply configuration updates (to be implemented)."""
        pass
    
    def _on_phase(self, sample):
        data = json.loads(sample.payload)
        phase = data.get("phase")
        factors = {
            "deep_sleep": self.tech_config.get('sleep', {}).get('deep_sleep_factor', 0.1),
            "light_sleep": self.tech_config.get('sleep', {}).get('light_sleep_factor', 0.3),
            "dream": self.tech_config.get('sleep', {}).get('dream_factor', 0.5),
            "wake": self.tech_config.get('sleep', {}).get('wake_factor', 1.0)
        }
        factor = factors.get(phase, 1.0)
        self.nerve_scheduler.set_activity_factor(factor)
        self.meta_scheduler.set_activity_factor(factor)
    
    def _on_sleep_weight(self, sample):
        data = json.loads(sample.payload)
        weight = data.get("v", 0.0)
        if weight > 0.8:
            self.nerve_scheduler.set_activity_factor(0.2)
            self.meta_scheduler.set_activity_factor(0.2)
    
    def _get_incoming_stats(self) -> List:
        return [
            {"topic": "circa/phase", "last": datetime.now().isoformat(), "freq": 0.1},
            {"topic": "circa/sleep_weight", "last": datetime.now().isoformat(), "freq": 0.1},
            {"topic": f"config/{self.name}", "last": datetime.now().isoformat(), "freq": 0.01}
        ]
    
    def _get_outgoing_stats(self) -> List:
        return [
            {"topic": f"health/{self.name}", "last": datetime.now().isoformat(), "freq": 1.0}
        ]
    
    def _get_sensor_summary(self) -> Dict:
        details = {}
        for name in self.orch.get_active():
            cfg = self.orch.get_sensor(name)
            if not cfg:
                continue
            val = self.orch.get_cached_value(name)
            trend = self.orch.get_cached_trend(name)
            details[name] = {
                "value": round(val, 3) if val is not None else None,
                "trend": trend,
                "suspended": cfg.suspended,
                "degraded": cfg.degraded,
                "unstable": cfg.unstable
            }
        return {
            "active": len(self.orch.get_active()),
            "suspended": sum(1 for s in self.orch.get_active() if self.orch.get_sensor(s) and self.orch.get_sensor(s).suspended),
            "pain": len([p for p in self.neural.pain_signals.values() if p.active]),
            "detail": details
        }
    
    def _get_self_metrics(self) -> Optional[Dict]:
        if not self.self_monitor.history:
            return None
        last = self.self_monitor.history[-1]
        return {
            "cpu": round(last["cpu"], 1),
            "memory": round(last["memory"], 1),
            "threads": int(self.self_collector._get_threads()),
            "fds": int(self.self_collector._get_fds()),
            "cpu_trend": round(self.self_monitor.trends.get("cpu", 0.0), 4),
            "memory_trend": round(self.self_monitor.trends.get("memory", 0.0), 4)
        }
    
    def _health_loop(self):
        while self.running:
            payload = self.health.get_payload()
            self.meta_session.put(f"health/{self.name}", json.dumps(payload))
            time.sleep(1.0)
    
    def stop(self):
        self.running = False
        self.self_monitor.stop()
        self.health_checker.stop()
        self.neural.cleanup()
        self.nerve_scheduler.stop()
        self.meta_scheduler.stop()
        self.nerve_session.close()
        self.hormonal_session.close()
        self.meta_session.close()
        print(f"🛑 {self.name} stopped")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python soma_core.py <zenoh_config> <rules_file> <self_file> <tech_file>")
        sys.exit(1)
    
    core = SomaCore(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    try:
        while True:
            time.sleep(1)
            if core.pending_reset:
                print("🔄 Performing deferred reset after deep sleep")
                # Reset logic would go here
                core.pending_reset = False
    except KeyboardInterrupt:
        core.stop()