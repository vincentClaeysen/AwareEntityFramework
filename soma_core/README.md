# SomaCore – Bio-inspired Sensor Acquisition Organ

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Version](https://img.shields.io/badge/version-2.3.0-green)

**SomaCore** is a bio‑inspired sensor acquisition **organ**. It is one of the many organs that constitute a synthetic entity within the **AwareEntityFramework (AEF)**. Its role is to continuously monitor:

- **External sensors** : hardware metrics (CPU, memory, temperature, battery)
- **Internal sensors** : its own process metrics (CPU, memory, threads)

It translates these readings into **frequency‑modulated nerve signals** using three distinct types of neural signaling, and communicates its state via **lymphatic channels** (health and configuration). It receives dynamic configuration via a dedicated retention topic and can validate proposed changes before acceptance.

SomaCore is a **functional digital organ** with autonomous behavior, self‑perception, graceful degradation, and deep‑sleep reset, designed to work in concert with other organs like SensoryCore, CircaCore, and the NAA.

---

## ✨ Key Features

### 🧠 **Three Types of Neural Signaling**
| Type | Name | Description | Example |
|------|------|-------------|---------|
| 🔴 **Diagnostics** | Spike (event) | Discrete signals for technical faults | Sensor timeout, exception, resource exhaustion |
| 🟡 **Pain** | Graduated + Heartbeat | Continuous signals for abnormal values | CPU > 90%, memory leak, temperature critical |
| 🔵 **Organ Failure** | Hybrid (spike + heartbeat) | Global organ failure | >50% sensors dead, CPU + RAM critical |

### 🌡️ **Dual Perception**
- **External** (`domain="soma"`) : System metrics (CPU, memory, temperature, battery)
- **Internal** (`domain="soma_core"`) : Process metrics (self_cpu, self_memory, self_threads)

### ⚡ **Frequency = Urgency, `v` = Amplitude**
- Frequency (0–50 Hz) codes the **stress level** (urgency)
- Payload field `"v"` contains the **amplitude** (raw value)

### 💤 **Sleep‑Aware Operation**
- Listens to `circa/phase` and `circa/sleep_weight` via `meta_session`
- Reduces activity during sleep (deep sleep factor 0.1, dream factor 0.5)
- Full reset after deep sleep exit

### 🩺 **Self‑Health Monitoring**
- Tracks its own CPU, memory, threads
- Detects memory leaks via trend analysis (linear regression)
- Emits self‑diagnostic spikes on critical conditions

### 🔧 **Dynamic Configuration**
- No persistent override file – in‑memory only
- Configuration received on `config/soma_core` (retention topic)
- Validation endpoint: `config/validate/request/soma_core` / `config/validate/response/soma_core`
- Starts with built‑in defaults if no config received

### 📊 **Health & Diagnostics**
- `health/soma_core` at 1 Hz (via `meta_session`)
- Includes incoming/outgoing topic stats, sensor summaries, trends
- Organ failure detection with heartbeat and spike events (via `nerve_session`)

### 🔌 **Two Zenoh Sessions**
| Hub | Session | Usage |
|-----|---------|-------|
| **Nerve** | `nerve_session` | Diagnostics, pain, organ (urgent signals) |
| **Meta** | `meta_session` | Configuration, health, validation |

---

## 📁 Project Structure
