import zenoh
import time
import json
import logging

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ClockPerceptive")

def clock_perceptive():
    conf = zenoh.Config()
    conf.from_file("zenoh_config.json5")
    z = zenoh.open(conf)

    pub = z.declare_publisher('clock/perceptive')
    
    cycle_id = 0
    dt_cycle = 1.0 / 25.0 # = 0.04 -> 40ms (25Hz)

    logger.info("ClockPerceptive LIFE 2026 actif.")
    logger.info(f"[CLOCK] Générateur d'impulsions Perceptif (periode: {dt_cycle}s).")
    
    while True:
        start_loop = time.perf_counter()
        
        #logger.info(f"[TICK] #{cycle_id}")

        # On émet pour les restes 0, 1 et 2 (soit 3 ticks consécutifs)
        window_pos = cycle_id % 250
        
        if 0 <= window_pos <= 2:
            payload = {
                "tick": cycle_id,
                "burst_pos": window_pos # 0, 1 ou 2 pour aider le subscriber
                #"ts": time.time()
            }            
            pub.put(json.dumps(payload).encode('utf-8'))
            logger.info(f"[SYNCHRO] Emission tick perceptif #{cycle_id} (Position {window_pos}/2).")

        cycle_id += 1
            
        # Maintien de la fréquence stricte
        elapsed = time.perf_counter() - start_loop
        time.sleep(max(0, dt_cycle - elapsed))

if __name__ == "__main__":
    clock_perceptive()