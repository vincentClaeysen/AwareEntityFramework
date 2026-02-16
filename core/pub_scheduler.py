import threading
import time
import logging
from typing import Dict, Optional, Callable, Any, List

logger = logging.getLogger(__name__)

class PubScheduler(threading.Thread):
    """
    Thread de publication optimisé pour SomaCore.
    Gère des flux à périodes variables en un seul thread.
    Chaque nerf possède un compteur (0..1) et un pas = base_period / période_cible.
    Quand le compteur atteint 1, on publie et on remet à 0.
    Les changements de période sont appliqués au moment de la publication suivante.
    """
    
    def __init__(self, 
                 publish_callback: Callable[[str, Any], None],
                 base_period: float = 0.01,
                 name: str = "PubScheduler"):
        """
        Args:
            publish_callback: fonction appelée pour publier (alias, payload)
            base_period: période de base du scheduler en secondes (ex: 0.01 = 10ms)
            name: nom du thread
        """
        super().__init__(daemon=True, name=name)
        self.base_period = base_period
        self.publish = publish_callback
        self.running = True
        self._lock = threading.RLock()  # RLock pour réentrance
        
        # État des nerfs : alias -> [compteur (float), pas_courant (float)]
        self.nerfs: Dict[str, List[float]] = {}
        
        # Dernier payload reçu pour chaque nerf
        self.registre: Dict[str, Any] = {}
        
        # Demandes de changement de pas (alias -> nouveau_pas)
        self.nouveaux_pas: Dict[str, float] = {}
        
        # Statistiques internes
        self.stats = {
            'cycles': 0,
            'publications': 0,
            'erreurs': 0
        }

    def _period_to_pas(self, periode: float) -> float:
        """Convertit une période en pas, borné pour éviter les dépassements."""
        return self.base_period / max(periode, self.base_period)

    def add_nerf(self, alias: str, periode_cible: float):
        """
        Ajoute un nerf à gérer.
        
        Args:
            alias: identifiant unique du nerf
            periode_cible: période de publication souhaitée (en secondes)
        """
        pas = self._period_to_pas(periode_cible)
        with self._lock:
            self.nerfs[alias] = [0.0, pas]
            self.registre.setdefault(alias, None)
        logger.debug(f"✅ Nerf ajouté: {alias} (période={periode_cible}s, pas={pas:.6f})")

    def update_payload(self, alias: str, payload: Any):
        """
        Met à jour le payload d'un nerf (appelé par un thread producteur).
        
        Args:
            alias: identifiant du nerf
            payload: dernier payload à publier (None si pas de donnée)
        """
        with self._lock:
            self.registre[alias] = payload

    def update_period(self, alias: str, nouvelle_periode: float):
        """
        Demande un changement de période pour un nerf.
        Le changement sera effectif au prochain cycle de publication.
        
        Args:
            alias: identifiant du nerf
            nouvelle_periode: nouvelle période cible (en secondes)
        """
        nouveau_pas = self._period_to_pas(nouvelle_periode)
        with self._lock:
            self.nouveaux_pas[alias] = nouveau_pas
        logger.debug(f"🔄 Changement période demandé: {alias} -> {nouvelle_periode}s (pas={nouveau_pas:.6f})")

    def remove_nerf(self, alias: str):
        """
        Supprime un nerf du scheduler.
        
        Args:
            alias: identifiant du nerf à supprimer
        """
        with self._lock:
            if alias in self.nerfs:
                del self.nerfs[alias]
            if alias in self.registre:
                del self.registre[alias]
            if alias in self.nouveaux_pas:
                del self.nouveaux_pas[alias]
        logger.debug(f"❌ Nerf supprimé: {alias}")

    def reset(self):
        """Réinitialise complètement le scheduler (vide tous les nerfs et registres)."""
        with self._lock:
            self.nerfs.clear()
            self.registre.clear()
            self.nouveaux_pas.clear()
            self.stats = {
                'cycles': 0,
                'publications': 0,
                'erreurs': 0
            }
        logger.info("🔄 PubScheduler réinitialisé")

    def run(self):
        """Boucle principale de gestion des flux."""
        logger.info(f"▶️ PubScheduler démarré (base_period={self.base_period}s)")
        
        while self.running:
            cycle_start = time.perf_counter()
            
            # Capture atomique des références (minimise le temps sous verrou)
            with self._lock:
                # On copie les clés pour itérer ensuite sans bloquer
                nerfs_items = list(self.nerfs.items())
            
            # Traitement de chaque nerf
            for alias, state in nerfs_items:
                # Incrémentation du compteur (potentiel)
                state[0] += state[1]
                
                if state[0] >= 1.0:
                    # Seuil de décharge atteint
                    state[0] = 0.0
                    
                    # Récupération du payload et application d'un éventuel nouveau pas
                    with self._lock:
                        payload = self.registre.get(alias)
                        if alias in self.nouveaux_pas:
                            old_pas = state[1]
                            state[1] = self.nouveaux_pas.pop(alias)
                            logger.debug(f"📊 Nerf {alias}: pas changé {old_pas:.6f} -> {state[1]:.6f}")
                    
                    # Publication (hors verrou pour ne pas bloquer)
                    if payload is not None:
                        try:
                            self.publish(alias, payload)
                            with self._lock:
                                self.stats['publications'] += 1
                        except Exception as e:
                            with self._lock:
                                self.stats['erreurs'] += 1
                            logger.error(f"❌ Erreur publication {alias}: {e}")
            
            # Maintien de la cadence
            elapsed = time.perf_counter() - cycle_start
            sleep_time = self.base_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Mise à jour des statistiques
            with self._lock:
                self.stats['cycles'] += 1
                if self.stats['cycles'] % 1000 == 0:
                    logger.debug(f"📈 Stats: cycles={self.stats['cycles']}, "
                               f"pub={self.stats['publications']}, "
                               f"err={self.stats['erreurs']}")

    def stop(self):
        """Arrêt propre du scheduler."""
        logger.info("🛑 Arrêt du PubScheduler...")
        self.running = False
        # Petit délai pour permettre la fin du cycle
        time.sleep(self.base_period * 2)
        logger.info("✅ PubScheduler arrêté")

    def get_stats(self) -> Dict:
        """Retourne une copie des statistiques courantes."""
        with self._lock:
            return self.stats.copy()

    def get_nerf_count(self) -> int:
        """Retourne le nombre de nerfs gérés."""
        with self._lock:
            return len(self.nerfs)