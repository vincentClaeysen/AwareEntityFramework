import sys
import os
import signal
import threading
import time
import json
import logging
from collections import deque
import numpy as np
import zenoh
import pygame
import math

# --- Configuration des logs ---
logger = logging.getLogger("StreamsMonitor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
logger.addHandler(handler)

# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    @staticmethod
    def load(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Validation basique
            required_keys = ['element_types', 'groups', 'oscilloscope']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise

# ============================================================================
# Color Pulse Engine
# ============================================================================

class ColorPulseEngine:
    def __init__(self, config):
        self.config = config
        self.palettes = {}
        for type_name, type_config in config['element_types'].items():
            self.palettes[type_name] = {
                'dark': self._hex_to_rgb(type_config['color_palette']['dark']),
                'medium': self._hex_to_rgb(type_config['color_palette']['medium']),
                'bright': self._hex_to_rgb(type_config['color_palette']['bright'])
            }
    
    def get_color_from_frequency(self, frequency, type_name):
        type_config = self.config['element_types'][type_name]
        palette = self.palettes[type_name]
        low, high = type_config['frequency_thresholds']['low'], type_config['frequency_thresholds']['high']
        
        if frequency < low:
            t = frequency / low if low > 0 else 0
            return self._interpolate_rgb(palette['dark'], palette['medium'], t)
        elif frequency < high:
            t = (frequency - low) / (high - low) if (high - low) > 0 else 0
            return self._interpolate_rgb(palette['medium'], palette['bright'], t)
        return palette['bright']

    def get_halo_intensity(self, frequency, type_name):
        """Calcule l'intensité du halo (0-1) basée sur la fréquence"""
        type_config = self.config['element_types'][type_name]
        low, high = type_config['frequency_thresholds']['low'], type_config['frequency_thresholds']['high']
        
        if frequency <= low:
            return 0.2  # Halo faible
        elif frequency >= high:
            return 1.0  # Halo maximum
        else:
            # Interpolation linéaire entre low et high
            return 0.2 + 0.8 * ((frequency - low) / (high - low))

    @staticmethod
    def _interpolate_rgb(c1, c2, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))

    @staticmethod
    def _hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ============================================================================
# Stream Line (Modèle de données)
# ============================================================================

class StreamLine:
    def __init__(self, topic, type_name, color_engine, window_duration=5.0, max_impulses=1000):
        self.topic = topic
        self.type = type_name
        self.color_engine = color_engine
        self.window_duration = window_duration  # Durée de la fenêtre temporelle
        self.max_impulses = max_impulses
        self.impulse_history = deque(maxlen=max_impulses)  # (timestamp, payload)
        self.last_message_time = None
        self.current_frequency = 0.0
        self.frequency_history = deque(maxlen=10)
        self.last_payload = {}
        self.message_count = 0
        self._lock = threading.Lock()
        
    def on_message_received(self, payload):
        with self._lock:
            now = time.time()
            self.impulse_history.append((now, payload.copy()))  # Stocker le payload complet
            self.message_count += 1
            self.last_payload = payload.copy()
            
            # Calcul de la fréquence instantanée
            if self.last_message_time:
                dt = now - self.last_message_time
                if dt > 0.001:
                    instant_freq = 1.0 / dt
                    if instant_freq < 1000:  # Max 1000 Hz
                        self.frequency_history.append(instant_freq)
                        self.current_frequency = np.mean(self.frequency_history) if self.frequency_history else 0.0
            
            self.last_message_time = now
    
    def get_recent_impulses(self, current_time):
        """Retourne les impulsions dans la fenêtre temporelle"""
        with self._lock:
            cutoff = current_time - self.window_duration
            recent = [(ts, payload) for ts, payload in self.impulse_history if ts >= cutoff]
            return recent

# ============================================================================
# Groupe pliable/dépliable
# ============================================================================

class CollapsibleGroup:
    def __init__(self, name, elements, x, y, width):
        self.name = name
        self.elements = elements
        self.x = x
        self.y = y
        self.width = width
        self.expanded = True
        self.height = 30  # Hauteur du titre uniquement
        self.collapsed_height = 30
        self.expanded_height = 30 + len(elements) * 120  # Titre + éléments
        
    def toggle(self):
        self.expanded = not self.expanded
        
    def get_height(self):
        return self.expanded_height if self.expanded else self.collapsed_height
    
    def contains_point(self, px, py):
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + 30)  # Seulement la zone du titre est cliquable

# ============================================================================
# Main Monitor (Pygame Edition)
# ============================================================================

class StreamsMonitor:
    def __init__(self, config_filepath):
        pygame.init()
        
        # Configuration
        self.config = ConfigLoader.load(config_filepath)
        self.color_engine = ColorPulseEngine(self.config)
        self.window_duration = self.config.get('oscilloscope', {}).get('window_duration', 5.0)
        self.target_fps = self.config.get('oscilloscope', {}).get('fps', 30)
        
        # Signal handlers pour fermeture propre
        self._setup_signal_handlers()
        
        # Zenoh Init
        self.zenoh_session = self._setup_zenoh()
        
        # Pygame Setup - Taille initiale 800x600
        self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
        pygame.display.set_caption("SAS - Stream Monitor")
        
        # Streams & Groups
        self.streams = {}
        self.groups = []
        self._setup_subscribers()
        self._setup_groups()
        
        # Polices
        self.font_main = pygame.font.SysFont("Monospace", 16, bold=True)
        self.font_small = pygame.font.SysFont("Monospace", 11)
        self.font_payload = pygame.font.SysFont("Monospace", 9)
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.last_stats_time = time.time()
        self.frame_count = 0
        self.scroll_offset = 0
        self.max_scroll = 0
        
        logger.info(f"StreamsMonitor initialized with {self.target_fps} FPS target")

    def _setup_signal_handlers(self):
        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _setup_zenoh(self):
        conf = zenoh.Config()
        zc = self.config.get('zenoh', {})
        
        if 'mode' in zc:
            conf.insert_json5("mode", json.dumps(zc['mode']))
        
        if 'connect' in zc and 'endpoints' in zc['connect']:
            endpoints = zc['connect']['endpoints']
            if isinstance(endpoints, list):
                conf.insert_json5("connect/endpoints", json.dumps(endpoints))
        
        try:
            session = zenoh.open(conf)
            logger.info("Zenoh session opened successfully")
            return session
        except Exception as e:
            logger.error(f"Failed to open Zenoh session: {e}")
            raise

    def _setup_subscribers(self):
        for group in self.config['groups']:
            for el in group['elements']:
                topic = el['topic']
                self.streams[topic] = StreamLine(
                    topic, 
                    el['type'], 
                    self.color_engine, 
                    self.window_duration
                )
                try:
                    self.zenoh_session.declare_subscriber(topic, self._zenoh_callback)
                    logger.info(f"Subscribed to topic: {topic}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to {topic}: {e}")

    def _setup_groups(self):
        """Initialise les groupes pliables"""
        y_offset = 60
        for group_config in self.config['groups']:
            elements = [self.streams[el['topic']] for el in group_config['elements']]
            group = CollapsibleGroup(
                group_config['name'],
                elements,
                10,  # x
                y_offset,  # y
                self.screen.get_width() - 20  # width
            )
            self.groups.append(group)
            y_offset += group.get_height() + 10

    def _zenoh_callback(self, sample):
        topic = str(sample.key_expr)
        try:
            payload = json.loads(sample.payload.to_string())
            if topic in self.streams:
                self.streams[topic].on_message_received(payload)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for topic {topic}: {e}")
        except Exception as e:
            logger.error(f"Zenoh callback error for {topic}: {e}")

    def draw(self):
        self.screen.fill((5, 5, 5))
        win_w, win_h = self.screen.get_size()
        
        # Dessiner une bordure autour de la fenêtre
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, win_w, win_h), 2)
        
        # Calcul FPS
        self.frame_count += 1
        now = time.time()
        if now - self.last_stats_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_stats_time = now
        
        # Header
        header_text = f"🧠 STREAMS Monitor - {self.window_duration}s | FPS: {getattr(self, 'fps', 0)}/{self.target_fps}"
        header_surf = self.font_main.render(header_text, True, (0, 255, 255))
        self.screen.blit(header_surf, (win_w//2 - header_surf.get_width()//2, 15))
        
        # Mise à jour des positions des groupes (pour le scrolling)
        y_offset = 60 - self.scroll_offset
        for group in self.groups:
            group.y = y_offset
            y_offset += group.get_height() + 10
        
        # Calcul du scroll maximum
        total_height = sum(g.get_height() + 10 for g in self.groups) + 60
        self.max_scroll = max(0, total_height - win_h + 50)
        
        # Dessiner les groupes
        for group in self.groups:
            self._draw_group(group, win_w)

    def _draw_group(self, group, win_w):
        # Fond du groupe
        group_rect = pygame.Rect(group.x, group.y, group.width, group.get_height())
        
        # Titre du groupe (toujours visible)
        title_rect = pygame.Rect(group.x, group.y, group.width, 30)
        
        # Fond du titre avec dégradé
        for i in range(30):
            color_value = 20 + i
            pygame.draw.line(self.screen, (color_value, color_value, color_value), 
                           (group.x, group.y + i), (group.x + group.width, group.y + i))
        
        # Bordure du titre
        pygame.draw.rect(self.screen, (60, 60, 60), title_rect, 1)
        
        # Icône de pliage/dépliage
        icon = "▼" if group.expanded else "▶"
        icon_surf = self.font_main.render(icon, True, (0, 191, 255))
        self.screen.blit(icon_surf, (group.x + 5, group.y + 7))
        
        # Nom du groupe
        name_surf = self.font_main.render(f"GROUPE: {group.name}", True, (0, 191, 255))
        self.screen.blit(name_surf, (group.x + 25, group.y + 7))
        
        # Si le groupe est expandé, dessiner les streams
        if group.expanded:
            # Légère ombre pour la séparation
            pygame.draw.line(self.screen, (40, 40, 40), 
                           (group.x, group.y + 30), (group.x + group.width, group.y + 30), 1)
            
            stream_y = group.y + 40
            for stream in group.elements:
                # Largeurs adaptatives
                osc_width = int(win_w * 0.55)  # 55% pour l'oscilloscope
                payload_width = win_w - osc_width - 40  # Le reste pour le payload
                
                self._draw_stream(stream, group.x + 10, stream_y, osc_width, 90, payload_width)
                stream_y += 100

  
def _draw_stream(self, stream, x, y, w, h, payload_width):
    now = time.time()
    
    # Couleur de base basée sur la fréquence
    base_color = stream.color_engine.get_color_from_frequency(stream.current_frequency, stream.type)
    halo_intensity = stream.color_engine.get_halo_intensity(stream.current_frequency, stream.type)
    
    # ===== DESSIN DE L'OSCILLOSCOPE =====
    # Dessiner le halo uniquement sur le cadre
    for i in range(3, 0, -1):
        alpha = int(20 * halo_intensity * (i / 3))
        if alpha > 0:
            # Dessiner le halo comme un contour plus épais
            halo_rect = pygame.Rect(x - i, y - i, w + 2*i, h + 2*i)
            pygame.draw.rect(self.screen, (*base_color, alpha), halo_rect, 2)
    
    # Fond sombre
    pygame.draw.rect(self.screen, (10, 10, 10), (x, y, w, h))
    pygame.draw.rect(self.screen, base_color, (x, y, w, h), 1)
    
    # Ligne horizontale centrale (le "temps")
    mid_y = y + h // 2
    line_color = self._interpolate_color(base_color, (200, 200, 200), 0.7)
    pygame.draw.line(self.screen, line_color, (x, mid_y), (x + w, mid_y), 2)
    
    # Récupérer les impulsions récentes
    recent_impulses = stream.get_recent_impulses(now)
    
    # Hauteur fixe des pics
    peak_h = int(h * 0.4)  # 40% de la hauteur totale
    
    # Trouver le pic le plus récent (s'il existe)
    most_recent_pic = None
    if recent_impulses:
        # Le dernier pic est celui avec le timestamp le plus grand
        most_recent_pic = max(recent_impulses, key=lambda x: x[0])
        recent_timestamp, recent_payload = most_recent_pic
    
    # Dessiner les pics (forme de /\ partant de la ligne centrale)
    for timestamp, payload in recent_impulses:
        # Calculer la position x basée sur le timestamp
        age = now - timestamp
        progress = 1.0 - (age / stream.window_duration)  # 0 = vieux, 1 = récent
        px = x + int(w * progress)
        
        # Couleur du pic (plus intense si récent)
        color_intensity = 0.5 + 0.5 * (1.0 - age / stream.window_duration)
        pic_color = self._interpolate_color(base_color, (255, 255, 255), color_intensity)
        
        # Dessiner le pic en forme de /\ partant de la ligne centrale
        if px > x and px < x + w:
            # Ligne gauche du /\ (de la ligne centrale vers le haut)
            pygame.draw.line(self.screen, pic_color,
                           (px - 3, mid_y - peak_h),  # Sommet gauche
                           (px, mid_y),  # Base au centre
                           2)
            # Ligne droite du /\ (de la ligne centrale vers le haut)
            pygame.draw.line(self.screen, pic_color,
                           (px + 3, mid_y - peak_h),  # Sommet droit
                           (px, mid_y),  # Base au centre
                           2)
            
            # Petit point au sommet
            pygame.draw.circle(self.screen, (255, 255, 255), (px, mid_y - peak_h), 2)
            
            # Petite ligne verticale à la base pour renforcer l'effet
            pygame.draw.line(self.screen, pic_color,
                           (px, mid_y - 1),
                           (px, mid_y + 1), 1)
    
    # === POINT BLANC SUR LE PIC LE PLUS RÉCENT ===
    if most_recent_pic:
        # Calculer la position du pic le plus récent
        recent_age = now - recent_timestamp
        recent_progress = 1.0 - (recent_age / stream.window_duration)
        recent_px = x + int(w * recent_progress)
        
        if recent_px > x and recent_px < x + w:
            # Dessiner le point blanc au sommet du pic le plus récent
            pygame.draw.circle(self.screen, (255, 255, 255), (recent_px, mid_y - peak_h), 5)
            # Petit halo autour du point
            pygame.draw.circle(self.screen, (*base_color, 100), (recent_px, mid_y - peak_h), 8, 1)
    else:
        # Si pas de pic, garder le point à l'extrême droite comme avant
        pygame.draw.circle(self.screen, (255, 255, 255), (x + w, mid_y), 3)
    
    # Informations textuelles sur l'oscilloscope
    topic_color = (0, 191, 255) if stream.type == "nerf" else (255, 165, 0)
    txt_topic = self.font_small.render(f"{stream.topic}", True, topic_color)
    self.screen.blit(txt_topic, (x + 5, y + 5))
    
    # Fréquence et compteur
    freq_text = f"{stream.current_frequency:.1f} Hz | {stream.message_count} msgs"
    txt_freq = self.font_small.render(freq_text, True, base_color)
    self.screen.blit(txt_freq, (x + 5, y + h - 35))
    
    # ===== ZONE D'AFFICHAGE DU PAYLOAD =====
    payload_x = x + w + 10
    payload_y = y
    payload_h = h
    
    # Fond pour le payload
    pygame.draw.rect(self.screen, (15, 15, 15), (payload_x, payload_y, payload_width, payload_h))
    pygame.draw.rect(self.screen, base_color, (payload_x, payload_y, payload_width, payload_h), 1)
    
    # Afficher le dernier payload
    if stream.last_payload:
        # Formater le payload de façon lisible
        payload_str = json.dumps(stream.last_payload)
        
        # Découper le payload en lignes
        max_chars = int(payload_width / 6)
        lines = []
        current_line = ""
        
        for char in payload_str:
            if len(current_line) < max_chars:
                current_line += char
            else:
                lines.append(current_line)
                current_line = char
        
        if current_line:
            lines.append(current_line)
        
        # Limiter à 3 lignes maximum
        lines = lines[:3]
        
        # Afficher chaque ligne
        line_y = payload_y + 10
        for line in lines:
            payload_surf = self.font_payload.render(line, True, (200, 200, 200))
            self.screen.blit(payload_surf, (payload_x + 5, line_y))
            line_y += 12
    else:
        # Aucun message reçu
        no_data = self.font_payload.render("⏳...", True, (100, 100, 100))
        self.screen.blit(no_data, (payload_x + 5, payload_y + 15))  
 
   def handle_click(self, pos):
        """Gère les clics souris pour plier/déplier les groupes"""
        for group in self.groups:
            if group.contains_point(pos[0], pos[1]):
                group.toggle()
                logger.debug(f"Group {group.name} toggled: {'expanded' if group.expanded else 'collapsed'}")
                return True
        return False

    def handle_scroll(self, y_scroll):
        """Gère le scrolling"""
        self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset - y_scroll))

    def run(self):
        logger.info(f"Starting visualization loop at {self.target_fps} FPS")
        frame_time = 1.0 / self.target_fps
        next_frame_time = time.time()
        
        try:
            while self.running:
                # Gestion des événements
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                        # Recalculer les largeurs de groupes
                        for group in self.groups:
                            group.width = self.screen.get_width() - 20
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Clic gauche
                            self.handle_click(event.pos)
                        elif event.button == 4:  # Molette haut
                            self.handle_scroll(30)
                        elif event.button == 5:  # Molette bas
                            self.handle_scroll(-30)
                
                # Rendu
                self.draw()
                pygame.display.flip()
                
                # Contrôle du FPS
                current_time = time.time()
                if current_time < next_frame_time:
                    time.sleep(max(0, next_frame_time - current_time))
                next_frame_time += frame_time
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up resources...")
        if self.zenoh_session:
            try:
                self.zenoh_session.close()
                logger.info("Zenoh session closed")
            except Exception as e:
                logger.error(f"Error closing Zenoh session: {e}")
        pygame.quit()
        logger.info("Pygame closed")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "streams_config.json"
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        logger.info(f"StreamsMonitor dir : {script_dir}")

        config_filepath = os.path.join(script_dir, config_file)

        logger.info(f"Starting StreamsMonitor with config: {config_filepath}")
        viz = StreamsMonitor(config_filepath)
        viz.run()
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        sys.exit(1)
