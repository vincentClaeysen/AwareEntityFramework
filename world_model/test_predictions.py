#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion Tracking par Flux Optique Sparse - Version corrigée
- Flux optique fonctionnel (prev_gray → curr_gray)
- Attention par proximité simulée (basée sur taille du blob)
- Prédiction linéaire simple (position future)
- Architecture multithread
"""

import cv2
import numpy as np
import math
import threading
import time
import queue
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Paramètres globaux"""
    # Vidéo
    width: int = 640
    height: int = 400
    target_fps: int = 20
    
    # Détection de points (Shi-Tomasi)
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: int = 7
    block_size: int = 7
    
    # Flux optique (Lucas-Kanade)
    lk_win_size: Tuple[int, int] = (15, 15)
    lk_max_level: int = 2
    
    # Clustering
    max_vector_distance: float = 15.0
    max_vector_angle_diff: float = 30.0    # degrés
    min_cluster_size: int = 5
    max_cluster_size: int = 500
    
    # Suivi
    max_match_distance: float = 50.0
    max_frames_missing: int = 5
    history_len: int = 5
    
    # Prédiction
    prediction_frames: int = 5              # prédire dans N frames
    prediction_color: Tuple[int, int, int] = (255, 255, 0)  # cyan
    
    # Attention par proximité (simulée par taille du blob)
    # En l'absence de depth map réelle, on utilise la surface du blob
    # plus c'est gros, plus c'est "proche"
    attention_foveal_min_area: int = 3000    # pixels
    attention_peripheral_min_area: int = 800
    
    # Affichage
    vector_scale: float = 1.5
    rect_color_foveal: Tuple[int, int, int] = (0, 255, 0)       # vert électrique
    rect_color_peripheral: Tuple[int, int, int] = (0, 200, 255) # orange
    rect_color_passive: Tuple[int, int, int] = (100, 100, 100)  # gris
    vector_color: Tuple[int, int, int] = (0, 255, 255)          # jaune flashy
    text_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Queues
    queue_maxsize: int = 5


config = Config()


# ============================================================
# CLASSE : BLOB DÉTECTÉ
# ============================================================

class DetectedBlob:
    """Blob détecté dans une frame"""
    
    def __init__(self, points: np.ndarray, vectors: np.ndarray):
        self.points = points
        self.vectors = vectors
        self.centroid = self._compute_centroid()
        self.bbox = self._compute_bbox()
        self.area = self.bbox[2] * self.bbox[3]
        self.attention_level = self._compute_attention_level()
        
    def _compute_centroid(self) -> Tuple[int, int]:
        if len(self.points) == 0:
            return (0, 0)
        return (int(np.mean(self.points[:, 0])), int(np.mean(self.points[:, 1])))
    
    def _compute_bbox(self) -> Tuple[int, int, int, int]:
        if len(self.points) == 0:
            return (0, 0, 0, 0)
        x_min = int(self.points[:, 0].min())
        x_max = int(self.points[:, 0].max())
        y_min = int(self.points[:, 1].min())
        y_max = int(self.points[:, 1].max())
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _compute_attention_level(self) -> str:
        """Simulation de l'attention par proximité basée sur la taille"""
        if self.area >= config.attention_foveal_min_area:
            return "foveal"
        elif self.area >= config.attention_peripheral_min_area:
            return "peripheral"
        else:
            return "passive"
    
    def get_median_vector(self) -> Tuple[float, float]:
        if len(self.vectors) == 0:
            return (0.0, 0.0)
        return (float(np.median(self.vectors[:, 0])), float(np.median(self.vectors[:, 1])))


# ============================================================
# CLASSE : BLOB SUIVI
# ============================================================

class TrackedBlob:
    """Blob suivi sur plusieurs frames"""
    _next_id = 0
    
    def __init__(self, detected: DetectedBlob):
        self.id = TrackedBlob._next_id
        TrackedBlob._next_id += 1
        
        self.centroid = detected.centroid
        self.bbox = detected.bbox
        self.area = detected.area
        self.attention_level = detected.attention_level
        self.points = detected.points.copy()
        self.vectors = detected.vectors.copy()
        self.positions = deque(maxlen=config.history_len)
        self.positions.append(detected.centroid)
        self.velocity = detected.get_median_vector()
        self.frames_missing = 0
        
    def update(self, detected: DetectedBlob):
        self.centroid = detected.centroid
        self.bbox = detected.bbox
        self.area = detected.area
        self.attention_level = detected.attention_level
        self.points = detected.points.copy()
        self.vectors = detected.vectors.copy()
        self.positions.append(detected.centroid)
        self.velocity = detected.get_median_vector()
        self.frames_missing = 0
        
    def mark_missing(self):
        self.frames_missing += 1
        
    def is_active(self) -> bool:
        return self.frames_missing < config.max_frames_missing
    
    def get_smoothed_position(self) -> Tuple[int, int]:
        if len(self.positions) == 0:
            return self.centroid
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        return (int(np.mean(xs)), int(np.mean(ys)))
    
    def predict_position(self, frames_ahead: int = config.prediction_frames) -> Tuple[int, int]:
        """Prédit la position future basée sur la vitesse actuelle"""
        cx, cy = self.get_smoothed_position()
        vx, vy = self.velocity
        return (int(cx + vx * frames_ahead), int(cy + vy * frames_ahead))
    
    def get_rect_color(self) -> Tuple[int, int, int]:
        """Couleur du rectangle selon le niveau d'attention"""
        if self.attention_level == "foveal":
            return config.rect_color_foveal
        elif self.attention_level == "peripheral":
            return config.rect_color_peripheral
        else:
            return config.rect_color_passive


# ============================================================
# CLASSE : DÉTECTEUR DE MOUVEMENT (Flux Optique Sparse - CORRIGÉ)
# ============================================================

class MotionDetector:
    """Détection de mouvement par flux optique sparse"""
    
    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.feature_params = dict(
            maxCorners=config.max_corners,
            qualityLevel=config.quality_level,
            minDistance=config.min_distance,
            blockSize=config.block_size
        )
        self.lk_params = dict(
            winSize=config.lk_win_size,
            maxLevel=config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def process(self, gray: np.ndarray) -> List[DetectedBlob]:
        """
        Traite une frame et retourne la liste des blobs détectés
        """
        # Première frame : initialisation
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            if self.prev_points is not None:
                self.prev_points = self.prev_points.reshape(-1, 2)
            return []
        
        # Détection des points dans la frame courante
        curr_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        if curr_points is not None:
            curr_points = curr_points.reshape(-1, 2)
        else:
            curr_points = np.array([])
        
        # Flux optique (CORRIGÉ : prev_gray → gray)
        good_prev, good_curr, vectors = self._compute_optical_flow(gray)
        
        # Clustering
        if len(good_curr) > 0 and len(vectors) > 0:
            clusters = self._cluster_points(good_curr, vectors)
            blobs = [DetectedBlob(p, v) for p, v in clusters]
        else:
            blobs = []
        
        # Mise à jour pour la prochaine frame
        self.prev_gray = gray.copy()
        self.prev_points = curr_points if len(curr_points) > 0 else None
        
        return blobs
    
    def _compute_optical_flow(self, curr_gray: np.ndarray):
        """Calcule le flux optique entre prev_gray et curr_gray"""
        if self.prev_points is None or len(self.prev_points) == 0:
            return np.array([]), np.array([]), np.array([])
        
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray,
            self.prev_points.reshape(-1, 1, 2), None, **self.lk_params
        )
        
        if next_pts is None:
            return np.array([]), np.array([]), np.array([])
        
        good_prev = self.prev_points[status.ravel() == 1]
        good_curr = next_pts[status.ravel() == 1]
        
        if len(good_prev) > 0:
            vectors = good_curr - good_prev
        else:
            vectors = np.array([])
        
        return good_prev, good_curr, vectors
    
    def _cluster_points(self, points: np.ndarray, vectors: np.ndarray):
        """Regroupe les points par similarité spatiale et vectorielle"""
        n = len(points)
        if n == 0:
            return []
        
        visited = [False] * n
        clusters = []
        
        max_angle_rad = math.radians(config.max_vector_angle_diff)
        
        for i in range(n):
            if visited[i]:
                continue
            
            cluster_points = [points[i]]
            cluster_vectors = [vectors[i]]
            visited[i] = True
            
            # Expansion BFS
            changed = True
            while changed:
                changed = False
                for j in range(n):
                    if visited[j]:
                        continue
                    
                    for k in range(len(cluster_points)):
                        dist = math.dist(cluster_points[k], points[j])
                        if dist > config.max_vector_distance:
                            continue
                        
                        # Similarité angulaire
                        vk = cluster_vectors[k]
                        vj = vectors[j]
                        norm_k = math.hypot(vk[0], vk[1])
                        norm_j = math.hypot(vj[0], vj[1])
                        
                        if norm_k > 0 and norm_j > 0:
                            dot = vk[0]*vj[0] + vk[1]*vj[1]
                            cos_angle = max(-1.0, min(1.0, dot / (norm_k * norm_j)))
                            angle_diff = math.acos(cos_angle)
                        else:
                            angle_diff = math.pi
                        
                        if angle_diff < max_angle_rad:
                            cluster_points.append(points[j])
                            cluster_vectors.append(vectors[j])
                            visited[j] = True
                            changed = True
                            break
            
            if config.min_cluster_size <= len(cluster_points) <= config.max_cluster_size:
                clusters.append((np.array(cluster_points), np.array(cluster_vectors)))
        
        return clusters


# ============================================================
# CLASSE : SUIVEUR DE BLOBS (inter-frame)
# ============================================================

class BlobTracker:
    """Suivi des blobs dans le temps"""
    
    def __init__(self):
        self.tracked_blobs: List[TrackedBlob] = []
    
    def update(self, detected_blobs: List[DetectedBlob]) -> List[TrackedBlob]:
        """
        Met à jour le suivi avec les blobs détectés
        Retourne la liste des blobs actifs
        """
        # Association
        matches, unmatched_detected, unmatched_tracked = self._match(detected_blobs)
        
        # Mise à jour des matchs
        for tracked_idx, detected_idx in matches:
            self.tracked_blobs[tracked_idx].update(detected_blobs[detected_idx])
        
        # Marquage des absents
        for tracked in unmatched_tracked:
            tracked.mark_missing()
        
        # Ajout des nouveaux
        for detected in unmatched_detected:
            self.tracked_blobs.append(TrackedBlob(detected))
        
        # Nettoyage des inactifs
        self.tracked_blobs = [b for b in self.tracked_blobs if b.is_active()]
        
        return self.tracked_blobs
    
    def _match(self, detected_blobs: List[DetectedBlob]):
        """Association greedy par distance"""
        if not detected_blobs or not self.tracked_blobs:
            return [], list(detected_blobs), list(self.tracked_blobs)
        
        distances = []
        for i, detected in enumerate(detected_blobs):
            for j, tracked in enumerate(self.tracked_blobs):
                dist = math.dist(detected.centroid, tracked.centroid)
                if dist < config.max_match_distance:
                    distances.append((dist, i, j))
        
        distances.sort(key=lambda x: x[0])
        
        matched_detected = set()
        matched_tracked = set()
        matches = []
        
        for dist, detected_idx, tracked_idx in distances:
            if detected_idx not in matched_detected and tracked_idx not in matched_tracked:
                matches.append((tracked_idx, detected_idx))
                matched_detected.add(detected_idx)
                matched_tracked.add(tracked_idx)
        
        unmatched_detected = [b for i, b in enumerate(detected_blobs) if i not in matched_detected]
        unmatched_tracked = [b for j, b in enumerate(self.tracked_blobs) if j not in matched_tracked]
        
        return matches, unmatched_detected, unmatched_tracked


# ============================================================
# CLASSE : OVERLAY VISUEL
# ============================================================

class VisualOverlay:
    """Dessin des éléments sur un calque transparent"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
    
    def clear(self):
        """Vide le calque (devient complètement transparent)"""
        self.layer.fill(0)
    
    def draw_tracked_blob(self, blob: TrackedBlob):
        """Dessine un blob suivi sur le calque"""
        x, y, w, h = blob.bbox
        cx, cy = blob.get_smoothed_position()
        
        # Rectangle (couleur selon attention)
        rect_color = blob.get_rect_color()
        cv2.rectangle(self.layer, (x, y), (x + w, y + h), rect_color, 2)
        
        # Centre (petit point blanc)
        cv2.circle(self.layer, (cx, cy), 3, (255, 255, 255), -1)
        
        # ID et niveau d'attention
        attention_char = "F" if blob.attention_level == "foveal" else ("P" if blob.attention_level == "peripheral" else "-")
        cv2.putText(self.layer, f"ID:{blob.id}{attention_char}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.text_color, 1)
        
        # Vecteur de mouvement (jaune flashy)
        vx, vy = blob.velocity
        speed = math.hypot(vx, vy)
        
        if speed > 0.5:
            length = min(speed * config.vector_scale, 60)
            if length > 1:
                angle = math.atan2(vy, vx)
                end_x = int(cx + length * math.cos(angle))
                end_y = int(cy + length * math.sin(angle))
                cv2.arrowedLine(self.layer, (cx, cy), (end_x, end_y),
                               config.vector_color, 2, tipLength=0.3)
        
        # Prédiction de position (seulement pour attention fovéale ou périphérique)
        if blob.attention_level in ("foveal", "peripheral"):
            pred_x, pred_y = blob.predict_position()
            
            # Cercle cyan pour la position prédite
            cv2.circle(self.layer, (pred_x, pred_y), 8, config.prediction_color, 2)
            cv2.circle(self.layer, (pred_x, pred_y), 3, config.prediction_color, -1)
            
            # Ligne pointillée entre position actuelle et prédite
            cv2.line(self.layer, (cx, cy), (pred_x, pred_y), config.prediction_color, 1, cv2.LINE_AA)
    
    def draw_info(self, tracked_count: int, point_count: int, fps: float, foveal_count: int, peripheral_count: int):
        """Dessine les informations de debug"""
        cv2.putText(self.layer, f"Blobs: {tracked_count} (F:{foveal_count} P:{peripheral_count})", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.text_color, 1)
        cv2.putText(self.layer, f"Points: {point_count}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.text_color, 1)
        cv2.putText(self.layer, f"FPS: {fps:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.text_color, 1)
        
        # Légende attention
        cv2.putText(self.layer, "F:Foveal P:Peripheral -:Passive", (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.text_color, 1)
    
    def get_layer(self) -> np.ndarray:
        return self.layer


# ============================================================
# CLASSE : TRAITEUR (consomme queue_input, produit overlay)
# ============================================================

class FrameProcessor:
    """Thread de traitement : détection + suivi + overlay"""
    
    def __init__(self, input_queue: queue.Queue, output_queue: queue.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.detector = MotionDetector()
        self.tracker = BlobTracker()
        self.overlay = VisualOverlay(config.width, config.height)
        self.thread = threading.Thread(target=self._run, daemon=True)
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def _run(self):
        fps_timer = time.time()
        frame_count = 0
        current_fps = 0.0
        
        while self.running:
            try:
                frame_data = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            frame = frame_data["frame"]
            timestamp = frame_data["timestamp"]
            
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Détection des blobs
            detected_blobs = self.detector.process(gray)
            
            # Suivi inter-frame
            tracked_blobs = self.tracker.update(detected_blobs)
            
            # Comptage par niveau d'attention
            foveal_count = sum(1 for b in tracked_blobs if b.attention_level == "foveal")
            peripheral_count = sum(1 for b in tracked_blobs if b.attention_level == "peripheral")
            
            # Calcul FPS traitement
            frame_count += 1
            if frame_count % 10 == 0:
                current_fps = 10.0 / (time.time() - fps_timer)
                fps_timer = time.time()
            
            # Construction du calque overlay (transparent)
            self.overlay.clear()
            for blob in tracked_blobs:
                self.overlay.draw_tracked_blob(blob)
            
            # Nombre de points optiques
            point_count = len(self.detector.prev_points) if self.detector.prev_points is not None else 0
            self.overlay.draw_info(len(tracked_blobs), point_count, current_fps, foveal_count, peripheral_count)
            
            # Envoi vers la queue de sortie
            self.output_queue.put({
                "frame": frame,
                "overlay": self.overlay.get_layer().copy(),
                "timestamp": timestamp,
                "stats": {
                    "blobs": len(tracked_blobs),
                    "foveal": foveal_count,
                    "peripheral": peripheral_count,
                    "points": point_count,
                    "fps": current_fps
                }
            })
    
    def join(self):
        self.thread.join()


# ============================================================
# CLASSE : CAPTURE (thread webcam)
# ============================================================

class CameraCapture:
    """Thread de capture webcam"""
    
    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue
        self.running = True
        self.cap = None
        self.thread = threading.Thread(target=self._run, daemon=True)
    
    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir la webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
    
    def _run(self):
        frame_interval = 1.0 / config.target_fps
        
        while self.running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, (config.width, config.height))
            
            self.output_queue.put({
                "frame": frame,
                "timestamp": start_time
            })
            
            # Contrôle du framerate
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def join(self):
        self.thread.join()


# ============================================================
# MAIN (thread affichage)
# ============================================================

def main():
    # Queues
    capture_queue = queue.Queue(maxsize=config.queue_maxsize)
    display_queue = queue.Queue(maxsize=config.queue_maxsize)
    
    # Threads
    capture = CameraCapture(capture_queue)
    processor = FrameProcessor(capture_queue, display_queue)
    
    # Démarrage
    print("Démarrage des threads...")
    capture.start()
    processor.start()
    print(f"Capture webcam à {config.target_fps} fps, traitement séparé.")
    print("Appuyez sur ESC pour quitter.")
    
    # Boucle principale (affichage)
    while True:
        try:
            result = display_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        # Empilage frame + overlay
        frame = result["frame"]
        overlay = result["overlay"]
        
        # Fusion (overlay par-dessus la frame)
        output = cv2.addWeighted(frame, 1.0, overlay, 1.0, 0)
        
        # Affichage des stats
        stats = result.get("stats", {})
        if stats:
            cv2.putText(output, f"Proc FPS: {stats.get('fps', 0):.1f}", 
                       (config.width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("ASE - Motion Tracking (Flux Optique Sparse)", output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    # Arrêt
    capture.stop()
    processor.stop()
    capture.join()
    processor.join()
    cv2.destroyAllWindows()
    print("Arrêt terminé.")


if __name__ == "__main__":
    main()