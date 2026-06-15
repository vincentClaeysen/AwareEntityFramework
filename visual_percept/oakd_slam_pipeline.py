import depthai as dai
import numpy as np
import cv2
import sys
import struct
from datetime import timedelta

from percept_geometry import GeometryAnalyzer

try:
    import zenoh
    ZENOH_AVAILABLE = True
except ImportError:
    ZENOH_AVAILABLE = False

# ============================================================================
# CONFIGURATION GLOBALE
# ============================================================================

DEBUG = True            # True = affichages 2D + 3D (PC) | False = pas de GUI (Pi5/robot)
USE_ZENOH = True        # True = bus nerveux réseau | False = mode local pur

# --- AFFICHAGE & ÉCHANTILLONNAGE (DEBUG=True uniquement) ---
USE_MESH = True
STEP = 3
POINT_SIZE = 1.5
DEPTH_ALPHA = 0.8

# Mode d'affichage 3D des couleurs -- cycle via touche 'C' :
#   0 = RGB (couleurs caméra d'origine)
#   1 = orientation (vert=horizontal, bleu=vertical, orange=oblique)
#   2 = régions (couleur pseudo-aléatoire stable par région)
# Dans tous les modes, les structures verticales élancées détectées
# (poteaux/troncs/montants) sont surlignées en jaune électrique.
DISPLAY_MODE = 0
DISPLAY_MODE_COUNT = 3

# --- VALEUR SENTINELLE ---
IMPROBABLE_VALUE = 0.0

# --- GÉOMÉTRIE DU MAILLAGE ---
MAX_DELTA_M = 0.25      # Écart max de profondeur (m) entre voisins pour tracer une arête

# --- PARAMÈTRES CAMÉRA ---
FPS = 20
MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 8.00
WIDTH = 640
HEIGHT = 360            # Ratio 16:9, cohérent avec isp scale 1/3 du capteur 1080p

RGB_JPEG_QUALITY = 80
SYNC_THRESHOLD_MS = 50

# ============================================================================
# PARAMÈTRES DE STABILITÉ SLAM -- PERCEPTION CONTINUE (ROBOT MOBILE)
# ============================================================================
#
# Philosophie : le robot scanne en mouvement constant. Tout filtre qui
# introduit du lag temporel (temporal filter firmware) crée du smearing
# sur les arêtes en déplacement -> faux positifs de mouvement pour
# motion_blobs.py et incohérence géométrique pour le SLAM. On privilégie
# donc des filtres spatiaux (par frame) et un rejet matériel agressif des
# mesures peu fiables, plutôt qu'un lissage temporel.

STEREO_MEDIAN_FILTER = dai.MedianFilter.KERNEL_5x5
# 5x5 : compromis bruit/densité avec subpixel actif. 7x7 sur-lisse les
# arêtes et réduit la densité de points valides en bordure d'objets.

STEREO_CONFIDENCE_THRESHOLD = 200
# Plus bas = plus strict (moins de points, mais plus fiables). 200 est
# un bon compromis sans temporal filter pour compenser l'absence de
# lissage temporel.

STEREO_LR_CHECK_THRESHOLD = 5
# Seuil de différence de disparité pour le left-right check (défaut: 5).
# Diminuer (ex: 3) rejette plus de points aux occlusions/bords -- utile
# si motion_blobs.py est sensible aux faux contours sur les arêtes
# d'objets en mouvement relatif (parallaxe du robot).

THRESHOLD_FILTER_MIN_MM = int(MIN_DEPTH_M * 1000)
THRESHOLD_FILTER_MAX_MM = int(MAX_DEPTH_M * 1000)
# Filtre matériel : rejette les mesures hors plage directement sur le
# device, avant transmission -- réduit le volume de données ET évite de
# traiter des valeurs aberrantes côté host.

DECIMATION_FACTOR = 1
# 1 = pas de décimation supplémentaire (on est déjà en 400P natif).
# Passer à 2 réduirait encore le bruit par moyennage si le Pi5 sature,
# au prix de la moitié de la résolution depth.

SPATIAL_FILTER_HOLE_FILLING_RADIUS = 0
# 0 = pas d'extrapolation arbitraire. Les trous restent des trous --
# critique pour ne pas halluciner de géométrie dans le WorldModel.

SPECKLE_RANGE = 50
# Taille max (en pixels de disparité) des îlots isolés supprimés.

# ============================================================================
# FONCTIONS UTILITAIRES AFFICHAGE 2D (DEBUG=True uniquement)
# ============================================================================

if DEBUG:
    turbo = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(256, 1),
        cv2.COLORMAP_TURBO
    )
    TURBO_INV = turbo[::-1].copy()


def depth_to_display(depth_m_stabilized):
    valid = (depth_m_stabilized > IMPROBABLE_VALUE)
    compressed = np.zeros_like(depth_m_stabilized)
    compressed[valid] = np.log(depth_m_stabilized[valid] / MIN_DEPTH_M) / np.log(MAX_DEPTH_M / MIN_DEPTH_M)
    compressed = np.clip(compressed, 0.0, 1.0)
    idx = (compressed * 255).astype(np.uint8)
    colored = TURBO_INV[idx, 0]
    colored[~valid] = (0, 0, 0)
    return colored


def add_depth_scale(img):
    h, w, _ = img.shape
    scale_width = 120
    canvas = np.full((h, w + scale_width, 3), 255, dtype=np.uint8)
    canvas[:, :w] = img
    x0 = w + 20
    y0 = 20
    bar_h = h - 40
    bar_w = 25
    for y in range(bar_h):
        ratio = y / (bar_h - 1)
        idx = int(ratio * 255)
        canvas[y0 + y, x0:x0 + bar_w] = TURBO_INV[idx, 0].tolist()
    ticks = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    for depth in ticks:
        pos = np.log(depth / MIN_DEPTH_M) / np.log(MAX_DEPTH_M / MIN_DEPTH_M)
        y = int(y0 + pos * bar_h)
        idx = int(pos * 255)
        color = tuple(int(v) for v in TURBO_INV[idx, 0])
        cv2.line(canvas, (x0 + bar_w, y), (x0 + bar_w + 8, y), (0, 0, 0), 1)
        cv2.putText(canvas, f"{depth:.1f}m", (x0 + bar_w + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


# ============================================================================
# CONFIGURATION DU PIPELINE OAK-D LITE
# ============================================================================

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
sync = pipeline.create(dai.node.Sync)
xout_sync = pipeline.create(dai.node.XLinkOut)
xin_control = pipeline.create(dai.node.XLinkIn)

xout_sync.setStreamName("sync")
xin_control.setStreamName("cam_control")

cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setIspScale(1, 3)  # 1080p / 3 -> 640x360, sans crop
cam_rgb.setFps(FPS)

left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setFps(FPS)
right.setFps(FPS)

stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.initialConfig.setMedianFilter(STEREO_MEDIAN_FILTER)
stereo.initialConfig.setConfidenceThreshold(STEREO_CONFIDENCE_THRESHOLD)
stereo.initialConfig.setLeftRightCheckThreshold(STEREO_LR_CHECK_THRESHOLD)

stereo_config = stereo.initialConfig.get()

# Temporal filter désactivé : voir bloc "PARAMÈTRES DE STABILITÉ SLAM" ci-dessus.
stereo_config.postProcessing.temporalFilter.enable = False

stereo_config.postProcessing.spatialFilter.enable = True
stereo_config.postProcessing.spatialFilter.holeFillingRadius = SPATIAL_FILTER_HOLE_FILLING_RADIUS

stereo_config.postProcessing.speckleFilter.enable = True
stereo_config.postProcessing.speckleFilter.speckleRange = SPECKLE_RANGE

stereo_config.postProcessing.thresholdFilter.minRange = THRESHOLD_FILTER_MIN_MM
stereo_config.postProcessing.thresholdFilter.maxRange = THRESHOLD_FILTER_MAX_MM

stereo_config.postProcessing.decimationFilter.decimationFactor = DECIMATION_FACTOR

stereo.initialConfig.set(stereo_config)

left.out.link(stereo.left)
right.out.link(stereo.right)

sync.setSyncThreshold(timedelta(milliseconds=SYNC_THRESHOLD_MS))
cam_rgb.isp.link(sync.inputs["rgb"])
stereo.depth.link(sync.inputs["depth"])
sync.out.link(xout_sync.input)

xin_control.out.link(cam_rgb.inputControl)


# ============================================================================
# HOOK D'EXPORT POUR LE SLAM / WORLDMODEL
# ============================================================================

def export_to_worldmodel(depth_m_stabilized, rgb_frame, timestamp_ns, fx, fy, cx, cy):
    if not (USE_ZENOH and ZENOH_AVAILABLE):
        return

    h, w = depth_m_stabilized.shape
    timestamp_int = int(timestamp_ns)

    depth_mm = np.clip(depth_m_stabilized * 1000.0, 0, 65535).astype(np.uint16)
    depth_header = struct.pack('<Qii4f', timestamp_int, h, w, fx, fy, cx, cy)
    pub_depth.put(depth_header + depth_mm.tobytes())

    ok, jpg = cv2.imencode('.jpg', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, RGB_JPEG_QUALITY])
    if ok:
        rgb_header = struct.pack('<Q', timestamp_int)
        pub_rgb.put(rgb_header + jpg.tobytes())


# ============================================================================
# PIPELINE DE RAFFINEMENT DES DONNÉES
# ============================================================================

def process_and_export(msg_group):
    rgb_msg = msg_group["rgb"]
    depth_msg = msg_group["depth"]

    rgb_frame = rgb_msg.getCvFrame()
    depth = depth_msg.getFrame()

    timestamp_ns = depth_msg.getTimestamp().total_seconds() * 1e9

    if depth.shape[1] != WIDTH or depth.shape[0] != HEIGHT:
        depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    depth_m = depth.astype(np.float32) / 1000.0
    valid_mask = ((depth_m > MIN_DEPTH_M) & (depth_m < MAX_DEPTH_M))

    kernel = np.ones((2, 2), dtype=np.uint8)
    valid_mask = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)

    depth_m_stabilized = np.where(valid_mask, depth_m, IMPROBABLE_VALUE)

    export_to_worldmodel(depth_m_stabilized, rgb_frame, timestamp_ns, fx, fy, cx, cy)

    return depth_m_stabilized, rgb_frame


# ============================================================================
# RECONSTRUCTION 3D -- CALCUL DES GÉOMÉTRIES (indépendant du backend de rendu)
# ============================================================================

def upsample_mask_to_full(mask_sampled, height, width, step):
    """
    Remonte un masque bool (h_sampled, w_sampled) -- issu de la grille
    échantillonnée (après STEP) -- à la résolution caméra complète
    (height, width), en étalant chaque point sur le bloc STEP x STEP
    correspondant (dilatation), pour un overlay visuellement continu.
    """
    full = np.zeros((height, width), dtype=bool)
    eh, ew = mask_sampled.shape
    full[:eh * step:step, :ew * step:step] = mask_sampled
    full = cv2.dilate(full.astype(np.uint8), np.ones((step, step), np.uint8)).astype(bool)
    return full


def build_pointcloud(depth_m_stabilized, rgb_frame, fx, fy, cx, cy):
    """Retourne points (N,3) et couleurs (N,3) en [0,1] pour les pixels valides."""
    z = depth_m_stabilized[::STEP, ::STEP]
    h, w = z.shape
    ys, xs = np.mgrid[0:HEIGHT:STEP, 0:WIDTH:STEP]
    valid = (z > IMPROBABLE_VALUE)

    x_3d = (xs - cx) * z / fx
    y_3d = (ys - cy) * z / fy
    coords = np.stack([x_3d, z, -y_3d], axis=-1)

    rgb_ds = rgb_frame[::STEP, ::STEP]
    colors = np.zeros((h, w, 3), dtype=np.float64)
    colors[..., 0] = rgb_ds[..., 2] / 255.0
    colors[..., 1] = rgb_ds[..., 1] / 255.0
    colors[..., 2] = rgb_ds[..., 0] / 255.0

    return coords, colors, valid


def build_mesh_edges(coords, colors, valid):
    """
    Retourne points (flat, N,3), edges (M,2) indices, edge_colors (M,3).
    4 familles d'arêtes : horizontale, verticale, diagonale, anti-diagonale.
    """
    h, w = valid.shape
    idx_grid = np.arange(h * w).reshape(h, w)
    z = coords[..., 1]  # composante "profondeur" (axe Y dans coords)

    edges_list, colors_list = [], []

    mask_h = valid[:, :-1] & valid[:, 1:] & (np.abs(z[:, :-1] - z[:, 1:]) < MAX_DELTA_M)
    if np.any(mask_h):
        edges_list.append(np.stack([idx_grid[:, :-1][mask_h], idx_grid[:, 1:][mask_h]], axis=1))
        colors_list.append(colors[:, :-1][mask_h])

    mask_v = valid[:-1, :] & valid[1:, :] & (np.abs(z[:-1, :] - z[1:, :]) < MAX_DELTA_M)
    if np.any(mask_v):
        edges_list.append(np.stack([idx_grid[:-1, :][mask_v], idx_grid[1:, :][mask_v]], axis=1))
        colors_list.append(colors[:-1, :][mask_v])

    mask_d = valid[:-1, :-1] & valid[1:, 1:] & (np.abs(z[:-1, :-1] - z[1:, 1:]) < MAX_DELTA_M)
    if np.any(mask_d):
        edges_list.append(np.stack([idx_grid[:-1, :-1][mask_d], idx_grid[1:, 1:][mask_d]], axis=1))
        colors_list.append(colors[:-1, :-1][mask_d])

    mask_ad = valid[1:, :-1] & valid[:-1, 1:] & (np.abs(z[1:, :-1] - z[:-1, 1:]) < MAX_DELTA_M)
    if np.any(mask_ad):
        edges_list.append(np.stack([idx_grid[1:, :-1][mask_ad], idx_grid[:-1, 1:][mask_ad]], axis=1))
        colors_list.append(colors[1:, :-1][mask_ad])

    points_flat = coords.reshape(-1, 3)

    if edges_list:
        edges = np.concatenate(edges_list, axis=0)
        edge_colors = np.concatenate(colors_list, axis=0)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        edge_colors = np.zeros((0, 3), dtype=np.float64)

    return points_flat, edges, edge_colors


# ============================================================================
# INITIALISATION SESSIONS & MATÉRIEL
# ============================================================================

zenoh_session = None
pub_rgb = None
pub_depth = None

if USE_ZENOH:
    if ZENOH_AVAILABLE:
        print("Initialisation du bus Zenoh...")
        zenoh_config = zenoh.Config()
        zenoh_session = zenoh.open(zenoh_config)
        pub_rgb = zenoh_session.declare_publisher("perception/rgb")
        pub_depth = zenoh_session.declare_publisher("perception/depth")
    else:
        print("ATTENTION : USE_ZENOH actif mais 'zenoh' non installé. Mode LOCAL.")

device = dai.Device(pipeline)
q_sync = device.getOutputQueue(name="sync", maxSize=4, blocking=False)
q_control = device.getInputQueue(name="cam_control")

control_msg = dai.CameraControl()
control_msg.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
q_control.send(control_msg)

calib_data = device.readCalibration()
intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, WIDTH, HEIGHT)
fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]

geometry_analyzer = GeometryAnalyzer()


def shutdown():
    device.close()
    if zenoh_session:
        zenoh_session.close()


# ============================================================================
# MODE HEADLESS (Pi5 / robot, pas de GUI)
# ============================================================================

if not DEBUG:
    print("Mode Headless actif. Lecture bloquante synchrone via matériel...")
    try:
        while True:
            msg_group = q_sync.get()
            process_and_export(msg_group)
    except KeyboardInterrupt:
        print("\nArrêt propre demandé. Clôture des flux.")
        shutdown()
        sys.exit()

# ============================================================================
# MODE DEBUG (PC) -- 2D via OpenCV, 3D via Open3D
# ============================================================================

import open3d as o3d

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="SAS - SLAM Pointcloud Debug", width=1280, height=720)

# Repère (frustum/origine caméra) pour orientation visuelle
axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
vis.add_geometry(axis_frame)

pcd = o3d.geometry.PointCloud()
lineset = o3d.geometry.LineSet()
vis.add_geometry(pcd)
vis.add_geometry(lineset)

render_opt = vis.get_render_option()
render_opt.point_size = POINT_SIZE
render_opt.background_color = np.array([0.05, 0.05, 0.05])

# Caméra : on regarde vers les Z+ (axe "profondeur" = Y dans coords_3d, mais
# Open3D définit sa propre vue caméra par défaut ; on règle un point de vue
# raisonnable une fois la première frame reçue, via reset_view_point.
view_initialized = False

# CHANGEMENT : callbacks clavier via GLFW (Open3D), indépendants du focus
# de la fenêtre OpenCV. cv2.waitKey() devient muet dès que la fenêtre
# Open3D a le focus -- ces callbacks restent actifs tant que la fenêtre
# Open3D a le focus, ce qui est le cas justement quand on manipule la vue.
_state = {"running": True}

GLFW_KEY_UP = 265
GLFW_KEY_DOWN = 264
GLFW_KEY_ESCAPE = 256


def _on_key_up(vis_):
    global DEPTH_ALPHA
    DEPTH_ALPHA = min(1.0, DEPTH_ALPHA + 0.05)
    return False


def _on_key_down(vis_):
    global DEPTH_ALPHA
    DEPTH_ALPHA = max(0.0, DEPTH_ALPHA - 0.05)
    return False


def _on_key_escape(vis_):
    _state["running"] = False
    return False


def _on_key_cycle_display(vis_):
    global DISPLAY_MODE
    DISPLAY_MODE = (DISPLAY_MODE + 1) % DISPLAY_MODE_COUNT
    mode_names = {0: "RGB", 1: "orientation", 2: "régions"}
    print(f"Mode d'affichage 3D : {mode_names[DISPLAY_MODE]}")
    return False


vis.register_key_callback(GLFW_KEY_UP, _on_key_up)
vis.register_key_callback(GLFW_KEY_DOWN, _on_key_down)
vis.register_key_callback(GLFW_KEY_ESCAPE, _on_key_escape)

GLFW_KEY_C = 67
vis.register_key_callback(GLFW_KEY_C, _on_key_cycle_display)

print("Mode DEBUG actif. ESC ou fermeture de la fenêtre 3D pour quitter.")

try:
    while _state["running"]:
        msg_group = q_sync.tryGet()
        if msg_group is not None:
            depth_m_stabilized, rgb_frame = process_and_export(msg_group)
            depth_colored = depth_to_display(depth_m_stabilized)

            coords, colors, valid = build_pointcloud(depth_m_stabilized, rgb_frame, fx, fy, cx, cy)

            # --- Analyse géométrique du percept (percept_geometry) ---
            normals, normal_valid = geometry_analyzer.compute_normals(coords, valid)
            orientation = geometry_analyzer.classify_orientation(normals, normal_valid)
            region_map, regions, bridge_candidates = geometry_analyzer.segment_regions(
                coords, normals, orientation, normal_valid
            )
            regions = geometry_analyzer.merge_coplanar_via_bridge(regions, bridge_candidates)
            frame_contents = geometry_analyzer.detect_frames(regions, region_map, depth_m_stabilized, STEP)
            elongated = geometry_analyzer.detect_elongated_vertical(normals, orientation, region_map, regions)

            # Sélection de la base de couleurs selon DISPLAY_MODE
            if DISPLAY_MODE == 1:
                display_colors = geometry_analyzer.orientation_to_colors(orientation)
            elif DISPLAY_MODE == 2:
                display_colors = geometry_analyzer.region_to_colors(region_map, regions)
            else:
                display_colors = colors

            # Overlay jaune électrique : structures verticales élancées
            # (poteaux/troncs/montants), visible dans tous les modes 3D.
            display_colors = geometry_analyzer.elongated_to_colors(elongated, display_colors)

            # Overlay jaune électrique : zones intérieures de cadres détectés
            # (encadrements de porte/fenêtre). Même teinte que les structures
            # élancées, intensité modulée selon l'état du contenu (COPLANAR
            # = jaune plein, RECESSED/UNKNOWN = jaune atténué).
            frame_overlay = geometry_analyzer.frame_content_to_overlay_mask(frame_contents, region_map)
            display_colors = geometry_analyzer.frame_overlay_to_colors(frame_overlay, display_colors)

            # --- Affichage 2D (OpenCV) ---
            overlay_view = cv2.addWeighted(rgb_frame, 1.0 - DEPTH_ALPHA, depth_colored, DEPTH_ALPHA, 0)
            cv2.putText(overlay_view, f"Depth Opacity: {DEPTH_ALPHA*100:.0f}%", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Overlay jaune transparent sur la vue RGB pour les structures
            # élancées détectées (poteaux/troncs/montants). `elongated` est
            # à la résolution de la grille échantillonnée (après STEP) ;
            # on remonte à la résolution caméra pour l'overlay.
            elongated_full = upsample_mask_to_full(elongated, HEIGHT, WIDTH, STEP)

            yellow_layer = np.zeros_like(rgb_frame)
            yellow_layer[elongated_full] = (0, 255, 255)  # BGR -> jaune électrique plein

            # Overlay jaune transparent sur la vue RGB pour les zones
            # intérieures de cadres détectés (encadrements de porte/fenêtre).
            # Même teinte, intensité modulée selon l'état du contenu
            # (COPLANAR = jaune plein, RECESSED/UNKNOWN = jaune atténué).
            frame_coplanar_full = upsample_mask_to_full(
                frame_overlay == 0, HEIGHT, WIDTH, STEP  # FRAME_CONTENT_COPLANAR
            )
            frame_other_full = upsample_mask_to_full(
                (frame_overlay == 1) | (frame_overlay == 2), HEIGHT, WIDTH, STEP  # RECESSED/UNKNOWN
            )
            yellow_layer[frame_coplanar_full] = (0, 255, 255)   # jaune électrique plein
            yellow_layer[frame_other_full] = (0, 150, 150)      # jaune atténué (BGR)

            rgb_frame_overlay = cv2.addWeighted(rgb_frame, 1.0, yellow_layer, 0.4, 0)

            cv2.imshow("Depth Map", add_depth_scale(depth_colored))
            cv2.imshow("RGB Aligned View", rgb_frame_overlay)
            cv2.imshow("RGB-D Overlay", overlay_view)
            cv2.waitKey(1)  # nécessaire pour rafraîchir les fenêtres OpenCV ;
                             # la gestion des touches se fait via Open3D ci-dessous.

            # CHANGEMENT : remove_geometry + add_geometry au lieu de
            # update_geometry. Le nombre de points/arêtes varie à chaque
            # frame (densité de validité changeante) -- update_geometry
            # peut figer ou lever une erreur interne sur un changement de
            # taille de buffer selon la version d'Open3D. reset_bounding_box
            # =False évite de recadrer/recentrer la caméra à chaque frame.
            vis.remove_geometry(pcd, reset_bounding_box=False)
            vis.remove_geometry(lineset, reset_bounding_box=False)

            if USE_MESH:
                points_flat, edges, edge_colors = build_mesh_edges(coords, display_colors, valid)
                lineset.points = o3d.utility.Vector3dVector(points_flat)
                lineset.lines = o3d.utility.Vector2iVector(edges)
                lineset.colors = o3d.utility.Vector3dVector(edge_colors)
                pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            else:
                pcd.points = o3d.utility.Vector3dVector(coords[valid])
                pcd.colors = o3d.utility.Vector3dVector(display_colors[valid])
                lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int64))

            vis.add_geometry(pcd, reset_bounding_box=False)
            vis.add_geometry(lineset, reset_bounding_box=False)

            if not view_initialized and np.any(valid):
                vis.reset_view_point(True)
                view_initialized = True

            # Inspection console des cadres détectés (étape 2e) -- pas
            # d'overlay visuel dédié encore, juste le suivi des états
            # géométriques bruts (COPLANAR/RECESSED/UNKNOWN).
            if frame_contents:
                for fc in frame_contents:
                    state_names = {0: "COPLANAR", 1: "RECESSED", 2: "UNKNOWN"}
                    print(f"Cadre détecté : contenu = {state_names.get(fc.frame_content_state, '?')} "
                          f"bbox_rows={fc.bbox_rows} bbox_cols={fc.bbox_cols}")

        if not vis.poll_events():
            break
        vis.update_renderer()

except KeyboardInterrupt:
    pass
finally:
    vis.destroy_window()
    cv2.destroyAllWindows()
    shutdown()
