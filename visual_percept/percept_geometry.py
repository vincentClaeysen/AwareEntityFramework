"""
percept_geometry.py

Module d'analyse géométrique du percept visuel pour ASE.

Approche itérative :
  Étape 1 (v0.1) : normales locales par point + classification d'orientation
      (horizontal / vertical / oblique).
  Étape 2 (v0.4) : régions connexes par similarité de normale
      (orientation + azimut), caractérisation de l'étendue (grande surface
      vs petite surface posée) et de l'élongation.
      - 2c/2d : fusion par pont d'occlusion -- des régions coplanaires non
        connexes, reliées (directement ou en chaîne, transitivité via
        union-find) par une ou plusieurs régions d'orientation différente
        adjacentes à au moins deux d'entre elles, sont fusionnées en une
        seule région virtuelle (is_inferred=True). Les petits objets sous
        min_region_area mais >= min_bridge_area (crochets, poignées,
        interrupteurs) peuvent servir de pont sans être eux-mêmes des
        régions normales (cf. bridge_candidates).
      - 2e : détection de cadres (4 segments de même plan encadrant une zone
        creuse) et état géométrique brut du contenu (FRAME_CONTENT_COPLANAR
        / RECESSED / UNKNOWN) -- sans classification porte/fenêtre/état.
  Étape 3 (v0.3, ce module) : structures verticales élancées (poteaux/troncs)
      via détection de changement de signe de la normale horizontale (nx) sur
      des colonnes étroites (cylindrique), ET régions verticales à fort ratio
      hauteur/largeur (polygonal) -- remplissage par masque exact via
      region_map (pas par bounding box).
  Étape 4 (à venir) : confiance temporelle par région (hypothèses
      confirmées/infirmées dans le temps, avec compensation IMU).

Ce module est volontairement sans dépendance sur le reste du pipeline
(pas d'import depthai/zenoh) -- il prend des tableaux numpy en entrée et
retourne des tableaux numpy en sortie, pour rester testable isolément et
réutilisable côté Jetson (NaiveWorldModel / eRetina.py).
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


# ============================================================================
# VALEUR SENTINELLE (cohérente avec oakd_slam_pipeline.py)
# ============================================================================

IMPROBABLE_VALUE = 0.0  # depth_m_stabilized == IMPROBABLE_VALUE -> point invalide


# ============================================================================
# CONSTANTES DE CLASSIFICATION D'ORIENTATION
# ============================================================================

# Seuil sur la composante verticale de la normale (|normal.y| dans le repère
# coords_3d = (x, profondeur, -hauteur), donc l'axe "vertical" du monde réel
# est l'axe Z de coords_3d, i.e. la 3e composante).
#
# normal proche de (0,0,±1) -> surface HORIZONTALE (sol/plafond/assise...)
# normal proche du plan XY  -> surface VERTICALE (mur/face d'objet...)
# entre les deux            -> OBLIQUE (rampe, toit en pente, transition...)
ORIENTATION_HORIZONTAL_THRESHOLD = 0.85  # |nz| > ce seuil -> horizontal
ORIENTATION_VERTICAL_THRESHOLD = 0.35    # |nz| < ce seuil -> vertical
# entre les deux seuils -> oblique

# Codes d'orientation (utilisés pour la colorisation et en sortie de classify)
ORIENTATION_HORIZONTAL = 0
ORIENTATION_VERTICAL = 1
ORIENTATION_OBLIQUE = 2
ORIENTATION_UNKNOWN = -1  # normale non calculable (bord de grille, voisinage invalide)


# États géométriques bruts du contenu d'un cadre détecté (étape 2e).
# Aucune sémantique porte/fenêtre/ouvert/fermé -- juste l'observation.
FRAME_CONTENT_COPLANAR = 0     # le contenu est dans le même plan que le cadre (panneau)
FRAME_CONTENT_RECESSED = 1     # le contenu a une profondeur différente (ouverture/passage)
FRAME_CONTENT_UNKNOWN = 2      # pas de données fiables (hors plage, bruit)


# ============================================================================
# CONSTANTES DE DÉTECTION DES STRUCTURES VERTICALES ÉLANCÉES
# ============================================================================
#
# Deux signatures recherchées :
#
# (a) Cylindrique : sur une colonne de pixels (même x, y variable), la
# composante horizontale de la normale (nx) change de signe/direction de
# façon répétée sur une fenêtre étroite -- typique d'une surface cylindrique
# verticale (poteau, tronc, pied de table) vue de face. Un mur plan a une
# normale horizontale stable (pas de changement de signe).
#
# (b) Polygonale : une région VERTICAL dont la bbox 3D a un ratio
# hauteur/largeur élevé (ex: montant de porte, pied de meuble à section
# carrée) -- pas de changement de signe de nx, mais une forme clairement
# élancée. Remplissage par masque exact (region_map), pas par bbox.
#
# On ne calcule la "courbure horizontale" que sur les points VERTICAUX
# (cf. classify_orientation) -- un poteau est par nature une surface
# verticale, donc cette analyse est un raffinement de la catégorie VERTICAL,
# pas une 4e catégorie d'orientation.

# Largeur de la fenêtre glissante (en colonnes) pour détecter un changement
# de signe de nx -- doit être de l'ordre de la largeur apparente d'un poteau
# fin à distance moyenne. Valeur en pixels de la grille échantillonnée
# (après STEP), pas en pixels caméra natifs.
COLUMN_CURVATURE_WINDOW = 3

# Hauteur minimale (en points de la grille échantillonnée) d'une colonne
# candidate pour être retenue comme structure élancée -- élimine les
# faux positifs isolés (1-2 points de bruit).
MIN_ELONGATED_HEIGHT = 4

# Seuil sur |nx| en deçà duquel la normale est considérée comme "tournant"
# (proche de l'axe de vue, donc surface courbe vue de face) -- combiné au
# changement de signe pour confirmer la courbure.
CURVATURE_NX_THRESHOLD = 0.3

# Seuil d'élongation pour les structures polygonales (non cylindriques).
# Ratio hauteur (dz) / largeur (max(dx,dy)) de la bbox 3D au-delà duquel une
# région VERTICAL est considérée comme "élancée" même sans changement de
# signe de nx.
POLYGONAL_ELONGATION_THRESHOLD = 3.0


# ============================================================================
# CONSTANTES DE RÉGIONS & ÉTENDUE
# ============================================================================
#
# Étape 2 : regroupement des points en régions connexes par similarité de
# normale, puis caractérisation de chaque région (aire, bounding box,
# élongation, niveau). Le regroupement se fait en deux temps :
#   1. Discrétisation directionnelle : la normale (unitaire) est projetée
#      sur un petit nombre de directions de référence -- deux points
#      appartiennent au même "bucket directionnel" si leurs normales sont
#      proches. Ça évite de fusionner deux murs perpendiculaires qui
#      seraient tous deux ORIENTATION_VERTICAL.
#   2. Connexité spatiale : scipy.ndimage.label sur chaque bucket pour
#      obtenir des régions connexes (4-connexité).
#
# Une région est ensuite caractérisée par :
#   - aire (nombre de points)
#   - bounding box dans la grille (H,W) -> largeur/hauteur en cellules
#   - étendue spatiale réelle (m) via les coordonnées 3D min/max
#   - élongation (ratio dimension max / dimension min de la bbox 3D)
#   - niveau moyen (hauteur moyenne, -coords[...,2])
#   - orientation dominante (ORIENTATION_*)

# Seuil de similarité directionnelle (produit scalaire entre deux normales)
# pour appartenir au même bucket. 1.0 = identique, plus bas = plus permissif.
# (Réservé pour un raffinement futur du bucketing -- non utilisé directement
# dans la discrétisation par azimut actuelle.)
NORMAL_SIMILARITY_THRESHOLD = 0.92

# Nombre de buckets directionnels par catégorie d'orientation. Les normales
# VERTICAL et OBLIQUE sont sous-catégorisées par leur azimut (direction dans
# le plan horizontal) pour séparer des murs perpendiculaires ; HORIZONTAL
# n'a généralement pas besoin de sous-catégorisation (sol/plafond ont peu
# de variantes), mais on garde le même mécanisme pour uniformité.
AZIMUTH_BUCKETS = 8  # tous les 45°

# Aire minimale (en points de la grille échantillonnée) pour qu'une région
# soit retenue -- élimine le bruit résiduel (régions de 1-3 points).
MIN_REGION_AREA = 6

# Aire minimale pour qu'une région puisse servir de PONT d'occlusion
# (cf. merge_coplanar_via_bridge / detect_frames), même si elle est trop
# petite pour être retenue comme région "normale" dans `regions`. Permet
# aux petits objets (crochets, interrupteurs, poignées...) de jouer leur
# rôle de pont entre deux fragments de mur/table sans pour autant être
# caractérisés comme une "surface" ou un "objet" à part entière.
# Doit être <= MIN_REGION_AREA.
MIN_BRIDGE_AREA = 2

# Seuil d'élongation (ratio dimension_max / dimension_min de la bbox 3D)
# au-delà duquel une région est considérée "élancée" plutôt que "surface".
ELONGATION_RATIO_THRESHOLD = 2.5

# Seuil d'aire (en m², approximatif via bbox 3D largeur x profondeur ou
# largeur x hauteur selon orientation) séparant "grande surface" (sol, mur,
# plafond) de "petite surface posée" (assise, étagère...).
LARGE_SURFACE_AREA_M2 = 0.6


# ============================================================================
# CONSTANTES DE FUSION PAR PONT D'OCCLUSION & DÉTECTION DE CADRE
# ============================================================================
#
# Étape 2c/2d : deux régions A, B de même plan approximatif (même
# orientation + azimut bucket, niveau/position cohérente) sont fusionnées en
# un plan continu hypothétique SI une région C (orientation différente,
# typiquement perpendiculaire) est adjacente à la fois à A et à B -- C agit
# alors comme "pont d'occlusion" attestant que A et B sont deux fragments
# visibles d'une même surface partiellement masquée par C.
#
# Étape 2e : un "cadre" est un cas particulier où 4 segments de mur (haut,
# bas, gauche, droite, même plan) entourent une zone intérieure -- on
# caractérise alors l'état géométrique du contenu de cette zone :
#   - coplanaire avec le cadre  -> panneau fermé (porte/fenêtre/volet fermés)
#   - profondeur différente     -> ouverture (on voit "à travers" ou plus loin)
#   - pas de données            -> indéterminé (hors plage stéréo, reflet...)
# Aucune classification porte/fenêtre/ouvert/fermé n'est faite ici -- ce sont
# des états géométriques bruts, à interpréter plus haut dans le pipeline.

# Distance max (en cellules de la grille échantillonnée) entre les bbox de
# deux régions pour les considérer "adjacentes" (pont ou cadre).
ADJACENCY_MAX_GAP = 1

# Tolérance sur le niveau moyen (m) entre deux régions horizontales pour
# les considérer comme appartenant au même plan (même hauteur de table).
PLANE_LEVEL_TOLERANCE_M = 0.05

# Tolérance sur la position du plan (m) le long de l'axe normal, pour deux
# régions verticales/obliques de même azimut (même mur, à des hauteurs
# différentes par ex.). Mesurée comme écart de la projection du centroïde
# sur la normale moyenne de la région.
PLANE_OFFSET_TOLERANCE_M = 0.08

# Tolérance de profondeur (m) pour juger qu'une zone intérieure de cadre est
# "coplanaire" avec le cadre lui-même (panneau fermé) plutôt que "derrière"
# (ouverture).
FRAME_CONTENT_COPLANAR_TOLERANCE_M = 0.10


# ============================================================================
# STRUCTURE DE DONNÉES : RÉGION
# ============================================================================

@dataclass
class Region:
    """
    Représente une région connexe de points partageant une orientation/
    normale similaire. Purement géométrique -- aucune étiquette sémantique.

    Attributs géométriques de base :
        label              : identifiant entier de la région (assigné lors
                             de segment_regions)
        orientation        : code ORIENTATION_* dominant de la région
        azimuth_bucket     : bucket directionnel (0..AZIMUTH_BUCKETS-1, ou -1
                             pour ORIENTATION_HORIZONTAL où l'azimut n'est
                             pas discriminant)
        area_px            : nombre de points (grille échantillonnée) dans la région
        bbox_rows          : (row_min, row_max) dans la grille
        bbox_cols          : (col_min, col_max) dans la grille
        extent_3d          : (dx, dy, dz) -- étendue en mètres sur chaque axe
                             de coords (x, profondeur, -hauteur)
        elongation_ratio   : dimension_max / dimension_min de extent_3d
                             (>1, inf si dimension_min == 0)
        mean_level         : hauteur moyenne de la région (= -mean(coords[...,2]))
        centroid_3d        : (x, y, z) centroïde de la région dans coords
        mean_normal        : (nx, ny, nz) normale moyenne de la région
        is_large_surface   : True si l'étendue dépasse LARGE_SURFACE_AREA_M2
        is_elongated       : True si elongation_ratio > ELONGATION_RATIO_THRESHOLD

    Attributs de fusion / cadre / structures élancées (remplis par
    merge_coplanar_via_bridge / detect_frames / detect_elongated_vertical,
    valeurs par défaut sinon) :
        merged_into        : label de la région virtuelle fusionnée
                             contenant cette région, ou son propre label
                             si non fusionnée.
        is_inferred        : True si cette région est une région VIRTUELLE
                             issue d'une fusion ou d'un cadre (n'existe pas
                             directement dans region_map).
        bridge_region_label: label de la région C ayant servi de pont
                             (uniquement pour les régions virtuelles fusionnées).
        is_frame           : True si cette région fait partie d'un cadre détecté.
        frame_content_state: un de FRAME_CONTENT_* si cette région est la
                             zone intérieure d'un cadre, sinon None.
        is_elongated_struct: True si cette région a été identifiée comme
                             structure verticale élancée (poteau/tronc/
                             montant), par détection cylindrique ou
                             polygonale (cf. detect_elongated_vertical).
    """
    label: int
    orientation: int
    azimuth_bucket: int
    area_px: int
    bbox_rows: Tuple[int, int]
    bbox_cols: Tuple[int, int]
    extent_3d: Tuple[float, float, float]
    elongation_ratio: float
    mean_level: float
    centroid_3d: Tuple[float, float, float]
    mean_normal: Tuple[float, float, float]
    is_large_surface: bool
    is_elongated: bool

    # Champs de fusion / cadre / structures élancées -- valeurs par défaut
    merged_into: int = field(default_factory=lambda: 0)
    is_inferred: bool = False
    bridge_region_label: Optional[int] = None
    is_frame: bool = False
    frame_content_state: Optional[int] = None
    is_elongated_struct: bool = False

    def __post_init__(self):
        if self.merged_into == 0:
            self.merged_into = self.label

    def __repr__(self) -> str:
        orient_names = {ORIENTATION_HORIZONTAL: "H", ORIENTATION_VERTICAL: "V",
                        ORIENTATION_OBLIQUE: "O", ORIENTATION_UNKNOWN: "?"}
        kind = "large" if self.is_large_surface else "small"
        elong = " elongated" if self.is_elongated else ""
        struct = " STRUCT" if self.is_elongated_struct else ""
        inferred = " INFERRED" if self.is_inferred else ""
        frame = f" frame_content={self.frame_content_state}" if self.frame_content_state is not None else ""
        return (f"Region(#{self.label} {orient_names.get(self.orientation, '?')} "
                f"area={self.area_px}px {kind}{elong}{struct}{inferred}{frame} "
                f"extent={tuple(round(v, 2) for v in self.extent_3d)}m "
                f"level={self.mean_level:.2f}m)")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GeometryAnalyzerConfig:
    """Configuration paramétrable du GeometryAnalyzer."""
    # Orientation
    horizontal_threshold: float = ORIENTATION_HORIZONTAL_THRESHOLD
    vertical_threshold: float = ORIENTATION_VERTICAL_THRESHOLD

    # Régions
    normal_similarity_threshold: float = NORMAL_SIMILARITY_THRESHOLD
    azimuth_buckets: int = AZIMUTH_BUCKETS
    min_region_area: int = MIN_REGION_AREA
    min_bridge_area: int = MIN_BRIDGE_AREA
    elongation_ratio_threshold: float = ELONGATION_RATIO_THRESHOLD
    large_surface_area_m2: float = LARGE_SURFACE_AREA_M2

    # Fusion et cadres
    adjacency_max_gap: int = ADJACENCY_MAX_GAP
    plane_level_tolerance_m: float = PLANE_LEVEL_TOLERANCE_M
    plane_offset_tolerance_m: float = PLANE_OFFSET_TOLERANCE_M
    frame_content_coplanar_tolerance_m: float = FRAME_CONTENT_COPLANAR_TOLERANCE_M

    # Structures élancées
    column_curvature_window: int = COLUMN_CURVATURE_WINDOW
    min_elongated_height: int = MIN_ELONGATED_HEIGHT
    curvature_nx_threshold: float = CURVATURE_NX_THRESHOLD
    polygonal_elongation_threshold: float = POLYGONAL_ELONGATION_THRESHOLD


# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class GeometryAnalyzer:
    """
    Analyse géométrique d'une grille de points 3D issue de build_pointcloud().

    Usage typique (voir points d'ancrage dans oakd_slam_pipeline.py) :

        analyzer = GeometryAnalyzer()
        normals, normal_valid = analyzer.compute_normals(coords, valid)
        orientation = analyzer.classify_orientation(normals, normal_valid)
        region_map, regions, bridge_candidates = analyzer.segment_regions(coords, normals, orientation, normal_valid)
        regions = analyzer.merge_coplanar_via_bridge(regions, bridge_candidates)
        frame_contents = analyzer.detect_frames(regions, region_map, depth_m_stabilized, STEP)
        elongated = analyzer.detect_elongated_vertical(normals, orientation, region_map, regions)

    coords : (H,W,3) -- repère (x, profondeur, -hauteur), cohérent avec
             build_pointcloud() du pipeline principal.
    valid  : (H,W) bool -- masque de validité des points.
    """

    def __init__(self, config: Optional[GeometryAnalyzerConfig] = None):
        """
        Args:
            config: Configuration personnalisée. Si None, utilise les
                    valeurs par défaut.
        """
        self.cfg = config or GeometryAnalyzerConfig()

    # ------------------------------------------------------------------
    # ÉTAPE 1a : normales locales
    # ------------------------------------------------------------------
    def compute_normals(self, coords: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule une normale approximative par point via les gradients locaux
        (différences finies sur la grille -- pas de PCA/voisinage circulaire,
        volontairement léger pour tourner en continu sur Pi5).

        Pour un point (i,j), on utilise les voisins (i,j+1) et (i+1,j) pour
        former deux vecteurs tangents, dont le produit vectoriel donne la
        normale locale.

        Retourne :
            normals     : (H,W,3) float32, normales unitaires (0 si non calculable)
            normal_valid: (H,W) bool, True si la normale a pu être calculée
                          (nécessite (i,j), (i,j+1) et (i+1,j) tous valides,
                          et un produit vectoriel non degenéré)
        """
        h, w = valid.shape

        # Tangente horizontale : (i,j+1) - (i,j)
        tangent_x = np.zeros((h, w, 3), dtype=np.float32)
        tangent_x[:, :-1] = coords[:, 1:] - coords[:, :-1]

        # Tangente verticale : (i+1,j) - (i,j)
        tangent_y = np.zeros((h, w, 3), dtype=np.float32)
        tangent_y[:-1, :] = coords[1:, :] - coords[:-1, :]

        # Produit vectoriel -> normale (non normalisée)
        cross = np.cross(tangent_x, tangent_y)
        norm = np.linalg.norm(cross, axis=-1)

        # Validité : les trois points du voisinage doivent être valides,
        # et le produit vectoriel ne doit pas être quasi-nul (points
        # quasi-colinéaires -> normale indéfinie, typique en bord de trou).
        neighbor_valid = np.zeros((h, w), dtype=bool)
        neighbor_valid[:-1, :-1] = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]

        nonzero = norm > 1e-6
        normal_valid = neighbor_valid & nonzero

        # Normalisation (éviter division par zéro hors du masque)
        safe_norm = np.where(nonzero, norm, 1.0)
        normals = cross / safe_norm[..., np.newaxis]

        # Orientation cohérente : la normale doit pointer vers la caméra
        # (composante "profondeur", axe Y de coords, négative côté caméra).
        # On force normals[...,1] <= 0 (pointe vers -profondeur, i.e. vers
        # l'observateur) pour une convention stable d'une frame à l'autre.
        flip = normals[..., 1] > 0
        normals[flip] *= -1

        normals[~normal_valid] = 0.0

        return normals, normal_valid

    # ------------------------------------------------------------------
    # ÉTAPE 1b : classification d'orientation
    # ------------------------------------------------------------------
    def classify_orientation(self, normals: np.ndarray, normal_valid: np.ndarray) -> np.ndarray:
        """
        Classifie chaque normale en HORIZONTAL / VERTICAL / OBLIQUE selon
        sa composante "hauteur" (3e composante de coords, axe -y du monde,
        donc normals[...,2] dans le repère (x, profondeur, -hauteur)).

        Retourne :
            orientation : (H,W) int8, valeurs parmi ORIENTATION_* ci-dessus.
                          ORIENTATION_UNKNOWN là où normal_valid est False.
        """
        h, w = normal_valid.shape
        orientation = np.full((h, w), ORIENTATION_UNKNOWN, dtype=np.int8)

        nz_abs = np.abs(normals[..., 2])

        is_horizontal = nz_abs > self.cfg.horizontal_threshold
        is_vertical = nz_abs < self.cfg.vertical_threshold
        is_oblique = normal_valid & ~is_horizontal & ~is_vertical

        orientation[normal_valid & is_horizontal] = ORIENTATION_HORIZONTAL
        orientation[normal_valid & is_vertical] = ORIENTATION_VERTICAL
        orientation[is_oblique] = ORIENTATION_OBLIQUE

        return orientation

    # ------------------------------------------------------------------
    # Fonction combinée (raccourci usuel) -- étape 1
    # ------------------------------------------------------------------
    def compute_normals_and_orientation(
        self, coords: np.ndarray, valid: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Raccourci : compute_normals() puis classify_orientation()."""
        normals, normal_valid = self.compute_normals(coords, valid)
        orientation = self.classify_orientation(normals, normal_valid)
        return normals, orientation

    # ------------------------------------------------------------------
    # ÉTAPE 2a : buckets directionnels (discrétisation de la normale)
    # ------------------------------------------------------------------
    def _direction_buckets(self, normals: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Discrétise chaque normale en un identifiant de bucket entier, pour
        que ndimage.label puisse séparer des régions de même catégorie
        d'orientation mais de direction différente (ex: deux murs
        perpendiculaires, tous deux ORIENTATION_VERTICAL).

        Le bucket combine :
          - la catégorie d'orientation (H/V/O) -> offset de base
          - pour V et O : l'azimut de la normale dans le plan horizontal
            (atan2(nz, nx)), discrétisé en azimuth_buckets secteurs.

        Retourne :
            buckets : (H,W) int32, -1 pour ORIENTATION_UNKNOWN.
        """
        h, w = orientation.shape
        buckets = np.full((h, w), -1, dtype=np.int32)

        nx = normals[..., 0]
        nz = normals[..., 2]

        azimuth = np.arctan2(nz, nx)  # [-pi, pi]
        n_buckets = self.cfg.azimuth_buckets
        azimuth_bucket = np.floor((azimuth + np.pi) / (2 * np.pi) * n_buckets).astype(np.int32)
        azimuth_bucket = np.clip(azimuth_bucket, 0, n_buckets - 1)

        base_h = 0
        base_v = n_buckets
        base_o = n_buckets * 2

        is_h = (orientation == ORIENTATION_HORIZONTAL)
        is_v = (orientation == ORIENTATION_VERTICAL)
        is_o = (orientation == ORIENTATION_OBLIQUE)

        buckets[is_h] = base_h
        buckets[is_v] = base_v + azimuth_bucket[is_v]
        buckets[is_o] = base_o + azimuth_bucket[is_o]

        return buckets

    # ------------------------------------------------------------------
    # ÉTAPE 2b : régions connexes + caractérisation
    # ------------------------------------------------------------------
    def segment_regions(
        self,
        coords: np.ndarray,
        normals: np.ndarray,
        orientation: np.ndarray,
        normal_valid: np.ndarray,
    ) -> Tuple[np.ndarray, List[Region], List[Region]]:
        """
        Segmente la grille en régions connexes de normale similaire et
        caractérise chacune (aire, étendue, élongation, niveau).

        Deux seuils d'aire distincts :
          - min_region_area  : seuil normal -- une région en dessous est
            ignorée (bruit).
          - min_bridge_area  : seuil plus bas (<= min_region_area) -- une
            région entre les deux seuils est trop petite pour être une
            "surface"/"objet" à part entière, mais reste assez grande pour
            servir de PONT d'occlusion (cf. merge_coplanar_via_bridge /
            detect_frames). Permet aux petits objets (crochets,
            interrupteurs, poignées...) de recoller des fragments de
            mur/table sans polluer la liste des régions caractérisées.

        Retourne :
            region_map        : (H,W) int32, identifiant de région par point
                                 (0 = pas de région / point invalide / région
                                 trop petite même pour un pont)
            regions            : liste de Region (area_px >= min_region_area)
            bridge_candidates  : liste de Region (min_bridge_area <= area_px
                                  < min_region_area) -- utilisables
                                  uniquement comme pont, pas comme région
                                  normale. Leurs pixels sont aussi marqués
                                  dans region_map (avec leur propre label,
                                  partageant l'espace de labels avec `regions`).
        """
        h, w = orientation.shape
        buckets = self._direction_buckets(normals, orientation)

        region_map = np.zeros((h, w), dtype=np.int32)
        regions: List[Region] = []
        bridge_candidates: List[Region] = []
        next_label = 1

        unique_buckets = np.unique(buckets)
        for b in unique_buckets:
            if b < 0:
                continue  # ORIENTATION_UNKNOWN

            bucket_mask = (buckets == b) & normal_valid
            if not np.any(bucket_mask):
                continue

            # Connexité spatiale (4-connexité) à l'intérieur du bucket
            labeled, n_components = ndimage.label(bucket_mask)

            for comp_id in range(1, n_components + 1):
                comp_mask = (labeled == comp_id)
                area_px = int(np.sum(comp_mask))
                if area_px < self.cfg.min_bridge_area:
                    continue  # trop petit même pour un pont -- bruit

                region = self._characterize_region(comp_mask, coords, normals, orientation, buckets, area_px)

                region_map[comp_mask] = next_label
                region.label = next_label
                region.merged_into = next_label

                if area_px < self.cfg.min_region_area:
                    bridge_candidates.append(region)
                else:
                    regions.append(region)

                next_label += 1

        return region_map, regions, bridge_candidates

    def _characterize_region(self, comp_mask: np.ndarray, coords: np.ndarray, normals: np.ndarray,
                              orientation: np.ndarray, buckets: np.ndarray, area_px: int) -> Region:
        """Calcule les attributs géométriques d'une région à partir de son masque."""
        rows, cols = np.where(comp_mask)
        bbox_rows = (int(rows.min()), int(rows.max()))
        bbox_cols = (int(cols.min()), int(cols.max()))

        points = coords[comp_mask]  # (N,3)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        extent_3d = (float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), float(maxs[2] - mins[2]))

        nonzero_extents = [e for e in extent_3d if e > 1e-6]
        if nonzero_extents:
            elongation_ratio = max(extent_3d) / min(nonzero_extents) if min(nonzero_extents) > 1e-6 else float('inf')
        else:
            elongation_ratio = 1.0

        mean_level = float(-np.mean(points[:, 2]))  # -z = hauteur
        centroid_3d = (float(np.mean(points[:, 0])), float(np.mean(points[:, 1])), float(np.mean(points[:, 2])))

        region_normals = normals[comp_mask]
        mean_normal_raw = np.mean(region_normals, axis=0)
        mn_norm = np.linalg.norm(mean_normal_raw)
        if mn_norm > 1e-6:
            mean_normal = (float(mean_normal_raw[0] / mn_norm),
                           float(mean_normal_raw[1] / mn_norm),
                           float(mean_normal_raw[2] / mn_norm))
        else:
            mean_normal = (0.0, 0.0, 0.0)

        # Orientation/azimut dominants = ceux du premier point (tous les
        # points du composant partagent le même bucket directionnel)
        dominant_orientation = int(orientation[comp_mask][0])
        dominant_bucket = int(buckets[comp_mask][0])
        if dominant_orientation == ORIENTATION_HORIZONTAL:
            azimuth_bucket = -1  # non discriminant pour l'horizontal
        else:
            base = self.cfg.azimuth_buckets if dominant_orientation == ORIENTATION_VERTICAL else self.cfg.azimuth_buckets * 2
            azimuth_bucket = dominant_bucket - base

        # Étendue 2D approximative selon l'orientation : pour une surface
        # horizontale, l'étendue pertinente est (dx, dy_profondeur) ; pour
        # une surface verticale/oblique, c'est (max(dx,dy), dz_hauteur).
        dx, dy, dz = extent_3d
        if dominant_orientation == ORIENTATION_HORIZONTAL:
            area_m2 = dx * dy
        else:
            area_m2 = max(dx, dy) * dz

        is_large_surface = area_m2 > self.cfg.large_surface_area_m2
        is_elongated = elongation_ratio > self.cfg.elongation_ratio_threshold

        return Region(
            label=0,  # assigné par l'appelant
            orientation=dominant_orientation,
            azimuth_bucket=azimuth_bucket,
            area_px=area_px,
            bbox_rows=bbox_rows,
            bbox_cols=bbox_cols,
            extent_3d=extent_3d,
            elongation_ratio=elongation_ratio,
            mean_level=mean_level,
            centroid_3d=centroid_3d,
            mean_normal=mean_normal,
            is_large_surface=is_large_surface,
            is_elongated=is_elongated,
        )

    # ------------------------------------------------------------------
    # ÉTAPE 2c : adjacence entre régions (utilitaire commun pont + cadre)
    # ------------------------------------------------------------------
    def _bboxes_adjacent(self, bbox_rows_a: Tuple[int, int], bbox_cols_a: Tuple[int, int],
                          bbox_rows_b: Tuple[int, int], bbox_cols_b: Tuple[int, int]) -> bool:
        """
        Deux bounding boxes (en cellules de grille) sont considérées
        adjacentes si :
          - elles se recouvrent (avec tolérance adjacency_max_gap) sur un axe
          - ET l'écart entre elles sur l'autre axe est <= adjacency_max_gap

        Rejette les contacts purement diagonaux (un seul coin en commun) :
        il faut un vrai recouvrement sur au moins un axe, pas seulement une
        proximité sur les deux axes simultanément.
        """
        max_gap = self.cfg.adjacency_max_gap

        ra0, ra1 = bbox_rows_a
        ca0, ca1 = bbox_cols_a
        rb0, rb1 = bbox_rows_b
        cb0, cb1 = bbox_cols_b

        row_gap = max(rb0 - ra1, ra0 - rb1, 0)
        col_gap = max(cb0 - ca1, ca0 - cb1, 0)

        row_overlap = not (ra1 < rb0 - max_gap or rb1 < ra0 - max_gap)
        col_overlap = not (ca1 < cb0 - max_gap or cb1 < ca0 - max_gap)

        # Adjacence "horizontale" : même bande de lignes (row_overlap),
        # colonnes proches (col_gap <= max_gap) -- régions côte à côte.
        horizontal_adjacent = row_overlap and (0 <= col_gap <= max_gap)
        # Adjacence "verticale" : même bande de colonnes (col_overlap),
        # lignes proches (row_gap <= max_gap) -- régions l'une sous l'autre.
        vertical_adjacent = col_overlap and (0 <= row_gap <= max_gap)

        return horizontal_adjacent or vertical_adjacent

    def _same_plane(self, region_a: Region, region_b: Region) -> bool:
        """
        Détermine si deux régions appartiennent probablement au même plan
        géométrique (même orientation/azimut, et position cohérente le
        long de la normale).

        - HORIZONTAL : même niveau (mean_level) à plane_level_tolerance_m près.
        - VERTICAL/OBLIQUE : même azimuth_bucket, et offset du centroïde
          projeté sur la normale moyenne cohérent à plane_offset_tolerance_m près.
        """
        if region_a.orientation != region_b.orientation:
            return False

        if region_a.orientation == ORIENTATION_HORIZONTAL:
            return abs(region_a.mean_level - region_b.mean_level) <= self.cfg.plane_level_tolerance_m

        if region_a.azimuth_bucket != region_b.azimuth_bucket:
            return False

        na = np.array(region_a.mean_normal)
        nb = np.array(region_b.mean_normal)
        mean_n = na + nb
        n_norm = np.linalg.norm(mean_n)
        if n_norm < 1e-6:
            return False
        mean_n /= n_norm

        ca = np.array(region_a.centroid_3d)
        cb = np.array(region_b.centroid_3d)
        offset_diff = abs(np.dot(ca - cb, mean_n))

        return offset_diff <= self.cfg.plane_offset_tolerance_m

    # ------------------------------------------------------------------
    # ÉTAPE 2d : fusion par pont d'occlusion (avec transitivité)
    # ------------------------------------------------------------------
    def merge_coplanar_via_bridge(self, regions: List[Region],
                                    bridge_candidates: Optional[List[Region]] = None) -> List[Region]:
        """
        Fusionne en une région virtuelle UNIQUE toutes les régions du même
        plan approximatif (cf. _same_plane) reliées entre elles, directement
        ou en chaîne, par des ponts d'occlusion -- régions d'orientation
        différente adjacentes à au moins deux d'entre elles.

        Transitivité : si A-B sont reliées par un pont, et B-C par un autre
        pont (même si A et C ne sont adjacentes à aucun pont commun direct),
        A, B et C sont fusionnées dans la MÊME région virtuelle. Implémenté
        via union-find sur l'ensemble des régions de `regions`.

        bridge_candidates : régions trop petites pour être des "surfaces"
        normales (cf. segment_regions / min_bridge_area) mais utilisables
        comme pont -- typiquement crochets, poignées, interrupteurs. Elles
        ne sont jamais fusionnées elles-mêmes (ne sont pas A/B), seulement
        utilisées comme C.

        Les régions A, B, ... d'origine voient leur `merged_into` mis à jour
        vers le label de la région virtuelle commune. Chaque région virtuelle
        est ajoutée à la liste retournée, avec is_inferred=True et une bbox/
        extent englobant tous ses membres et tous les ponts impliqués.

        Ne modifie pas region_map (la fusion est une information
        complémentaire, pas un redécoupage des pixels mesurés).

        Retourne :
            regions : liste mise à jour (originaux + régions virtuelles
                      ajoutées en fin de liste, une par composante connexe
                      de taille >= 2).
        """
        n = len(regions)
        if n < 2:
            return regions

        bridge_candidates = bridge_candidates or []
        # Pool de ponts potentiels : régions normales + petits objets-ponts.
        # Un pont ne peut pas être lui-même un membre A/B (filtré plus bas
        # via `is_inferred` pour les normales, et bridge_candidates ne sont
        # jamais ajoutées comme membres par construction).
        bridge_pool: List[Region] = [r for r in regions if not r.is_inferred] + list(bridge_candidates)

        # --- Union-Find sur les indices de `regions` ---
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # --- Recherche des paires (A,B) reliées par un pont C ---
        for i in range(n):
            region_a = regions[i]
            if region_a.is_inferred:
                continue
            for j in range(i + 1, n):
                region_b = regions[j]
                if region_b.is_inferred:
                    continue
                if not self._same_plane(region_a, region_b):
                    continue
                if find(i) == find(j):
                    continue  # déjà dans la même composante

                for region_c in bridge_pool:
                    if region_c.label in (region_a.label, region_b.label):
                        continue
                    if region_c.orientation == region_a.orientation:
                        continue  # le pont doit être d'une autre orientation

                    adj_a = self._bboxes_adjacent(region_a.bbox_rows, region_a.bbox_cols,
                                                   region_c.bbox_rows, region_c.bbox_cols)
                    adj_b = self._bboxes_adjacent(region_b.bbox_rows, region_b.bbox_cols,
                                                   region_c.bbox_rows, region_c.bbox_cols)

                    if adj_a and adj_b:
                        union(i, j)
                        break

        # --- Regroupement par composante connexe ---
        components: Dict[int, List[int]] = {}
        for idx in range(n):
            if regions[idx].is_inferred:
                continue
            root = find(idx)
            components.setdefault(root, []).append(idx)

        next_virtual_label = max((r.label for r in regions), default=0) + 1
        virtual_regions: List[Region] = []

        for indices in components.values():
            if len(indices) < 2:
                continue  # pas de fusion -- composante isolée

            members = [regions[idx] for idx in indices]
            virtual = self._make_virtual_region_from_group(next_virtual_label, members)

            for m in members:
                m.merged_into = next_virtual_label

            virtual_regions.append(virtual)
            next_virtual_label += 1

        return regions + virtual_regions

    def _make_virtual_region_from_group(self, label: int, members: List[Region]) -> Region:
        """
        Construit la région virtuelle résultant de la fusion d'un groupe de
        régions du même plan (>=2 membres), reliées en chaîne ou en étoile
        par des ponts d'occlusion. Généralisation n-aire de la fusion par
        paire : bbox/extent englobent l'ensemble du groupe.
        """
        first = members[0]

        bbox_rows = (min(m.bbox_rows[0] for m in members), max(m.bbox_rows[1] for m in members))
        bbox_cols = (min(m.bbox_cols[0] for m in members), max(m.bbox_cols[1] for m in members))

        extent_3d = (
            max(m.extent_3d[0] for m in members),
            max(m.extent_3d[1] for m in members),
            max(m.extent_3d[2] for m in members),
        )

        area_px = sum(m.area_px for m in members)  # zones occultées non comptées (non mesurées)
        mean_level = sum(m.mean_level for m in members) / len(members)
        centroid_3d = tuple(
            sum(m.centroid_3d[axis] for m in members) / len(members) for axis in range(3)
        )
        mean_normal = first.mean_normal  # même plan -> même normale (approx.)

        nonzero_extents = [e for e in extent_3d if e > 1e-6]
        if nonzero_extents:
            elongation_ratio = max(extent_3d) / min(nonzero_extents) if min(nonzero_extents) > 1e-6 else float('inf')
        else:
            elongation_ratio = 1.0

        dx, dy, dz = extent_3d
        if first.orientation == ORIENTATION_HORIZONTAL:
            area_m2 = dx * dy
        else:
            area_m2 = max(dx, dy) * dz

        virtual = Region(
            label=label,
            orientation=first.orientation,
            azimuth_bucket=first.azimuth_bucket,
            area_px=area_px,
            bbox_rows=bbox_rows,
            bbox_cols=bbox_cols,
            extent_3d=extent_3d,
            elongation_ratio=elongation_ratio,
            mean_level=mean_level,
            centroid_3d=centroid_3d,
            mean_normal=mean_normal,
            is_large_surface=area_m2 > self.cfg.large_surface_area_m2,
            is_elongated=elongation_ratio > self.cfg.elongation_ratio_threshold,
        )
        virtual.is_inferred = True
        # bridge_region_label n'a plus de sens unique avec plusieurs ponts
        # potentiels -- laissé à None ; les membres fusionnés restent
        # individuellement inspectables via merged_into.
        return virtual

    # ------------------------------------------------------------------
    # ÉTAPE 2e : détection de cadre + état géométrique du contenu
    # ------------------------------------------------------------------
    def detect_frames(self, regions: List[Region], region_map: np.ndarray,
                       depth_m_stabilized: np.ndarray, step: int) -> List[Region]:
        """
        Détecte les cadres : 4 régions verticales/obliques de même plan
        (même azimuth_bucket + niveau proche, cf. plane_level_tolerance_m)
        dont les bbox forment un rectangle creux (haut/bas/gauche/droite
        autour d'une zone intérieure). Caractérise l'état géométrique de la
        zone intérieure : FRAME_CONTENT_COPLANAR / RECESSED / UNKNOWN.

        Ne fait aucune classification porte/fenêtre/ouvert/fermé -- ce sont
        des observations géométriques brutes.

        Modifie les régions concernées in-place (is_frame=True sur les 4
        segments) et retourne une liste de régions virtuelles représentant
        chaque zone intérieure détectée, avec frame_content_state renseigné.

        Paramètres :
            depth_m_stabilized : (H_full, W_full) la depthmap stabilisée
                                 (résolution caméra, avant échantillonnage
                                 STEP) -- utilisée pour lire la profondeur
                                 de la zone intérieure du cadre.
            step               : valeur de STEP utilisée pour échantillonner
                                 coords/region_map depuis depth_m_stabilized.
        """
        next_label = max((r.label for r in regions), default=0) + 1
        frame_contents: List[Region] = []

        # Regrouper les régions non-inférées, verticales/obliques, par plan
        # (orientation + azimuth_bucket + niveau discrétisé).
        candidates = [r for r in regions if not r.is_inferred
                      and r.orientation in (ORIENTATION_VERTICAL, ORIENTATION_OBLIQUE)]

        plane_groups: Dict[tuple, List[Region]] = {}
        for r in candidates:
            key = (r.orientation, r.azimuth_bucket, round(r.mean_level / self.cfg.plane_level_tolerance_m))
            plane_groups.setdefault(key, []).append(r)

        for group in plane_groups.values():
            if len(group) < 4:
                continue

            # Cherche 4 segments formant un rectangle creux : top (au-dessus),
            # bottom (en dessous), left (à gauche), right (à droite) de la
            # zone intérieure, chacun couvrant la dimension transverse de
            # cette zone.
            for top in group:
                for bottom in group:
                    if bottom is top:
                        continue
                    if bottom.bbox_rows[0] <= top.bbox_rows[1]:
                        continue  # bottom doit être sous top
                    for left in group:
                        if left in (top, bottom):
                            continue
                        for right in group:
                            if right in (top, bottom, left):
                                continue
                            if right.bbox_cols[0] <= left.bbox_cols[1]:
                                continue  # right doit être à droite de left

                            inner_rows = (top.bbox_rows[1] + 1, bottom.bbox_rows[0] - 1)
                            inner_cols = (left.bbox_cols[1] + 1, right.bbox_cols[0] - 1)

                            if inner_rows[0] > inner_rows[1] or inner_cols[0] > inner_cols[1]:
                                continue  # pas de zone intérieure valide

                            if not self._is_valid_frame_segments(top, bottom, left, right, inner_rows, inner_cols):
                                continue

                            content_state = self._frame_content_state(
                                inner_rows, inner_cols, top, depth_m_stabilized, step
                            )

                            for seg in (top, bottom, left, right):
                                seg.is_frame = True

                            virtual = self._make_frame_content_region(
                                next_label, inner_rows, inner_cols, top, content_state
                            )
                            frame_contents.append(virtual)
                            next_label += 1

        return frame_contents

    def _is_valid_frame_segments(self, top: Region, bottom: Region, left: Region, right: Region,
                                   inner_rows: Tuple[int, int], inner_cols: Tuple[int, int]) -> bool:
        """
        Vérifie que les quatre segments d'un cadre candidat encadrent
        effectivement la zone intérieure : top/bottom doivent couvrir la
        largeur de la zone, left/right doivent couvrir sa hauteur.
        """
        top_covers = top.bbox_cols[0] <= inner_cols[0] and top.bbox_cols[1] >= inner_cols[1]
        bottom_covers = bottom.bbox_cols[0] <= inner_cols[0] and bottom.bbox_cols[1] >= inner_cols[1]
        left_covers = left.bbox_rows[0] <= inner_rows[0] and left.bbox_rows[1] >= inner_rows[1]
        right_covers = right.bbox_rows[0] <= inner_rows[0] and right.bbox_rows[1] >= inner_rows[1]

        return top_covers and bottom_covers and left_covers and right_covers

    def _frame_content_state(self, inner_rows: Tuple[int, int], inner_cols: Tuple[int, int],
                              frame_segment: Region, depth_m_stabilized: np.ndarray, step: int) -> int:
        """
        Lit la depthmap dans la zone intérieure du cadre et compare sa
        profondeur médiane à celle attendue pour le plan du cadre (déduite
        du centroïde de frame_segment).
        """
        r0, r1 = inner_rows
        c0, c1 = inner_cols

        h_full, w_full = depth_m_stabilized.shape
        row0_full, row1_full = r0 * step, min((r1 + 1) * step, h_full)
        col0_full, col1_full = c0 * step, min((c1 + 1) * step, w_full)

        if row0_full >= row1_full or col0_full >= col1_full:
            return FRAME_CONTENT_UNKNOWN

        inner_patch = depth_m_stabilized[row0_full:row1_full, col0_full:col1_full]
        valid_patch = inner_patch[inner_patch > IMPROBABLE_VALUE]

        if valid_patch.size < (inner_patch.size * 0.3):
            return FRAME_CONTENT_UNKNOWN

        inner_depth = float(np.median(valid_patch))
        frame_depth = float(frame_segment.centroid_3d[1])  # composante "profondeur" = axe Y de coords

        if abs(inner_depth - frame_depth) <= self.cfg.frame_content_coplanar_tolerance_m:
            return FRAME_CONTENT_COPLANAR

        return FRAME_CONTENT_RECESSED

    def _make_frame_content_region(self, label: int, inner_rows: Tuple[int, int], inner_cols: Tuple[int, int],
                                    frame_segment: Region, content_state: int) -> Region:
        """Construit la région virtuelle représentant la zone intérieure d'un cadre."""
        virtual = Region(
            label=label,
            orientation=frame_segment.orientation,
            azimuth_bucket=frame_segment.azimuth_bucket,
            area_px=(inner_rows[1] - inner_rows[0] + 1) * (inner_cols[1] - inner_cols[0] + 1),
            bbox_rows=inner_rows,
            bbox_cols=inner_cols,
            extent_3d=frame_segment.extent_3d,
            elongation_ratio=1.0,
            mean_level=frame_segment.mean_level,
            centroid_3d=frame_segment.centroid_3d,
            mean_normal=frame_segment.mean_normal,
            is_large_surface=frame_segment.is_large_surface,
            is_elongated=False,
        )
        virtual.is_inferred = True
        virtual.frame_content_state = content_state
        return virtual

    # ------------------------------------------------------------------
    # ÉTAPE 3 : structures verticales élancées (poteaux/troncs/montants)
    # ------------------------------------------------------------------
    def detect_elongated_vertical(
        self,
        normals: np.ndarray,
        orientation: np.ndarray,
        region_map: Optional[np.ndarray] = None,
        regions: Optional[List[Region]] = None,
    ) -> np.ndarray:
        """
        Détecte les points appartenant probablement à une structure
        verticale élancée (poteau, tronc, pied de meuble, montant de
        porte...), sans les étiqueter par nature. Deux mécanismes
        complémentaires :

        (a) Cylindrique : pour chaque colonne de la grille (x fixe), on
            regarde la composante horizontale de la normale (nx) le long de
            la colonne, restreinte aux points classés VERTICAL. Une surface
            cylindrique verticale vue de face produit un nx qui change de
            signe sur une fenêtre étroite (column_curvature_window) ; un mur
            plan produit un nx stable (même signe).

        (b) Polygonale : si region_map et regions sont fournis, toute région
            VERTICAL dont le ratio hauteur(dz)/largeur(max(dx,dy)) de la bbox
            3D dépasse polygonal_elongation_threshold est marquée comme
            structure élancée. Le remplissage se fait par MASQUE EXACT via
            region_map (region_map == region.label), pas par bounding box --
            évite de peindre des pixels hors de la région réelle (ex: fond
            visible à côté d'un poteau fin dans une bbox plus large).
            La région correspondante voit son champ is_elongated_struct mis
            à True in-place.

        Retourne :
            elongated : (H,W) bool, True pour les points retenus comme
                        candidats "structure verticale élancée".
        """
        h, w = orientation.shape
        elongated = np.zeros((h, w), dtype=bool)

        # --- (a) Détection cylindrique ---
        is_vertical = (orientation == ORIENTATION_VERTICAL)
        nx = normals[..., 0]

        win = self.cfg.column_curvature_window
        if h >= win:
            for row in range(h - win + 1):
                window_nx = nx[row:row + win, :]              # (win, w)
                window_valid = is_vertical[row:row + win, :]  # (win, w)

                sign = np.sign(window_nx)
                sign_where_valid = np.where(window_valid, sign, 0)

                has_pos_sign = np.any(sign_where_valid > 0, axis=0)
                has_neg_sign = np.any(sign_where_valid < 0, axis=0)
                sign_change = has_pos_sign & has_neg_sign

                cols_with_change = np.where(sign_change)[0]
                if cols_with_change.size > 0:
                    for c in cols_with_change:
                        elongated[row:row + win, c] |= window_valid[:, c]

        # Filtrage de hauteur minimale (élimine le bruit isolé de la
        # détection cylindrique, AVANT d'ajouter la détection polygonale
        # qui a sa propre validation via le seuil d'élongation de région).
        for col in range(w):
            column = elongated[:, col]
            if not np.any(column):
                continue
            diff = np.diff(np.concatenate(([0], column.astype(np.int8), [0])))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for s, e in zip(starts, ends):
                if (e - s) < self.cfg.min_elongated_height:
                    elongated[s:e, col] = False

        # --- (b) Détection polygonale (remplissage par masque exact) ---
        if region_map is not None and regions is not None:
            for region in regions:
                if region.is_inferred:
                    continue
                if region.orientation != ORIENTATION_VERTICAL:
                    continue

                dx, dy, dz = region.extent_3d
                width = max(dx, dy)
                height = dz
                if width > 1e-6 and height > width * self.cfg.polygonal_elongation_threshold:
                    mask = (region_map == region.label)
                    elongated |= mask
                    region.is_elongated_struct = True

        return elongated

    # ------------------------------------------------------------------
    # Visualisation (DEBUG)
    # ------------------------------------------------------------------
    @staticmethod
    def orientation_to_colors(orientation: np.ndarray) -> np.ndarray:
        """
        Convertit une grille d'orientation en couleurs RGB [0,1] pour
        visualisation dans le maillage 3D.

        Convention couleur :
            HORIZONTAL -> vert   (sol/plafond/surfaces posées)
            VERTICAL   -> bleu   (murs/faces verticales)
            OBLIQUE    -> orange (rampes, transitions)
            UNKNOWN    -> gris foncé (bords/trous)
        """
        h, w = orientation.shape
        colors = np.full((h, w, 3), 0.15, dtype=np.float64)  # gris foncé par défaut

        colors[orientation == ORIENTATION_HORIZONTAL] = (0.2, 0.85, 0.3)
        colors[orientation == ORIENTATION_VERTICAL] = (0.25, 0.45, 0.95)
        colors[orientation == ORIENTATION_OBLIQUE] = (0.95, 0.6, 0.15)

        return colors

    @staticmethod
    def region_to_colors(region_map: np.ndarray, regions: List[Region]) -> np.ndarray:
        """
        Colorise chaque région avec une couleur pseudo-aléatoire mais stable
        PAR LABEL (un RNG dédié par région, seedé sur son label), pour
        visualiser le découpage en régions. Les points hors région
        (region_map == 0) restent gris foncé.

        Note : les labels eux-mêmes ne sont pas stables d'une frame à
        l'autre (recalculés à chaque frame), donc les couleurs peuvent
        "clignoter" malgré la stabilité par-label -- comportement attendu
        tant que l'étape 4 (confiance/identité temporelle) n'est pas en place.
        """
        h, w = region_map.shape
        colors = np.full((h, w, 3), 0.1, dtype=np.float64)

        for region in regions:
            if region.is_inferred:
                continue  # pas de pixels propres dans region_map
            mask = (region_map == region.label)
            rng = np.random.default_rng(region.label)
            color = 0.3 + 0.7 * rng.random(3)
            colors[mask] = color

        return colors

    @staticmethod
    def elongated_to_colors(elongated: np.ndarray, base_colors: np.ndarray) -> np.ndarray:
        """
        Surcharge base_colors en jaune électrique partout où elongated est
        True, laisse base_colors inchangé ailleurs. Utilisé pour la
        visualisation 3D (points/maillage en jaune électrique).
        """
        colors = base_colors.copy()
        colors[elongated] = (1.0, 1.0, 0.0)  # jaune électrique
        return colors

    @staticmethod
    def frame_content_to_overlay_mask(frame_contents: List[Region], region_map: np.ndarray) -> np.ndarray:
        """
        Construit un masque (H,W) int8 indiquant, pour les zones intérieures
        de cadres détectées, leur état géométrique (FRAME_CONTENT_*).
        -1 = hors zone de cadre.

        Note : les régions virtuelles de contenu de cadre ne correspondent
        pas à des pixels mesurés dans region_map (elles couvrent une zone
        de la grille, potentiellement creuse) -- ce masque est donc
        construit par bbox plutôt que par masque de pixels exact.
        """
        h, w = region_map.shape
        overlay = np.full((h, w), -1, dtype=np.int8)

        for region in frame_contents:
            r0, r1 = region.bbox_rows
            c0, c1 = region.bbox_cols
            r0, r1 = max(0, r0), min(h - 1, r1)
            c0, c1 = max(0, c0), min(w - 1, c1)
            if r0 > r1 or c0 > c1:
                continue
            overlay[r0:r1 + 1, c0:c1 + 1] = region.frame_content_state

        return overlay

    @staticmethod
    def frame_overlay_to_colors(frame_overlay: np.ndarray, base_colors: np.ndarray) -> np.ndarray:
        """
        Surcharge base_colors en jaune électrique partout où frame_overlay
        indique une zone intérieure de cadre détectée (frame_overlay >= 0),
        laisse base_colors inchangé ailleurs (frame_overlay == -1).

        Même teinte que elongated_to_colors (jaune électrique), mais avec
        une intensité modulée par l'état géométrique du contenu :
          - FRAME_CONTENT_COPLANAR (panneau)  -> jaune plein (1.0, 1.0, 0.0)
          - FRAME_CONTENT_RECESSED (ouverture)
            / FRAME_CONTENT_UNKNOWN            -> jaune atténué (0.6, 0.6, 0.0)
        Permet de distinguer "cadre détecté, contenu coplanaire" de "cadre
        détecté, contenu en retrait/indéterminé" sans changer de teinte de
        base -- utile pour le debug visuel sans surcharger le code couleur.
        """
        colors = base_colors.copy()

        coplanar_mask = (frame_overlay == FRAME_CONTENT_COPLANAR)
        other_mask = (frame_overlay == FRAME_CONTENT_RECESSED) | (frame_overlay == FRAME_CONTENT_UNKNOWN)

        colors[coplanar_mask] = (1.0, 1.0, 0.0)   # jaune électrique plein
        colors[other_mask] = (0.6, 0.6, 0.0)      # jaune atténué

        return colors
