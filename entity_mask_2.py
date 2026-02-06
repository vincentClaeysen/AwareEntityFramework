import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import random

# ----------------------------
# Configuration LIFE 2025
# ----------------------------
WIDTH, HEIGHT = 500, 500
FPS = 25

class StarField:
    def __init__(self):
        self.num_stars = 800
        self.stars = []
        for _ in range(self.num_stars):
            x, y = random.uniform(-6, 6), random.uniform(-6, 6)
            z = random.uniform(-10, -6)
            brightness = random.uniform(0.3, 1.0)
            size = random.choice([1.0, 1.5, 2.0])
            twinkle_offset = random.uniform(0, 6.28)
            self.stars.append([x, y, z, brightness, size, twinkle_offset])
        
        self.nebula_clouds = []
        for _ in range(12):
            x, y = random.uniform(-4, 4), random.uniform(-4, 4)
            z = random.uniform(-9, -7)
            scale = random.uniform(1.5, 3.5)
            intensity = random.uniform(0.15, 0.35)
            hue = random.choice([0, 1, 2])
            drift = [random.uniform(-0.005, 0.005) for _ in range(2)]
            self.nebula_clouds.append([x, y, z, scale, intensity, hue, drift[0], drift[1]])
        self.time = 0.0

    def update(self, dt):
        self.time += dt
        for cloud in self.nebula_clouds:
            cloud[0] += cloud[6]; cloud[1] += cloud[7]
            if abs(cloud[0]) > 5.0: cloud[6] *= -1
            if abs(cloud[1]) > 5.0: cloud[7] *= -1

    def draw(self, ambient_brightness):
        star_alpha = 1.0 - ambient_brightness * 0.6
        nebula_alpha = 1.0 - ambient_brightness * 0.8
        glPushMatrix()
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        neb_cols = [(0.1, 0.3, 0.6), (0.15, 0.5, 0.7), (0.2, 0.6, 0.8)]
        for cloud in self.nebula_clouds:
            x, y, z, scale, intensity, hue, _, _ = cloud
            color = neb_cols[hue]
            pulse = math.sin(self.time * 0.3 + x) * 0.15 + 0.85
            glPointSize(40.0 * scale)
            glBegin(GL_POINTS)
            glColor4f(color[0], color[1], color[2], intensity * pulse * nebula_alpha)
            glVertex3f(x, y, z)
            glEnd()
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        for star in self.stars:
            x, y, z, b, size, t_off = star
            tw = math.sin(self.time * 2.0 + t_off) * 0.3 + 0.7
            glPointSize(size)
            glBegin(GL_POINTS)
            f_b = b * tw * star_alpha
            glColor4f(f_b, f_b, f_b * 1.1, 1.0)
            glVertex3f(x, y, z)
            glEnd()
        glPopMatrix()

class LifeOrganism:
    def __init__(self):
        self.sim_time = 0.0
        self.config = {
            "cycle_duration": 3.0, 
            "expansion_ratio": 0.4, 
            "max_height_ratio": 0.80, 
            "min_height_ratio": 0.45, 
            "particle_count": 6500
        }
        self.pending_duration = 3.0
        self.unit_h = 3.7 
        self.recalculate_curves()

        # NOYAU (QUADS -15% Volume)
        self.res_u, self.res_v = 40, 40
        self.base_radius_factor = (0.62 / 0.83) * 0.85 
        self.rot_core = [0.0, 0.0, 0.0]
        self.drift_core = [random.uniform(0.4, 0.9) for _ in range(3)]
        
        # MEMBRANE & FOND
        self.starfield = StarField()
        self.rot_memb = [0.0, 0.0, 0.0]
        self.drift_memb = [random.uniform(0.15, 0.35) * 2.5 for _ in range(3)]
        self.particles = [[random.uniform(0, 2*math.pi), math.acos(random.uniform(-1,1)), random.random()] for _ in range(self.config["particle_count"])]

        # CHROMATIQUE NÉON
        self.palette_a = np.array([[0.0, 0.3, 1.0], [0.6, 0.0, 1.0]], dtype=float) # Bleu/Violet
        self.palette_b = np.array([[1.0, 0.8, 0.0], [1.0, 0.2, 0.0]], dtype=float) # Or/Orange
        self.current_palette = self.palette_a.copy()
        self.target_palette = self.palette_a.copy()
        self.is_transitioning = False
        self.transition_progress = 0

    def recalculate_curves(self):
        """Génère les tables d'accélération exponentielle (EXP) et logarithmique (RET)"""
        self.resp_lut = []
        total_f = max(1, int(self.config["cycle_duration"] * FPS))
        exp_f = int(total_f * self.config["expansion_ratio"])
        ret_f = total_f - exp_f
        min_r, max_r = (self.config["min_height_ratio"] * self.unit_h)/2.0, (self.config["max_height_ratio"] * self.unit_h)/2.0
        
        for f in range(exp_f):
            t = f / exp_f
            self.resp_lut.append(min_r + (max_r - min_r) * (1 - math.pow(1-t, 3)))
        for f in range(ret_f):
            t = f / ret_f
            self.resp_lut.append(min_r + (max_r - min_r) * (1 - (math.log(1 + 9*t)/math.log(10))))

    def adjust_duration(self, delta):
        self.pending_duration = max(0.25, self.pending_duration + delta)
        print(f"Période en attente : {self.pending_duration}s")

    def trigger_mutation(self):
        if not self.is_transitioning:
            self.start_palette = self.current_palette.copy()
            self.target_palette = self.palette_b if np.allclose(self.target_palette, self.palette_a) else self.palette_a
            self.is_transitioning = True
            self.transition_progress = 0

    def _get_color(self, val):
        mix = max(0.0, min(1.0, val))
        return self.current_palette[0] * (1.0 - mix) + self.current_palette[1] * mix

    def update(self):
        dt = 1.0 / FPS
        total_f = len(self.resp_lut)
        current_index = int((self.sim_time * FPS) % total_f)
        
        # CONDITION DE RECALCUL : Rétractation max ET changement de valeur
        if current_index == 0 and self.config["cycle_duration"] != self.pending_duration:
            self.config["cycle_duration"] = self.pending_duration
            self.recalculate_curves()
            self.sim_time = 0.0
            total_f = len(self.resp_lut)
            current_index = 0

        self.current_radius = self.resp_lut[current_index]
        self.current_core_base = self.current_radius * self.base_radius_factor
        
        self.sim_time += dt
        self.starfield.update(dt)
        for i in range(3):
            self.rot_core[i] += self.drift_core[i]
            self.rot_memb[i] += self.drift_memb[i]
            
        if self.is_transitioning:
            self.transition_progress += 1
            t = self.transition_progress / 25.0
            self.current_palette = self.start_palette + (self.target_palette - self.start_palette) * t
            if self.transition_progress >= 25: self.is_transitioning = False

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        brightness = np.mean(self.current_palette) * (self.current_radius / 1.5)
        glDisable(GL_DEPTH_TEST); self.starfield.draw(brightness)

        # NOYAU (QUADS OPAQUES)
        glEnable(GL_DEPTH_TEST); glPushMatrix()
        glTranslatef(0.0, math.sin(self.sim_time * 0.5) * 0.03, 0.0)
        glRotatef(self.rot_core[0], 1,0,0); glRotatef(self.rot_core[1], 0,1,0); glRotatef(self.rot_core[2], 0,0,1)
        glBegin(GL_QUADS)
        for i in range(self.res_u):
            for j in range(self.res_v):
                for d_u, d_v in [(0,0), (1,0), (1,1), (0,1)]:
                    u, v = (i + d_u) / self.res_u, (j + d_v) / self.res_v
                    lat, lng = np.pi * (u - 0.5), 2 * np.pi * v
                    x_r, y_r, z_r = math.cos(lng)*math.cos(lat), math.sin(lng)*math.cos(lat), math.sin(lat)
                    d = math.sin(x_r * 2.2 + self.sim_time) * math.cos(y_r * 1.8 + self.sim_time * 0.8) * 0.2
                    r = self.current_core_base + d + (lat * 0.00001)
                    c = self._get_color((d + 0.2) * 2.5); br = 1.0 + d * 1.5
                    glColor3f(c[0]*br, c[1]*br, c[2]*br)
                    glVertex3f(x_r * r, y_r * r, z_r * r)
        glEnd(); glPopMatrix()

        # MEMBRANE (ARRONDIE & BALANCED LUMINESCENCE)
        glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glPushMatrix()
        glRotatef(self.rot_memb[0], 1,0,0); glRotatef(self.rot_memb[1], 0,1,0); glRotatef(self.rot_memb[2], 0,0,1)
        glBegin(GL_POINTS)
        for theta, phi, seed in self.particles:
            wave = 0.04 * math.sin(4.0*theta + self.sim_time*1.2) + 0.04 * math.sin(4.0*phi - self.sim_time*1.0)
            r = self.current_radius + (self.current_radius * wave)
            x, y, z = r*math.sin(phi)*math.cos(theta), r*math.cos(phi), r*math.sin(phi)*math.sin(theta)
            depth = ((z/r)+1.0)*0.4 + 0.2
            c = self._get_color((math.sin(self.sim_time*0.7 + theta*1.5 + seed*6.28)+1.0)*0.5)
            rad = (3.4 + abs(wave) * 20.25) * depth
            glColor4f(c[0]*rad, c[1]*rad, c[2]*rad, 0.9*depth)
            glVertex3f(x, y, z)
        glEnd(); glPopMatrix()

def main():
    pygame.init(); pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("LIFE 2025 - Optimized Sync")
    clock = pygame.time.Clock(); gluPerspective(45, WIDTH/HEIGHT, 0.1, 50.0)
    glTranslatef(0.0, 0.0, -4.5); glEnable(GL_POINT_SMOOTH); glPointSize(1.5)
    organism = LifeOrganism()
    while True:
        for event in pygame.event.get():
            if event.type == QUIT: pygame.quit(); return
            if event.type == KEYDOWN:
                if event.key == K_s: organism.trigger_mutation()
                if event.key == K_LEFT: organism.adjust_duration(-0.25)
                if event.key == K_RIGHT: organism.adjust_duration(0.25)
        organism.update(); organism.draw(); pygame.display.flip(); clock.tick(FPS)

if __name__ == "__main__":
    main()
