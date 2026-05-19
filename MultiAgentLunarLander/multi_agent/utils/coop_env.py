# -*- coding: utf-8 -*-
# multi_agent/utils/coop_env.py
# Build mulit-agent overlay

#%% Import packages

import collections
import math

import numpy as np

import Box2D
from Box2D.b2 import (
    edgeShape, polygonShape, circleShape,
    fixtureDef, revoluteJointDef, contactListener,
)

from shared_utils.constants import (
    FPS, SCALE, MAIN_ENGINE_POWER, SIDE_ENGINE_POWER, INITIAL_RANDOM,
    LANDER_POLY, LEG_AWAY, LEG_DOWN, LEG_W, LEG_H, LEG_SPRING_TORQUE,
    SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY, MAIN_ENGINE_Y_LOCATION,
    VIEWPORT_W, VIEWPORT_H,
    COOP_SUCCESS_THRESHOLD, ZONE_PENALTY_SCALE, ZONE_BONUS, CONTACT_PENALTY, STEP_PENALTY,
    LANDER1_SPAWN_X_OFFSET, LANDER2_SPAWN_X_OFFSET,
    HOVER_WINDOW_SIZE, HOVER_PROGRESS_THRESHOLD_X, HOVER_PROGRESS_THRESHOLD_Y,
    HOVER_BASE_PENALTY, HOVER_PENALTY_EXPONENT, HOVER_MAX_PENALTY,
    HOVER_MIN_ALTITUDE, HOVER_COUNTER_DECAY, HOVER_ACTIVATION_STEP,
    HOVER_ESCALATION_STEP, HOVER_ESCALATION_MULTIPLIER,
    PROGRESS_BONUS_SCALE_TIER1, PROGRESS_BONUS_SCALE_TIER2,
)


#%% Class definitions

class CoopContactDetector(contactListener):
    '''Contact listener that distinguishes terrain crashes from lander-to-lander bounces.'''

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
        self.lander_contact = False

    def BeginContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body
        bodies = {bodyA, bodyB}

        all_lander1_bodies = {self.env.lander1} | set(self.env.legs1)
        all_lander2_bodies = {self.env.lander2} | set(self.env.legs2)

        # Define lander-to-lander contact
        if bodies & all_lander1_bodies and bodies & all_lander2_bodies:
            self.lander_contact = True
            return

        # Define crash conditions
        if self.env.lander1 in bodies:
            self.env.game_over1 = True
        if self.env.lander2 in bodies:
            self.env.game_over2 = True

        # Leg ground contact tracking
        for i in range(2):
            if self.env.legs1[i] in bodies:
                self.env.legs1[i].ground_contact = True
            if self.env.legs2[i] in bodies:
                self.env.legs2[i].ground_contact = True

    def EndContact(self, contact):
        bodyA = contact.fixtureA.body
        bodyB = contact.fixtureB.body
        bodies = {bodyA, bodyB}

        all_lander1_bodies = {self.env.lander1} | set(self.env.legs1)
        all_lander2_bodies = {self.env.lander2} | set(self.env.legs2)

        if bodies & all_lander1_bodies and bodies & all_lander2_bodies:
            self.lander_contact = False
            return

        for i in range(2):
            if self.env.legs1[i] in bodies:
                self.env.legs1[i].ground_contact = False
            if self.env.legs2[i] in bodies:
                self.env.legs2[i].ground_contact = False


class CooperativeLunarLander:
    '''Dual-lander environment with shared Box2D physics world.

    Parameters
    ----------
    obs_mode : str
        'partial'  — both agents get 12-dim (own 8 + partner x,y,vx,vy)
        'full'     — both agents get 16-dim (own 8 + partner full 8)
        'mixed'    — agent 1 gets 12-dim, agent 2 gets 16-dim
    '''

    def __init__(self, obs_mode='partial',
                 zone_penalty_scale=ZONE_PENALTY_SCALE,
                 zone_bonus=ZONE_BONUS, contact_penalty=CONTACT_PENALTY,
                 step_penalty=STEP_PENALTY, render_mode=None,
                 hover_window_size=HOVER_WINDOW_SIZE,
                 hover_progress_thresh_x=HOVER_PROGRESS_THRESHOLD_X,
                 hover_progress_thresh_y=HOVER_PROGRESS_THRESHOLD_Y,
                 hover_base_penalty=HOVER_BASE_PENALTY,
                 hover_penalty_exponent=HOVER_PENALTY_EXPONENT,
                 hover_max_penalty=HOVER_MAX_PENALTY,
                 hover_min_altitude=HOVER_MIN_ALTITUDE,
                 hover_counter_decay=HOVER_COUNTER_DECAY,
                 hover_activation_step=HOVER_ACTIVATION_STEP):
        if obs_mode not in ('partial', 'full', 'mixed'):
            raise ValueError(f'Unknown obs_mode {obs_mode!r}')
        self.obs_mode = obs_mode
        self.zone_penalty_scale = zone_penalty_scale
        self.zone_bonus = zone_bonus
        self.contact_penalty = contact_penalty
        self.step_penalty = step_penalty
        self.render_mode = render_mode

        # Hover penalty configuration
        self.hover_window_size = hover_window_size
        self.hover_progress_thresh_x = hover_progress_thresh_x
        self.hover_progress_thresh_y = hover_progress_thresh_y
        self.hover_base_penalty = hover_base_penalty
        self.hover_penalty_exponent = hover_penalty_exponent
        self.hover_max_penalty = hover_max_penalty
        self.hover_min_altitude = hover_min_altitude
        self.hover_counter_decay = hover_counter_decay
        self.hover_activation_step = hover_activation_step

        self.world = None
        self.moon = None
        self.lander1 = None
        self.lander2 = None
        self.legs1 = []
        self.legs2 = []
        self.particles = []
        self.sky_polys = []

        self.game_over1 = False
        self.game_over2 = False
        self.done1 = False
        self.done2 = False
        self.prev_shaping1 = None
        self.prev_shaping2 = None
        self.contact_count = 0

        # Hover penalty tracking
        self._hover_history1 = collections.deque(maxlen=self.hover_window_size)
        self._hover_history2 = collections.deque(maxlen=self.hover_window_size)
        self._hover_violations1 = 0
        self._hover_violations2 = 0
        self._step_count = 0
        self._prev_dist1 = None
        self._prev_dist2 = None

        # Pygame state
        self.screen = None
        self.clock = None
        self.surf = None

        # Random number generator
        self.np_random = np.random.default_rng()

    def _destroy(self):
        if self.world is None:
            return
        bodies_to_destroy = []
        if self.moon is not None:
            bodies_to_destroy.append(self.moon)
        if self.lander1 is not None:
            bodies_to_destroy.append(self.lander1)
        if self.lander2 is not None:
            bodies_to_destroy.append(self.lander2)
        bodies_to_destroy.extend(self.legs1)
        bodies_to_destroy.extend(self.legs2)
        bodies_to_destroy.extend(self.particles)
        for body in bodies_to_destroy:
            self.world.DestroyBody(body)
        self.moon = None
        self.lander1 = None
        self.lander2 = None
        self.legs1 = []
        self.legs2 = []
        self.particles = []

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _create_lander(self, initial_x, initial_y, category_bits, mask_bits,
                       leg_category_bits, leg_mask_bits, color1, color2):
        '''Create a lander body + 2 legs at the given position.'''
        lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=category_bits,
                maskBits=mask_bits,
                restitution=0.0,
            ),
        )
        lander.color1 = color1
        lander.color2 = color2

        lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            ),
            True,
        )

        legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=leg_category_bits,
                    maskBits=leg_mask_bits,
                ),
            )
            leg.ground_contact = False
            leg.color1 = color1
            leg.color2 = color2
            rjd = revoluteJointDef(
                bodyA=lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            legs.append(leg)

        return lander, legs

    def reset(self, seed=None):
        '''Reset the environment and create two landers.'''
        self._destroy()

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.world = Box2D.b2World(gravity=(0, -10.0))
        self.game_over1 = False
        self.game_over2 = False
        self.done1 = False
        self.done2 = False
        self.prev_shaping1 = None
        self.prev_shaping2 = None
        self.contact_count = 0
        self._hover_history1 = collections.deque(maxlen=self.hover_window_size)
        self._hover_history2 = collections.deque(maxlen=self.hover_window_size)
        self._hover_violations1 = 0
        self._hover_violations2 = 0
        self._step_count = 0
        self._prev_dist1 = None
        self._prev_dist2 = None
        self.particles = []

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Generate terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4

        # Zone landing targets midpoint on each half of the pad
        zone_center1 = (self.helipad_x1 + W / 2) / 2
        zone_center2 = (W / 2 + self.helipad_x2) / 2
        self.target_x1 = (zone_center1 - W / 2) / (W / 2)
        self.target_x2 = (zone_center2 - W / 2) / (W / 2)
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]

        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H / SCALE

        # Create Lander 1 (left zone — blue)
        initial_x1 = VIEWPORT_W / SCALE / 2 + LANDER1_SPAWN_X_OFFSET
        self.lander1, self.legs1 = self._create_lander(
            initial_x=initial_x1,
            initial_y=initial_y,
            category_bits=0x0010,
            mask_bits=0x0001 | 0x0040 | 0x0080,
            leg_category_bits=0x0020,
            leg_mask_bits=0x0001 | 0x0040 | 0x0080,
            color1=(100, 140, 230),
            color2=(60, 80, 160),
        )

        # Create Lander 2 (right zone — orange)
        initial_x2 = VIEWPORT_W / SCALE / 2 + LANDER2_SPAWN_X_OFFSET
        self.lander2, self.legs2 = self._create_lander(
            initial_x=initial_x2,
            initial_y=initial_y,
            category_bits=0x0040,
            mask_bits=0x0001 | 0x0010 | 0x0020,
            leg_category_bits=0x0080,
            leg_mask_bits=0x0001 | 0x0010 | 0x0020,
            color1=(230, 150, 60),
            color2=(180, 100, 30),
        )

        # Set up contact detector
        self.world.contactListener_keepref = CoopContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        self.drawlist1 = [self.lander1] + self.legs1
        self.drawlist2 = [self.lander2] + self.legs2

        # Get initial observations
        obs1, obs2 = self._get_observations()

        if self.render_mode == 'human':
            self.render()

        return obs1, obs2

    def _get_state(self, lander, legs):
        '''Compute 8-dim state for one lander (same normalization as gymnasium).'''
        pos = lander.position
        vel = lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            lander.angle,
            20.0 * lander.angularVelocity / FPS,
            1.0 if legs[0].ground_contact else 0.0,
            1.0 if legs[1].ground_contact else 0.0,
        ]
        return state

    def _build_obs(self, s1, s2):
        '''Build observation arrays based on obs_mode.'''
        if self.obs_mode == 'partial':
            obs1 = np.array(s1 + [s2[0], s2[1], s2[2], s2[3]], dtype=np.float32)
            obs2 = np.array(s2 + [s1[0], s1[1], s1[2], s1[3]], dtype=np.float32)
        elif self.obs_mode == 'full':
            obs1 = np.array(s1 + s2, dtype=np.float32)
            obs2 = np.array(s2 + s1, dtype=np.float32)
        else:  # mixed
            obs1 = np.array(s1 + [s2[0], s2[1], s2[2], s2[3]], dtype=np.float32)
            obs2 = np.array(s2 + s1, dtype=np.float32)
        return obs1, obs2

    def _get_observations(self):
        '''Build observations for both agents.'''
        s1 = self._get_state(self.lander1, self.legs1)
        s2 = self._get_state(self.lander2, self.legs2)
        return self._build_obs(s1, s2)

    def _apply_action(self, lander, action):
        '''Apply engine forces to a lander. Returns (m_power, s_power).'''
        tip = (math.sin(lander.angle), math.cos(lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if action == 2:
            m_power = 1.0
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (lander.position[0] + ox, lander.position[1] + oy)
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse(
                    (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                    impulse_pos, True,
                )
            lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos, True,
            )

        s_power = 0.0
        if action in [1, 3]:
            direction = action - 2
            s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                lander.position[0] + ox - tip[0] * 17 / SCALE,
                lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                    impulse_pos, True,
                )
            lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos, True,
            )

        return m_power, s_power

    def _compute_hover_penalty(self, state, target_x, hover_history, violations):
        '''Compute hover penalty based on directional progress toward landing target.

        Returns (penalty_value, updated_violations).
        '''
        dist_to_target_x = abs(state[0] - target_x)
        current_y = state[1]
        hover_history.append((dist_to_target_x, current_y))

        # Not enough history yet
        if len(hover_history) < self.hover_window_size:
            return 0.0, violations

        # Skip if both legs touching
        if state[6] == 1.0 and state[7] == 1.0:
            violations = max(0, violations - self.hover_counter_decay)
            return 0.0, violations

        # Skip below minimum altitude
        if state[1] < self.hover_min_altitude:
            violations = max(0, violations - self.hover_counter_decay)
            return 0.0, violations

        # Progress over window
        oldest_dist_x, oldest_y = hover_history[0]
        newest_dist_x, newest_y = hover_history[-1]
        delta_goal_x = oldest_dist_x - newest_dist_x
        delta_y = oldest_y - newest_y

        # Violation: insufficient progress in BOTH dimensions
        if (delta_goal_x < self.hover_progress_thresh_x
                and delta_y < self.hover_progress_thresh_y):
            violations += 1
            penalty = min(
                self.hover_base_penalty * (
                    self.hover_penalty_exponent ** violations),
                self.hover_max_penalty)
            return penalty, violations

        violations = max(0, violations - self.hover_counter_decay)
        return 0.0, violations

    def step(self, action1, action2):
        '''Step both landers in the shared physics world.'''
        self._step_count += 1

        m_power1, s_power1 = 0.0, 0.0
        m_power2, s_power2 = 0.0, 0.0

        if not self.done1:
            m_power1, s_power1 = self._apply_action(self.lander1, action1)
        if not self.done2:
            m_power2, s_power2 = self._apply_action(self.lander2, action2)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        self._clean_particles(False)

        # Get states and build observations
        state1 = self._get_state(self.lander1, self.legs1)
        state2 = self._get_state(self.lander2, self.legs2)
        obs1, obs2 = self._build_obs(state1, state2)

        # Track lander-to-lander contact
        lander_contact = self.world.contactListener_keepref.lander_contact
        if lander_contact:
            self.contact_count += 1

        # Compute reward for Agent 1
        reward1 = 0.0
        if not self.done1:
            shaping1 = (
                -100 * np.sqrt((state1[0] - self.target_x1) ** 2 + state1[1] ** 2)
                - 50 * np.sqrt(state1[2] ** 2 + state1[3] ** 2)
                - 100 * abs(state1[4])
                + 10 * state1[6]
                + 10 * state1[7]
            )
            if self.prev_shaping1 is not None:
                reward1 = shaping1 - self.prev_shaping1
            self.prev_shaping1 = shaping1

            reward1 -= m_power1 * 0.10
            reward1 -= s_power1 * 0.03
            reward1 -= self.step_penalty

            # Zone shaping: Agent 1 should stay left (x < 0)
            if state1[0] > 0:
                reward1 -= self.zone_penalty_scale * state1[0]
            else:
                reward1 += self.zone_bonus

            # Contact penalty (disabled in final 300 steps to not block landing)
            if lander_contact and self._step_count < 700:
                reward1 -= self.contact_penalty

            # Hover penalty (only after activation step, increases at 800)
            if self._step_count >= self.hover_activation_step:
                hover_pen1, self._hover_violations1 = self._compute_hover_penalty(
                    state1, self.target_x1, self._hover_history1,
                    self._hover_violations1)
                if self._step_count >= HOVER_ESCALATION_STEP:
                    hover_pen1 *= HOVER_ESCALATION_MULTIPLIER
                reward1 -= hover_pen1

            # Progress bonus for closing distance to landing zone
            curr_dist1 = np.sqrt(
                (state1[0] - self.target_x1) ** 2 + state1[1] ** 2)
            if (self._prev_dist1 is not None
                    and self._step_count >= self.hover_activation_step):
                progress1 = self._prev_dist1 - curr_dist1
                if progress1 > 0:
                    scale = (PROGRESS_BONUS_SCALE_TIER2
                             if self._step_count >= HOVER_ESCALATION_STEP
                             else PROGRESS_BONUS_SCALE_TIER1)
                    reward1 += scale * progress1
            self._prev_dist1 = curr_dist1

            # Terminal conditions
            terminated1 = False
            if self.game_over1 or abs(state1[0]) >= 1.0:
                terminated1 = True
                reward1 = -50
            if not self.lander1.awake:
                terminated1 = True
                both_legs1 = state1[6] == 1.0 and state1[7] == 1.0
                in_zone1 = state1[0] < 0
                if both_legs1 and in_zone1:
                    reward1 = +500
                elif both_legs1:
                    reward1 = +250
                else:
                    reward1 = 0

            if terminated1 and not self.done1:
                self.done1 = True
                self.lander1.linearVelocity = (0, 0)
                self.lander1.angularVelocity = 0
                for leg in self.legs1:
                    leg.linearVelocity = (0, 0)
                    leg.angularVelocity = 0

        # Compute reward for Agent 2
        reward2 = 0.0
        if not self.done2:
            shaping2 = (
                -100 * np.sqrt((state2[0] - self.target_x2) ** 2 + state2[1] ** 2)
                - 50 * np.sqrt(state2[2] ** 2 + state2[3] ** 2)
                - 100 * abs(state2[4])
                + 10 * state2[6]
                + 10 * state2[7]
            )
            if self.prev_shaping2 is not None:
                reward2 = shaping2 - self.prev_shaping2
            self.prev_shaping2 = shaping2

            reward2 -= m_power2 * 0.10
            reward2 -= s_power2 * 0.03
            reward2 -= self.step_penalty

            # Zone shaping: Agent 2 should stay right (x > 0)
            if state2[0] < 0:
                reward2 -= self.zone_penalty_scale * abs(state2[0])
            else:
                reward2 += self.zone_bonus

            # Contact penalty (disabled in final 300 steps to not block landing)
            if lander_contact and self._step_count < 700:
                reward2 -= self.contact_penalty

            # Hover penalty (only after activation step, increases at 800)
            if self._step_count >= self.hover_activation_step:
                hover_pen2, self._hover_violations2 = self._compute_hover_penalty(
                    state2, self.target_x2, self._hover_history2,
                    self._hover_violations2)
                if self._step_count >= HOVER_ESCALATION_STEP:
                    hover_pen2 *= HOVER_ESCALATION_MULTIPLIER
                reward2 -= hover_pen2

            # Progress bonus for closing distance to landing zone
            curr_dist2 = np.sqrt(
                (state2[0] - self.target_x2) ** 2 + state2[1] ** 2)
            if (self._prev_dist2 is not None
                    and self._step_count >= self.hover_activation_step):
                progress2 = self._prev_dist2 - curr_dist2
                if progress2 > 0:
                    scale = (PROGRESS_BONUS_SCALE_TIER2
                             if self._step_count >= HOVER_ESCALATION_STEP
                             else PROGRESS_BONUS_SCALE_TIER1)
                    reward2 += scale * progress2
            self._prev_dist2 = curr_dist2

            # Terminal conditions
            terminated2 = False
            if self.game_over2 or abs(state2[0]) >= 1.0:
                terminated2 = True
                reward2 = -50
            if not self.lander2.awake:
                terminated2 = True
                both_legs2 = state2[6] == 1.0 and state2[7] == 1.0
                in_zone2 = state2[0] > 0
                if both_legs2 and in_zone2:
                    reward2 = +500
                elif both_legs2:
                    reward2 = +250
                else:
                    reward2 = 0

            if terminated2 and not self.done2:
                self.done2 = True
                self.lander2.linearVelocity = (0, 0)
                self.lander2.angularVelocity = 0
                for leg in self.legs2:
                    leg.linearVelocity = (0, 0)
                    leg.angularVelocity = 0

        both_done = self.done1 and self.done2

        if self.render_mode == 'human':
            self.render()

        return (obs1, reward1, self.done1), (obs2, reward2, self.done2), both_done

    def _create_particle(self, mass, x, y, power):
        '''Create a visual particle for engine exhaust.'''
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x0001,
                restitution=0.3,
            ),
        )
        p.ttl = 1.0
        p.color1 = (int(max(0.2, 0.15 + p.ttl) * 255),
                     int(max(0.2, 0.5 * p.ttl) * 255),
                     int(max(0.2, 0.5 * p.ttl) * 255))
        p.color2 = p.color1
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particles=False):
        '''Remove expired particles.'''
        while self.particles and (all_particles or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def render(self):
        '''Render both landers in the same frame.'''
        if self.render_mode is None:
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise ImportError('pygame is required for rendering') from e

        if self.screen is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            import pygame
            self.clock = pygame.time.Clock()

        import pygame
        from pygame import gfxdraw

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        # Update particle colors
        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                int(max(0.2, 0.15 + obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
                int(max(0.2, 0.5 * obj.ttl) * 255),
            )
            obj.color2 = obj.color1

        self._clean_particles(False)

        # Draw terrain
        for p in self.sky_polys:
            scaled_poly = []
            for coord in p:
                scaled_poly.append((coord[0] * SCALE, coord[1] * SCALE))
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, (0, 0, 0))

        # Draw particles + both landers
        for obj in self.particles + self.drawlist1 + self.drawlist2:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    center = trans * f.shape.pos * SCALE
                    center = (int(center[0]), int(center[1]))
                    radius = int(f.shape.radius * SCALE)
                    pygame.draw.circle(self.surf, color=obj.color1,
                                       center=center, radius=radius)
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    path = [(int(p[0]), int(p[1])) for p in path]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    gfxdraw.aapolygon(self.surf, path, obj.color1)
                    pygame.draw.aalines(self.surf, color=obj.color2, points=path, closed=True)

        # Draw colored landing pad strips under each zone
        pad_y = int(self.helipad_y * SCALE)
        pad_h = 4
        left_x1 = int(self.helipad_x1 * SCALE)
        center_x = VIEWPORT_W // 2
        right_x2 = int(self.helipad_x2 * SCALE)
        # Agent 1 (blue) zone: left flag to center
        pygame.draw.rect(self.surf, (100, 140, 230),
                         (left_x1, pad_y - pad_h, center_x - left_x1, pad_h))
        # Agent 2 (orange) zone: center to right flag
        pygame.draw.rect(self.surf, (230, 150, 60),
                         (center_x, pad_y - pad_h, right_x2 - center_x, pad_h))

        # Draw color-matched landing pad flags
        flag_colors = [
            (100, 140, 230),  # left flag: blue (Agent 1)
            (230, 150, 60),   # right flag: orange (Agent 2)
        ]
        for x, flag_color in zip([self.helipad_x1, self.helipad_x2], flag_colors):
            x = x * SCALE
            flagy1 = self.helipad_y * SCALE
            flagy2 = flagy1 + 50
            pygame.draw.line(self.surf, color=(255, 255, 255),
                             start_pos=(x, flagy1), end_pos=(x, flagy2), width=1)
            pygame.draw.polygon(
                self.surf, color=flag_color,
                points=[(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(int(x), int(flagy2)), (int(x), int(flagy2 - 10)), (int(x + 25), int(flagy2 - 5))],
                flag_color,
            )

        # Flip Y axis
        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == 'human':
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(FPS)
            pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

#%% End of Script