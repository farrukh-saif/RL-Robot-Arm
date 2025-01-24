import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math

class RobotArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, targetSize=15, maxSteps=200, screen_size=400):
        super().__init__()
        
        self.targetSize = targetSize
        self.maxSteps = maxSteps

        # Rendering
        self.render_mode = render_mode
        self.screen_size = screen_size
        self.screen = None
        self.clock = None

        # Fixed arm lengths
        self.humerus_length = 100
        self.forearm_length = 100
        self.max_reach = self.humerus_length + self.forearm_length
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -self.max_reach, -self.max_reach, 0], dtype=np.float32),
            high=np.array([90, 180, self.max_reach, self.max_reach, self.max_reach], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize state through reset
        self.reset()

    def _spawnTarget(self):
        #Randomly Spawn in a quarter-circle
        newTarget = np.array([self.np_random.uniform(0, self.max_reach), -self.np_random.uniform(0, self.max_reach)])
        r = self.max_reach * np.sqrt(np.random.uniform(0, 1))
        theta = np.random.uniform(0, np.pi/2)
        x = r * np.cos(theta)
        y = - r * np.sin(theta)
        newTarget = np.array([x, y])

        #Check if in left semi-circle of the bottom-right quadrant - if so move to top semi-circle
        oldx, oldy = newTarget
        if np.linalg.norm((0, -100) - newTarget) < 100:
            newTarget = np.array([-oldy, oldx])

        return newTarget

    def reset(self, seed=None):
        super().reset()
        self.a1 = 45
        self.a2 = 90
        
        self.target = self._spawnTarget()

        self._elapsed_steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.a1, # Shoulder angle
            self.a2, # Elbow angle
            *self.target, # Target position
            self._calculate_distance() # Distance to target
            ], dtype=np.float32)

    def _calculate_positions(self):
        a1_rad = np.deg2rad(self.a1 - 90)
        a2_rad = a1_rad + np.deg2rad(self.a2)
        
        # Calculate joint positions
        shoulder = (0, 0)
        elbow = (
            self.humerus_length * np.cos(a1_rad),
            self.humerus_length * np.sin(a1_rad)
        )
        wrist = (
            elbow[0] + self.forearm_length * np.cos(a2_rad),
            elbow[1] + self.forearm_length * np.sin(a2_rad)
        )
        
        return shoulder, elbow, wrist

    def _calculate_distance(self):
        *_, wrist = self._calculate_positions()
        return np.linalg.norm(wrist - self.target)

    def step(self, action): 
        action = int(action)
        
        # Store previous state for reward calculation
        prev_distance = self._calculate_distance()

        # Update angles
        angle_actions = {
            0: lambda: (self.a1 < 90, 1, 0),
            1: lambda: (self.a1 > 0, -1, 0),
            2: lambda: (self.a2 < 180, 1, 1),
            3: lambda: (self.a2 > 0, -1, 1)
        }
        
        if action in angle_actions:
            condition, delta, idx = angle_actions[action]()
            if condition:
                if idx == 0:
                    self.a1 += delta
                elif idx == 1:
                    self.a2 += delta

        # Calculate new state
        distance = self._calculate_distance()
        self._elapsed_steps += 1

        # Reward components
        distance_reduction = prev_distance - distance
        time_penalty = 1.0
        action_penalty = 0.05
        success_bonus = 1000 if distance < self.targetSize else 0
        proximity_reward = 1 / (distance + 1e-6)  # Avoid division by zero

        reward = (
            distance_reduction * 5.0 +      # Encourage moving closer
            proximity_reward * 0.5 +        # Incentivize proximity
            success_bonus -                 # Big reward for success
            time_penalty -                  # Discourage slow solutions
            action_penalty                  # Discourage unnecessary movements
        )

        # Termination conditions
        terminated = distance < self.targetSize or self._elapsed_steps >= self.maxSteps

        return self._get_obs(), reward, terminated, False, {}
    
    # Looks scary but honestly its just because of the fancy styling
    def render(self):
        if self.render_mode not in self.metadata["render_modes"]:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()
        
        coord = lambda x, y: (int(x + 200), int(200 - y))
        canvas = pygame.Surface((400, 400))
        canvas.fill((25, 25, 30))  # Dark industrial background

        # Draw subtle grid
        grid_color = (40, 40, 45)
        for i in range(-200, 201, 40):
            pygame.draw.line(canvas, grid_color, coord(i, -200), coord(i, 200), 1)
            pygame.draw.line(canvas, grid_color, coord(-200, i), coord(200, i), 1)

        # Draw pulsating target
        target_pos = coord(*self.target)
        glow_intensity = abs(pygame.time.get_ticks() % 1000 - 500) / 500  # 0-1 pulse
        pygame.draw.circle(canvas, (255, 0, 0), target_pos, 13 + int(3 * glow_intensity))
        pygame.draw.circle(canvas, (255, 100, 100), target_pos, 10, 2)

        # Calculate joint positions
        shoulder, elbow, wrist = self._calculate_positions()
        joints = [shoulder, elbow, wrist]

        # Draw armored cabling between segments
        for i, (a, b) in enumerate([(shoulder, elbow), (elbow, wrist)]):
            start = coord(*a)
            end = coord(*b)
            
            # Core hydraulic line
            pygame.draw.line(canvas, (80, 80, 90), start, end, 9)
            
            # Armored segment casing
            length = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5
            angle = math.degrees(math.atan2(end[1]-start[1], end[0]-start[0]))
            
            # Create armored segment with battle damage
            segment = pygame.Surface((length, 20), pygame.SRCALPHA)
            pygame.draw.rect(segment, (50, 50, 55), (0, 0, length, 20), border_radius=8)
            pygame.draw.rect(segment, (100, 100, 110), (0, 0, length, 20), 2, border_radius=8)
            
            # Add hazard stripes near joints
            if i == 0:  # Shoulder-to-elbow gets warning stripes
                pygame.draw.rect(segment, (255, 180, 0), (length-30, 0, 30, 20), border_radius=8)
                pygame.draw.rect(segment, (0, 0, 0), (length-30, 0, 30, 20), 2, border_radius=8)
            
            # Add panel details
            for x in range(20, int(length)-20, 40):
                pygame.draw.line(segment, (70, 70, 80), (x, 5), (x+20, 5), 2)
                pygame.draw.line(segment, (30, 30, 35), (x+10, 10), (x+10, 15), 2)
            
            # Rotate and position armor over core line
            rotated = pygame.transform.rotate(segment, -angle)
            canvas.blit(rotated, (start[0] + (end[0]-start[0])/2 - rotated.get_width()/2,
                                start[1] + (end[1]-start[1])/2 - rotated.get_height()/2))

        # Draw reinforced joints with glowing cores
        for joint in joints:
            pos = coord(*joint)
            # Heavy duty joint base
            pygame.draw.circle(canvas, (60, 60, 65), pos, 18)
            pygame.draw.circle(canvas, (30, 30, 35), pos, 18, 3)
            
            # Glowing core
            pygame.draw.circle(canvas, (100, 100, 255), pos, 8)
            pygame.draw.circle(canvas, (200, 200, 255), pos, 3)
            
            # Bolted reinforcement
            for angle in range(0, 360, 60):
                bolt_pos = (pos[0] + 12 * math.cos(math.radians(angle)),
                            pos[1] + 12 * math.sin(math.radians(angle)))
                pygame.draw.circle(canvas, (80, 80, 90), bolt_pos, 3)

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.pixels3d(canvas), (1, 0, 2))
    
    def close(self):
        if self.screen is not None:
            pygame.quit()

# Interact with the environment using keyboard
if __name__ == "__main__":
    env = RobotArmEnv(render_mode="human")
    observation, _ = env.reset()

    # Normally a PyGame window would automatically be created with the first render call, 
    # but we need to create it manually so we can handle keyboard events
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    running = True
    while running:
        action = 4  # Default to no operation
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            action = 0  # Increase a1
        elif keys[pygame.K_LEFT]:
            action = 1  # Decrease a1
        if keys[pygame.K_UP]:
            action = 2  # Increase a2
        elif keys[pygame.K_DOWN]:
            action = 3  # Decrease a2

        # Perform the action
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated)

        # Render the environment
        env.render()
        
        # Reset if episode ends
        if terminated or truncated:
            observation, _ = env.reset()

    env.close()
    pygame.quit()

        