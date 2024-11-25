import numpy as np
import gymnasium as gym
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv
from PIL import Image, ImageEnhance
import cv2

class OpticalIllusionPushCube(gym.Env):
    def __init__(self, illusion_type='checker_shadow'):
        # Initialize the base PushCube environment from ManiSkill
        self.env = gym.make('PushCube-v1', render_mode='rgb_array')  # Replace with the exact environment ID in ManiSkill
        self.illusion_type = illusion_type
    
    def reset(self):
        obs = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def capture_and_save_frame(self, filename="first_frame.png"):
        """
        Capture and save the first rendered frame of the environment.
        
        :param filename: Name of the file to save the image as.
        """
        frame = self.env.render()
        
        # Convert tensor to numpy if necessary, then remove extra dimensions
        frame = frame.squeeze().numpy()
        frame = np.squeeze(frame)
            
        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (255 * frame).astype(np.uint8)
        
        # Apply post-processing illusion effects
        processed_frame = self.apply_illusion_effect(frame)
        
        # Save the processed frame
        image = Image.fromarray(processed_frame)
        image.save(filename)
        print(f"Frame with illusion saved as {filename}")

    def apply_illusion_effect(self, frame):
        """
        Apply the selected optical illusion effect to the given frame.
        
        :param frame: Numpy array of the rendered frame.
        :return: Processed frame with optical illusion effect applied.
        """
        if self.illusion_type == 'checker_shadow':
            return self.apply_checker_shadow_illusion(frame)
        elif self.illusion_type == 'perspective_illusion':
            return self.apply_perspective_illusion(frame)
        elif self.illusion_type == 'depth_illusion':
            return self.apply_depth_illusion(frame)
        return frame  # If no effect specified, return the original frame

    def apply_checker_shadow_illusion(self, frame):
        """Apply a checker shadow illusion by simulating a shadow gradient on the image."""
        # Convert frame to PIL image for easy manipulation
        image = Image.fromarray(frame)
        
        # Create a shadow gradient overlay
        width, height = image.size
        shadow_overlay = Image.new('L', (width, height), color=0)
        for y in range(height):
            gradient_value = int(255 * (y / height))  # Shadow gradient
            cv2.line(np.array(shadow_overlay), (0, y), (width, y), gradient_value, 1)
        
        # Combine shadow overlay with the image
        image = ImageEnhance.Brightness(image).enhance(0.6)  # Darken the image slightly
        shadow_overlay = shadow_overlay.convert("RGB")
        return cv2.addWeighted(np.array(image), 1.0, np.array(shadow_overlay), 0.5, 0)

    def apply_perspective_illusion(self, frame):
        """Apply a forced perspective effect by warping the image."""
        height, width, _ = frame.shape
        src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
        dst_points = np.float32([[width * 0.1, height * 0.33], [width * 0.9, height * 0.2], 
                                 [width * 0.2, height * 0.9], [width * 0.8, height * 0.7]])
        
        # Apply perspective warp
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_frame = cv2.warpPerspective(frame, matrix, (width, height))
        return warped_frame

    def apply_depth_illusion(self, frame):
        """Apply a depth illusion by scaling certain regions to make them appear closer or further."""
        height, width, _ = frame.shape
        # Select a central region to scale
        center_region = frame[height//4:3*height//4, width//4:3*width//4]
        scaled_center = cv2.resize(center_region, (width, height))
        
        # Blend scaled center with original to create the illusion of depth
        depth_illusion_frame = cv2.addWeighted(frame, 0.4, scaled_center, 0.6, 0)
        return depth_illusion_frame

# Usage example
if __name__ == "__main__":
    # Choose the type of optical illusion
    illusion_type = 'depth_illusion'  # Options: 'checker_shadow', 'perspective_illusion', 'depth_illusion'
    
    # Create the environment with the chosen illusion type
    env = OpticalIllusionPushCube(illusion_type=illusion_type)
    obs = env.reset()
    
    # Capture and save the first frame with the illusion effect
    env.capture_and_save_frame("first_frame_with_illusion.png")
    
    # Sample random actions to see the effect in the environment
    #for _ in range(10):
    #    action = env.env.action_space.sample()  # Get a random action from the action space
    #    obs, reward, done, info = env.step(action)
    #    if done:
    #       obs = env.reset()
