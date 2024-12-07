import sapien.core as sapien
from sapien.core import Pose
import tensorflow as tf
import random
import matplotlib.pyplot as plt

class SceneManager:
    def __init__(self, agent):
        self.env = gym.make("PushCube-v1", obs_mode='rgbd', control_mode='pd_ee_delta_pos', render_mode="rgb_array")
        self.tmpEnv = gym.make("PushCube-v1", obs_mode='state_dict', control_mode='pd_ee_delta_pos', render_mode="rgb_array")
        self.env = env = RecordEpisode(self.env, output_dir='push/', save_trajectory=False, save_video=True, video_fps=30, max_steps_per_video=10000)
        self.scene = self.env.scene
        self.agent = agent

    def add_new_box_to_scene(self):
        # Create an actor builder for the new box
        builder = self.scene.create_actor_builder()

        # Define the box's size (half extents) and pose
        box_half_extents = [0.02, 0.02, 0.02]  # Half size in each dimension
        box_pose = Pose([0.1, 0.1, 0.01])  # (x, y, z)

        # Add a box geometry to the builder
        builder.add_box_collision(half_size=box_half_extents)  # For collision
        builder.add_box_visual(
            half_size=box_half_extents,
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.2]),
        )  # For rendering

        # Build the actor and add it to the scene
        new_box = builder.build_static(name="new_box")
        new_box.set_pose(box_pose)  # Set the position of the box

        print("New box added to the scene!")

    def add_dim_circles_to_scene(self):
        # Circle 1
        builder1 = self.scene.create_actor_builder()
        circle1_radius = 0.1
        circle1_pose = Pose([0.1, -0.3, 0.0])  # Slightly raised for visibility
        builder1.add_capsule_visual(
            radius=circle1_radius,
            half_length=0,
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.5]),
        )  # Gray and dim
        circle1 = builder1.build_static(name="dim_circle_1")
        circle1.set_pose(circle1_pose)

        # Circle 2
        builder2 = self.scene.create_actor_builder()
        circle2_radius = 0.1
        circle2_pose = Pose([-0.1, 0.1, 0.0])  # Slightly raised for visibility
        builder2.add_capsule_visual(
            radius=circle2_radius,
            half_length=0,
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.5]),
        )  # Gray and dim
        circle2 = builder2.build_static(name="dim_circle_2")
        circle2.set_pose(circle2_pose)

        print("Dim circles added to the scene!")

    def add_simulation(self, step=10000, type=None):
        counter_false = 0
        counter_false_with_obj = 0
        counter_true = 0
        if type=="obj":
            self.add_new_box_to_scene(self.scene)
            self.env.scene.step()
            self.env.scene.update_render()
            for n in range(step):
                done = False
                timestep = 100
                seed = random.randint(1,step)
                obs, info = self.env.reset(seed=seed)
                tmpObs, info = self.tmpEnv.reset(seed=seed)
                while (done != True) or (timestep < 100):
                    action = self.agent.get_action(obs)
                    next_obs, reward, done, truncated, info = self.env.step(action)
                    tmp_next_obs,_,_,_,_ = self.tmpEnv.step(action)
                    obs = next_obs
                current_grip = tmpObs['extra']['tcp_pose'][0,:3]
                delta_2 = current_grip - torch.tensor([0.1, 0.1, 0.01])
                delta_1 = current_grip - tmpObs['extra']['obj_pose'][0,:3]
                if (done != True):
                    if (delta_1 > delta_2):
                        if (tmp_next_obs['extra']['obj_pose'][0,:3] == tmpObs['extra']['obj_pose'][0,:3]):
                            counter_false += 1
                            counter_false_with_obj += 1
                        else:
                            counter_false += 1
                    else:
                        counter_false += 1
                else:
                    counter_true += 1
            return counter_false/step, counter_false_with_obj/step, counter_true/sterp
        else:
            self.add_dim_circles_to_scene(self.scene)
            self.env.scene.step()
            self.env.scene.update_render()
            for n in range(step):
                done = False
                timestep = 100
                seed = random.randint(1,step)
                obs, info = self.env.reset(seed=seed)
                tmpObs, info = self.tmpEnv.reset(seed=seed)
                while (done != True) or (timestep < 100):
                    action = self.agent.get_action(obs)
                    next_obs, reward, done, truncated, info = self.env.step(action)
                    tmp_next_obs,_,_,_,_ = self.tmpEnv.step(action)
                    obs = next_obs
                current_grip = tmpObs['extra']['tcp_pose'][0,:3]
                delta_2 = current_grip - tf.convert_to_tensor([0.1, -0.3, 0.0])
                delta_3 = current_grip - tf.convert_to_tensor([-0.1, 0.1, 0.0])
                delta_1 = current_grip - tmpObs['extra']['obj_pose'][0,:3]
                if (done != True):
                    if (delta_1 > delta_2) or (delta_1 > delta_3):
                        counter_false += 1
                        counter_false_with_obj += 1
                    else:
                        counter_false += 1
                else:
                    counter_true += 1
            return counter_false/step, counter_false_with_obj/step, counter_true/step
        
def plot_result(algo, l1, l2):
    plt.figure(figsize=(10, 6))
    plt.scatter(l1, l2, color='blue', label='Success Rate')  # Scatter plot
    plt.plot(l1, l2, color='blue', linestyle='-', linewidth=2)  # Line connecting dots

    # Title and Labels
    plt.title(f"{algo} Task Success Rate Over Total Number of Timesteps", fontsize=14)
    plt.xlabel("Total Timesteps", fontsize=12)
    plt.ylabel("Task Success Rate (%)", fontsize=12)

    # Grid and Legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Display the plot
    plt.show()

class AgentLoader:
    def __init__(self, model_checkpoint_path):
        self.model_checkpoint_path = model_checkpoint_path
        self.agent = None

    def load_agent(self):
        # Load the TensorFlow model from the checkpoint
        try:
            self.agent = tf.keras.models.load_model(self.model_checkpoint_path)
            print("Agent loaded successfully from checkpoint!")
        except Exception as e:
            print(f"Failed to load agent: {e}")

    def get_agent(self):
        return self.agent
def plot_task_outcome(timesteps, success_rate, unsuccessful_agent, unsuccessful_illusion, title="Task Outcome Distribution"):
    # Bar positions
    x = np.arange(len(timesteps))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(x, success_rate, label="Success Rate", color="green")
    plt.bar(x, unsuccessful_agent, bottom=success_rate, label="Unsuccessful (Agent)", color="red")
    plt.bar(x, unsuccessful_illusion, 
            bottom=[s + u for s, u in zip(success_rate, unsuccessful_agent)],
            label="Unsuccessful (Illusion)", color="blue")
    
    # Labels and Title
    plt.xticks(x, timesteps, rotation=45)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title(title, fontsize=14)
    
    # Legend
    plt.legend()
    
    # Grid and Display
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    #checkpoints = [
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733073763/final_ckpt.pt",
    #   "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733073795/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733073814/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733073857/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733073899/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733073919/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733074892/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733074995/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733075214/final_ckpt.pt",
    #   "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733075336/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076446/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076475/final_ckpt.pt",
    #   "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076526/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076576/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076598/final_ckpt.pt",
    #   "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076649/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076691/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076713/final_ckpt.pt",
    #    "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076792/final_ckpt.pt",
    #   "/home/szq2sj/ManiSkill-main/examples/baselines/ppo/runs/PushCube-v1__ppo_rgb__1__1733076874/final_ckpt.pt",
    #]
    #types = ["obj", "target"]
    #counter_false = []
    #ounter_false_2 = []
    #counter_true = []
    #for type in types:
    #    for ck in checkpoints:
    #        agent = AgentLoader(ck)
    #        agent.load_agent()
    #        manager = SceneManager(agent)
    #        counter_1, counter_2, counter_3 = manager.add_simulation(type=type)
    #        counter_false.append(counter_1*100)
    #        counter_false_2.append(counter_2*100)
    #        counter_true.append(counter_3*100)

    #     counter_false_ppo = counter_false[:10]
    #     counter_false_2_ppo = counter_false_2[:10]
    #     counter_true_ppo = counter_true[:10]
    #     counter_false_sac = counter_false[10:]
    #     counter_false_2_sac = counter_false_2[10:]
    #     counter_true_sac = counter_true[10:]
    #     timesteps = [f"{i}k" for i in range(100, 1100, 100)]
    #     plot_task_outcome(timesteps, counter_true_ppo, counter_false_ppo - counter_false_2_ppo, counter_false_2_ppo,title="Task Outcome Distribution Across Timesteps for PPO")
    #     plot_task_outcome(timesteps, counter_true_sac, counter_false_sac - counter_false_2_sac, counter_false_2_sac,title="Task Outcome Distribution Across Timesteps for SAC")


