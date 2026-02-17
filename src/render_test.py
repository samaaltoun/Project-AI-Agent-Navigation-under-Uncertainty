import gymnasium as gym
import imageio

def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, info = env.reset(seed=0)

    frames = []
    for _ in range(250):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()          # returns an RGB image (H, W, 3)
        frames.append(frame)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    imageio.mimsave("cartpole_test.gif", frames, fps=30)
    print("Saved cartpole_test.gif ✅")

if __name__ == "__main__":
    main()
