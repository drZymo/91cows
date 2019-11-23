from gym.utils import seeding
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from swocgym import SwocGym

swocEnv = SwocGym()
env = DummyVecEnv([lambda: swocEnv])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs={'layers': [2024,1024,1024,1024,512]})
# model.learn(total_timesteps=1000000)

#demo
obs = env.reset()
for i in range(1000):
     action, _states = model.predict(obs)
     obs, rewards, dones, info = env.step(action)
     env.render()