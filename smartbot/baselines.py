from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from swocgym import SwocGym
from pathlib import Path

if __name__ == "__main__":
     SaveFile = Path('baselines_model')

     env = SubprocVecEnv([(lambda i=i: SwocGym(i+1, i)) for i in range(16)])

     model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs={'layers': [2024,1024,1024,1024,512]})
     if SaveFile.exists():
          print('loading...')
          model.load_parameters(SaveFile)
     model.learn(total_timesteps=20000)
     model.save(SaveFile)
     model.learn(total_timesteps=20000)
     model.save(SaveFile)

     #demo
     obs = env.reset()
     for i in range(1000):
          action, _states = model.predict(obs)
          obs, rewards, dones, info = env.step(action)
          env.render()