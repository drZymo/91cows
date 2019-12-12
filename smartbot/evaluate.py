from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from swocgym import SwocGym
from pathlib import Path
import numpy as np

# Hide warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

SaveFile = Path('saved_model.zip')
GameServicePath = Path.cwd()/'..'/'..'/'build'/'GameService'/'GameService'

def evaluate(env, model):
     obs = env.reset()
     totalRewards = None
     for i in range(1000):
          action, _states = model.predict(obs)
          obs, rewards, dones, info = env.step(action)
          totalRewards = totalRewards + rewards if totalRewards is not None else rewards
          env.render()
     return totalRewards

def main():
     env = SubprocVecEnv([(lambda i=i: SwocGym(i+1, GameServicePath, i, actionRepeat=4, oneTarget=True)) for i in range(4)])
     try:
          model = PPO2("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [256,256,256,128,128,128], 'act_fun': tf.nn.relu},
                         n_steps=256, ent_coef=0.0, learning_rate=1e-5)
          if SaveFile.exists():
               print('loading...')
               model.load_parameters(SaveFile)
          else:
               print('Warning: No save file loaded')

          print('evaluating...', end='')
          totalRewards = evaluate(env, model)
          print(f'mean reward: {np.mean(totalRewards)}')

     except KeyboardInterrupt:
          print('closing...')
     finally:
          env.close()
     print('closed')

if __name__ == "__main__":
     main()
