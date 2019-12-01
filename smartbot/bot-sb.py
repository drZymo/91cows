from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from swocgym import SwocGym
from pathlib import Path
import numpy as np

# Hide warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

RewardsLog = Path('rewards.log')
SaveFile = Path('saved_model.zip')
SaveEvery = 20
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

def callback(locals, globals):
     # log the total reward of this batch
     batchReward = np.sum(locals['true_reward'])
     print(f'batchReward: {batchReward}')
     with open(RewardsLog, 'a+') as file:
          file.write(f'{int(batchReward)}\n')

     # save the model every N updates
     update = locals['update']
     if update > 0 and update % SaveEvery == 0:
          print('saving...')
          locals['self'].save(SaveFile)

     return True

def main():
     env = SubprocVecEnv([(lambda i=i: SwocGym(i+1, GameServicePath, i)) for i in range(16)])
     try:
          model = PPO2("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [1024,512,256,128,64], 'act_fun': tf.nn.relu},
                         n_steps=512, ent_coef=0.0, learning_rate=3e-5, tensorboard_log='/home/ralph/swoc2019/log')
          if SaveFile.exists():
               print('loading...')
               model.load_parameters(SaveFile)
          else:
               # No weights loaded, so remove history
               with open(RewardsLog, 'w+') as file:
                    file.write('')

          try:
               print('learning...')
               model.learn(total_timesteps=100000000, callback=callback, reset_num_timesteps=False)

               print('evaluating...', end='')
               totalRewards = evaluate(env, model)
               print(f'mean reward: {np.mean(totalRewards)}')
          finally:
               print('saving...')
               model.save(SaveFile)
               print('saved!')

     finally:
          env.close()


if __name__ == "__main__":
     main()
