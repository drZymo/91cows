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

def callback(locals, globals):
     # log the total reward of this batch
     batchReward = np.sum(locals['true_reward'])
     print(f'batchReward: {batchReward:.1f}')
     with open(RewardsLog, 'a+') as file:
          file.write(f'{int(batchReward)}\n')

     # save the model every N updates
     update = locals['update']
     if update > 0 and update % SaveEvery == 0:
          print('saving...')
          locals['self'].save(SaveFile)

     return True

def main():
     env = SubprocVecEnv([(lambda i=i: SwocGym(i+1, GameServicePath, i, actionRepeat=4, oneTarget=True)) for i in range(16)])
     try:
          model = PPO2("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [256,256,256,128,128,128], 'act_fun': tf.nn.relu},
                         n_steps=32, ent_coef=0.1, learning_rate=1e-4, tensorboard_log='/home/ralph/swoc2019/log')
          if SaveFile.exists():
               print('loading...')
               model.load_parameters(SaveFile)
          else:
               # No weights loaded, so remove history
               with open(RewardsLog, 'w+') as file:
                    file.write('')

          try:
               print('learning...')
               model.learn(total_timesteps=100000000, callback=callback)
          finally:
               print('saving...')
               model.save(SaveFile)
               print('saved!')

     except KeyboardInterrupt:
          print('closing...')
     finally:
          env.close()
     print('closed')

if __name__ == "__main__":
     main()
