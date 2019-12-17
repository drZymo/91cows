from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
#from swocgymgrid import SwocGym
from mazegym import MazeGym
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
     #update = locals['update']
     #if update > 0 and update % SaveEvery == 0:
     #     print('saving...', end='')
     #     locals['self'].save(SaveFile)
     #     print('saved!')

     return True

def main():
#     env = SubprocVecEnv([(lambda i=i: SwocGym(i+1, GameServicePath, i, fieldWidth=10, fieldHeight=10)) for i in range(16)])
     env = SubprocVecEnv([(lambda i=i: MazeGym(mazeWidth=10, mazeHeight=10, nrWallsToRemove=60)) for i in range(12)])
     try:
          #model = PPO2("MlpPolicy", env, verbose=1, ent_coef=0.01, tensorboard_log='/home/ralph/swoc2019/log')
          #model = PPO2("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [1024,1024,512,512,256,256,128,128,64,64], 'act_fun': tf.nn.relu},
          #               n_steps=64, ent_coef=0.01, learning_rate=1e-5, tensorboard_log='/home/ralph/swoc2019/log')
          model = PPO2("MlpPolicy", env, verbose=1, policy_kwargs={'net_arch': [1024,1024,512,512,256,256,128,128,64,64], 'act_fun': tf.nn.relu})
          if SaveFile.exists():
               print('loading...', end='')
               model.load_parameters(SaveFile)
               print('loaded!')
          else:
               # No weights loaded, so remove history
               with open(RewardsLog, 'w+') as file:
                    file.write('')

          try:
               print('learning...')
               model.learn(total_timesteps=100000000, callback=callback)
          finally:
               print('saving...', end='')
               model.save(SaveFile)
               print('saved!')

     except KeyboardInterrupt:
          print('closing...', end='')
     finally:
          env.close()
     print('closed!')

if __name__ == "__main__":
     main()
