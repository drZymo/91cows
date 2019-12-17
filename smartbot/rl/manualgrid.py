#from swocgymgrid import SwocGym
from mazegym import MazeGym
import pyglet
from pyglet.window import key

#GameServicePath = '../../build/GameService/GameService'

fieldWidth, fieldHeight = 10, 10

#env = SwocGym(1, GameServicePath, 0, fieldWidth=fieldWidth, fieldHeight=fieldHeight)
env = MazeGym(mazeWidth=fieldWidth, mazeHeight=fieldHeight, nrWallsToRemove=60)
print(f'obs: {env.observation_space}')

obs, reward, done  = env.reset(), 0, False
rawImage = env.render('rgb_array', 800, 800)
width, height, _ = rawImage.shape

window = pyglet.window.Window(width=width, height=height)
image = pyglet.image.ImageData(width, height, 'RGBA', rawImage.tobytes(), pitch=-width*4)

@window.event
def on_draw():
    image.blit(0, 0)

def update(dt):
    global rawImage
    image.set_data(format='RGBA', data=rawImage.tobytes(), pitch=-width*4)

@window.event
def on_key_press(symbol, modifiers):
    global rawImage

    reward, done = 0, False
    if symbol == key.UP:
        obs, reward, done, info = env.step(0)
    elif symbol == key.RIGHT:
        obs, reward, done, info = env.step(1)
    elif symbol == key.DOWN:
        obs, reward, done, info = env.step(2)
    elif symbol == key.LEFT:
        obs, reward, done, info = env.step(3)

    if reward != 0:
        print(f'reward: {reward}')
    
    if done:
        print('done')
        obs, reward, done  = env.reset(), 0, False

    rawImage = env.render('rgb_array', 800, 800)

pyglet.clock.schedule_interval(update, 0.05)
pyglet.app.run()

env.close()
