from swocgym import SwocGym
from draw import DrawObservation
import pyglet
from pyglet.window import key

GameServicePath = '../../build/GameService/GameService'

env = SwocGym(1, GameServicePath, 0, actionRepeat=4, fieldWidth=5, fieldHeight=5, oneTarget=True)

obs, reward, done  = env.reset(), 0, False
rawImage = env.render('rgb_array', 800, 800)
width, height, _ = rawImage.shape


window = pyglet.window.Window(width=width, height=height)
image = pyglet.image.ImageData(width, height, 'RGB', rawImage.tobytes(), pitch=-width*3)

@window.event
def on_draw():
    image.blit(0, 0)

def update(dt):
    global rawImage
    image.set_data(format='RGB', data=rawImage.tobytes(), pitch=-width*3)

@window.event
def on_key_press(symbol, modifiers):
    global rawImage

    reward, done = 0, False
    if symbol == key.UP:
        obs, reward, done, info = env.step(0)
        print(obs[-5:])
    elif symbol == key.RIGHT:
        obs, reward, done, info = env.step(1)
        print(obs[-5:])
    elif symbol == key.LEFT:
        obs, reward, done, info = env.step(2)
        print(obs[-5:])

    if reward != 0:
        print(f'reward: {reward}')
    
    if done:
        print('done')
        obs, reward, done  = env.reset(), 0, False

    rawImage = env.render('rgb_array', width, height)

pyglet.clock.schedule_interval(update, 0.05)
pyglet.app.run()

env.close()
