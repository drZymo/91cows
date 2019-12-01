from swoc1 import SwocEnv
from draw import DrawObservation
import pyglet
from pyglet.window import key

GameServicePath = '../../build/GameService/GameService'

env = SwocEnv(1, GameServicePath)

width, height = 800, 800
window = pyglet.window.Window(width=width, height=height)

obs, totalReward, done  = env.reset(10, 10), 0, False
rawImage = DrawObservation(obs, width, height)

image = pyglet.image.ImageData(width, height, 'L', rawImage.tobytes(), pitch=-width)

@window.event
def on_draw():
    image.blit(0, 0, 0)

def update(dt):
    rawImage = DrawObservation(obs, width, height)
    image.set_data(format='L', data=rawImage.tobytes(), pitch=-width)

@window.event
def on_key_press(symbol, modifiers):
    global obs, reward, done, totalReward, index

    for _ in range(4):
        reward = 0
        if symbol == key.UP:
            obs, reward, done = env.step(0)
        elif symbol == key.RIGHT:
            obs, reward, done = env.step(1)
        elif symbol == key.LEFT:
            obs, reward, done = env.step(2)

        if reward != 0:
            print(f'reward: {reward}')
        totalReward += reward

        if reward < 0 or done:
            break

    if done:
        obs, totalReward, done  = env.reset(10, 10), 0, False

pyglet.clock.schedule_interval(update, 0.05)

pyglet.app.run()
