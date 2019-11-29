from swoc1 import SwocEnv
from draw import DrawObservation
import pyglet
from pyglet.window import key
import numpy as np

width, height = 800, 800
botId = 1
index = 0

ActionNames = {None: '?', 0: 'forward', 1: 'right', 2: 'left'}
def readObs():
    global prevObs
    try:
        obs = np.load(f'/home/ralph/swoc2019/episode/{botId}-{index}-obs.npy')
        fieldObs, botObs = obs[:1100], obs[1100:]
        prevObs = fieldObs.reshape(10,10,11), botObs
    except:
        pass
    return prevObs

def readAct():
    try:
        return np.load(f'/home/ralph/swoc2019/episode/{botId}-{index}-act.npy')
    except:
        return None, None, None

observation = readObs()
action, reward, done = readAct()
rawImage = DrawObservation(observation, width, height)

window = pyglet.window.Window(width=width, height=height+30)
image = pyglet.image.ImageData(width, height, format='L', data=rawImage.tobytes(), pitch=-width)
indexLabel = pyglet.text.Label(f'#{botId} - {index}', x=0, y=height+5)
actionLabel = pyglet.text.Label(f'Action: ?', x=100, y=height+5)
rewardLabel = pyglet.text.Label(f'Reward: ?', x=300, y=height+5)
doneLabel = pyglet.text.Label(f'Done: ?', x=500, y=height+5)

@window.event
def on_draw():
    window.clear()
    image.blit(0, 0)
    indexLabel.draw()
    actionLabel.draw()
    rewardLabel.draw()
    doneLabel.draw()

def update(dt):
    rawImage = DrawObservation(observation, width, height)
    image.set_data(format='L', data=rawImage.tobytes(), pitch=-width)
    indexLabel.text = f'#{botId} - {index}'
    actionLabel.text = f'Action: {ActionNames[action]}'
    rewardLabel.text = f'Reward: {reward}'
    doneLabel.text = f'Done: {"yes" if done else "no"}'

@window.event
def on_key_press(symbol, modifiers):
    global botId, index, observation, action, reward, done
    if symbol == key.RIGHT: index += 1
    elif symbol == key.LEFT: index -= 1
    elif symbol == key.UP: botId += 1
    elif symbol == key.DOWN: botId -= 1
    index = np.clip(index, 0, 1000)
    botId = np.clip(botId, 1, 16)
    
    observation = readObs()
    action, reward, done = readAct()

pyglet.clock.schedule_interval(update, 0.05)
pyglet.app.run()
