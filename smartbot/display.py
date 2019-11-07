from PIL import Image, ImageDraw
import socket
import json
import numpy as np

buffer = ''

def GetGameState():
    global buffer
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 9735))

        newLinePos = buffer.find('\n')
        while newLinePos < 0:
            data = s.recv(16384).decode('utf-8')
            buffer += data
            newLinePos = buffer.find('\n')

        line = buffer[:newLinePos]
        buffer = buffer[newLinePos+1:]
        gameState = json.loads(line)

    return gameState

def GetField(gameState):
    return np.array([
        [(int(entry['type']), int(entry['orientation']) // 90) for entry in row]
        for row in gameState['data']])


walls = {
    #   t,r,b,l
    0: np.array([0,0,0,0]),
    1: np.array([0,0,0,1]),
    2: np.array([0,1,0,1]),
    3: np.array([1,0,0,1]),
    4: np.array([1,1,0,1]),
}


def DrawImage(gameState, width, height):
    im = Image.new('L', (width, height + 20), color=0)

    draw = ImageDraw.Draw(im)

    def DrawCell(x, y, cell):
        cellType, cellOrient = cell
        l, t = x * 80, height - (y+1) * 80
        r, b = (x+1) * 80 - 1, height - y * 80 - 1

        draw.rectangle((l + 2, t + 2, r - 2, b - 2), fill=None, outline=32)
        wall = walls[cellType]

        while cellOrient > 0:
            wall = wall[[1,2,3,0]]
            cellOrient -= 1
        
        if wall[0]:
            draw.rectangle((l, t, r, t-2), fill=128, outline=None)
        if wall[1]:
            draw.rectangle((r-2, t, r, b), fill=128, outline=None)
        if wall[2]:
            draw.rectangle((l, b+2, r, b), fill=128, outline=None)
        if wall[3]:
            draw.rectangle((l, t, l+2, b), fill=128, outline=None)

    def DrawField(field):
        nrRows, nrColumns = field.shape[0], field.shape[1]

        cellWidth = width / nrColumns
        cellHeight = height / nrRows

        for y,row in enumerate(field):
            for x,cell in enumerate(row):
                DrawCell(x, y, cell)

    def DrawBot(bot):
        position = np.array(bot['position'])
        forward = np.array(bot['forward'])
        right = np.array(bot['right'])

        rightForward = position + forward + right
        leftForward = position + forward - right
        rightBackward = position - forward + right
        leftBackward = position - forward - right

        lines = np.array([
            [leftForward, rightForward],
            [leftBackward, rightBackward],
            [rightBackward, rightForward],
            [leftForward, leftBackward],
            [leftBackward, position + forward],
            [rightBackward, position + forward]
        ])

        for line in lines:
            xy = line * [width, height]
            draw.line(xy.flatten().tolist(), fill=255)

    gameTick = gameState['gameTick']
    draw.text((10, height+5), f'tick: {gameTick}', fill=255)

    field = GetField(gameState)

    DrawField(field)

    for bot in gameState['bots']:
        DrawBot(bot)

    del draw
    return im.tobytes()

import pyglet
from pyglet.window import mouse

width, height = 800, 800
window = pyglet.window.Window(width=width, height=height)

gameState = GetGameState()
rawImage = DrawImage(gameState, width, height)
image = pyglet.image.ImageData(width, height, 'L', rawImage, pitch=-width)

@window.event
def on_draw():
    image.blit(0, 0, 0)

def update(dt):
    gameState = GetGameState()
    rawImage = DrawImage(gameState, width, height)
    image.set_data(fmt='L', data=rawImage, pitch=-width)

pyglet.clock.schedule_interval(update, 0.1)

pyglet.app.run()
