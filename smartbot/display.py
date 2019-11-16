from PIL import Image, ImageDraw
import socket
import json
import numpy as np

from swoc import Observer



def DrawImage(obs, width, height):
    field, bots, scores, gameTick = obs

    im = Image.new('L', (width, height + 20), color=0)

    draw = ImageDraw.Draw(im)

    def DrawCell(x, y, cell):
        l, t = x * 80, height - (y+1) * 80
        r, b = (x+1) * 80 - 1, height - y * 80 - 1
        cx, cy = (l+r)/2, (b+t)/2

        draw.rectangle((l + 2, t + 2, r - 2, b - 2), fill=None, outline=32)

        if cell[0]:
            draw.rectangle((l, t, r, t-2), fill=192, outline=None)
        if cell[1]:
            draw.rectangle((r-2, t, r, b), fill=192, outline=None)
        if cell[2]:
            draw.rectangle((l, b+2, r, b), fill=192, outline=None)
        if cell[3]:
            draw.rectangle((l, t, l+2, b), fill=192, outline=None)

        if cell[4]:
            draw.ellipse((l+30,t+30,r-30,b-30), fill=128, outline=192)
        if cell[5]:
            draw.rectangle((l+30,t+30,r-30,b-30), fill=96, outline=128)
        if cell[6]:
            draw.rectangle((l+30,t+30,r-30,b-30), fill=None, outline=128)
        if cell[7]:
            draw.rectangle((l+30,t+30,r-30,b-30), fill=96, outline=128)
        if cell[8]:
            draw.polygon((l+30,b-30,l+35,t+30,l+40,b-30,l+45,t+30,l+50,b-30), fill=96, outline=128)
        if cell[9]:
            draw.ellipse((l+35,t+30,r-35,b-30), fill=96, outline=128)
        if cell[10]:
            draw.ellipse((l+35,t+30,r-35,b-30), fill=None, outline=128)


    def DrawField(field):
        nrRows, nrColumns = field.shape[0], field.shape[1]

        cellWidth = width / nrColumns
        cellHeight = height / nrRows

        for y,row in enumerate(field):
            for x,cell in enumerate(row):
                DrawCell(x, y, cell)

    def DrawBot(bot):
        if not bot[0]: return

        position = bot[1:3]
        orientation = bot[3]
        
        c, s = np.cos(orientation), np.sin(orientation)
        R = np.array(((c, -s), (s, c)))
        forward = np.matmul(R, [0.025, 0.0])
        right = np.matmul(R, [0.0, 0.025])

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


    draw.text((10, height+5), f'tick: {gameTick}', fill=255)

    DrawField(field)

    for bot in bots:
        DrawBot(bot)

    del draw
    return im.tobytes()

import pyglet
from pyglet.window import mouse

observer = Observer()

width, height = 800, 800
window = pyglet.window.Window(width=width, height=height)

rawImage = DrawImage(observer.getObservation(), width, height)

image = pyglet.image.ImageData(width, height, 'L', rawImage, pitch=-width)

@window.event
def on_draw():
    image.blit(0, 0, 0)

def update(dt):
    rawImage = DrawImage(observer.getObservation(), width, height)
    image.set_data(fmt='L', data=rawImage, pitch=-width)

pyglet.clock.schedule_interval(update, 0.05)

pyglet.app.run()
