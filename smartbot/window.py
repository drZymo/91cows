import pyglet
from pyglet.window import key

class BotWindow(object):

    def __init__(self, rawImage):
        width, height = rawImage.size
        self.window = pyglet.window.Window(width=width, height=height)
        self.image = pyglet.image.ImageData(width, height, 'RGBA', rawImage.tobytes(), pitch=-width*4)
        pyglet.app.run()

    def update(self, rawImage):
        self.image.set_data(format='RGBA', data=rawImage.tobytes(), pitch=-width*4)
        self.image.blit(0, 0)



