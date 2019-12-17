from swoc import Observer
from draw import DrawObservation
import pyglet
from pyglet.window import mouse

observer = Observer()

def GetObservation():
    field, bots, _, _ = observer.getObservation()
    return field, bots

width, height = 800, 800
window = pyglet.window.Window(width=width, height=height)
rawImage = DrawObservation(GetObservation(), width, height)

image = pyglet.image.ImageData(width, height, 'L', rawImage.tobytes(), pitch=-width)

@window.event
def on_draw():
    image.blit(0, 0, 0)

def update(dt):
    rawImage = DrawObservation(GetObservation(), width, height)
    image.set_data(fmt='L', data=rawImage.tobytes(), pitch=-width)

pyglet.clock.schedule_interval(update, 0.05)

pyglet.app.run()
