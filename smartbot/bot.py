import numpy as np
from pathlib import Path
from env import GameController, BotController, Observer
from draw import DrawObservation
from window import BotWindow
import pyglet
from time import sleep
import matplotlib.pyplot as plt
from PIL import Image

BotId = 1
GameServicePath = Path.cwd()/'..'/'..'/'build'/'GameService'/'GameService'
Hostname = 'localhost'
WindowWidth, WindowHeight = 1200, 600

FieldWidth, FieldHeight = 5, 5

EMPTY = 0
WALL = 1
BOT = 2
ITEM = 3

def getObs(observer):
    # get an observation
    field, bots, scores, gameTick = observer.getObservation()
    bot = bots[BotId]
    score = scores[BotId]
    return (field, bot)


def initWindow(fieldImg, gridImg):
    global window, windowImageField, windowImageGrid
    window = pyglet.window.Window(width=WindowWidth, height=WindowHeight)
    width, height, pitch = fieldImg.shape
    windowImageField = pyglet.image.ImageData(width, height, 'RGB', fieldImg.tobytes(), pitch=width*pitch)
    width, height, pitch = gridImg.shape
    windowImageGrid = pyglet.image.ImageData(width, height, 'RGB', gridImg.tobytes(), pitch=-width*pitch)


def updateWindow(fieldImg, gridImg):
    global window, windowImageField, windowImageGrid
    width, height, pitch = fieldImg.shape
    windowImageField.set_data(format='RGB', data=fieldImg.tobytes(), pitch=width*pitch)
    windowImageField.blit(0, 0)

    width, height, pitch = gridImg.shape
    windowImageGrid.set_data(format='RGBA', data=gridImg.tobytes(), pitch=-width*pitch)
    windowImageGrid.blit(WindowWidth//2, 0)

    pyglet.clock.tick()

    for window in pyglet.app.windows:
        window.switch_to()
        window.dispatch_events()
        window.dispatch_event('on_draw')
        window.flip()


def drawField(fieldObs, botObs):
    img = np.array(DrawObservation((fieldObs, botObs), WindowWidth//2, WindowHeight))
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    return img


def convertToGrid(fieldObs, botObs):
    fieldHeight, fieldWidth, _ = fieldObs.shape

    grid = np.full((fieldHeight*2+1, fieldWidth*2+1), WALL)

    # empty cells
    for y in range(fieldObs.shape[0]):
        cy = y*2+1
        for x in range(fieldObs.shape[1]):
            cx = x*2+1
            t,r,b,l = fieldObs[y, x, :4]
            grid[cy, cx] = EMPTY

            if not t:
                grid[cy+1, cx] = EMPTY
            if not b:
                grid[cy-1, cx] = EMPTY
            if not l:
                grid[cy, cx-1] = EMPTY
            if not r:
                grid[cy, cx+1] = EMPTY
                
            if fieldObs[y,x,4]: # coin
                grid[cy, cx] = ITEM
            elif fieldObs[y,x,5]: # treasure chest
                grid[cy, cx] = ITEM
            elif fieldObs[y,x,8]: # spike trap
                grid[cy, cx] = WALL # pretend its a wall
            else:
                # ignore: empty chest, mimic chest, bottle, test tube
                pass

    # bot
    bx = int(botObs[0]*fieldWidth*2 + 0.5)  # round to nearest int
    by = int(botObs[1]*fieldHeight*2 + 0.5)
    bx = np.clip(bx, 0, grid.shape[1]-1)
    by = np.clip(by, 0, grid.shape[0]-1)
    grid[by, bx] = BOT

    return grid


def drawGrid(grid):
    img = grid / 3
    img = np.uint8(plt.get_cmap('viridis')(img)*255)
    img = np.array(Image.fromarray(img, mode='RGBA').resize((WindowWidth//2, WindowHeight)))
    return img



def findDistances(grid, startPos):
    gridHeight, gridWidth = grid.shape
    distances = np.full((gridHeight, gridWidth), np.inf)

    y,x = startPos
    cellsToCheck = [(y,x)]
    distances[y,x] = 0

    def checkAndAdd(y, x, distance):
        if distances[y, x] > distance + 1:
            distances[y, x] = distance + 1
            cellsToCheck.append((y, x))

    while cellsToCheck:
        cellY, cellX = cellsToCheck[0]
        cellsToCheck = cellsToCheck[1:]
        
        distance = distances[cellY, cellX]

        if grid[cellY-1,cellX] != WALL:
            checkAndAdd(cellY-1, cellX, distance)
        if grid[cellY+1,cellX] != WALL:
            checkAndAdd(cellY+1, cellX, distance)
        if grid[cellY,cellX-1] != WALL:
            checkAndAdd(cellY, cellX-1, distance)
        if grid[cellY,cellX+1] != WALL:
            checkAndAdd(cellY, cellX+1, distance)
    return distances


def main():
    game = GameController(GameServicePath, Hostname)
    bot = BotController(BotId, Hostname)
    observer = Observer(Hostname)

    # Start a simulation
    game.createGame(FieldWidth, FieldHeight)
    bot.reset(FieldWidth, FieldHeight)
    game.startGame()


    fieldObs, botObs = getObs(observer)

    fieldImg = drawField(fieldObs, botObs)
    grid = convertToGrid(fieldObs, botObs)
    gridImg = drawGrid(grid)

    initWindow(fieldImg, gridImg)
    updateWindow(fieldImg, gridImg)

    try:
        while True:
            fieldObs, botObs = getObs(observer)

            fieldImg = drawField(fieldObs, botObs)
            grid = convertToGrid(fieldObs, botObs)
            gridImg = drawGrid(grid)

            botPos = np.argwhere(grid == BOT)[0]

            distances = findDistances(grid, botPos)
            
            itemPositions = np.argwhere(grid == ITEM)
            itemDistances = distances[itemPositions[:,0], itemPositions[:,1]]

            closestItemPosition = itemPositions[np.argmin(itemDistances)]


            pos = closestItemPosition

            path = []
            while grid[pos[0], pos[1]] != BOT:
                distance = distances[pos[0], pos[1]]

                if distances[pos[0]-1, pos[1]] < distance:
                    pos = (pos[0]-1, pos[1])
                elif distances[pos[0]+1, pos[1]] < distance:
                    pos = (pos[0]+1, pos[1])
                elif distances[pos[0], pos[1]-1] < distance:
                    pos = (pos[0], pos[1]-1)
                elif distances[pos[0], pos[1]+1] < distance:
                    pos = (pos[0], pos[1]+1)
                path.append(pos)

            print(list(reversed(path)))

            #print(botPos, '->', closestItemPosition)



            
            updateWindow(fieldImg, gridImg)


            

            sleep(0.2)

    finally:
        game.close()


if __name__ == "__main__":
     main()
