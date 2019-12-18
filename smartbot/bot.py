import numpy as np
from pathlib import Path
from env import GameController, BotController, Observer
from draw import DrawObservation
from window import BotWindow
import pyglet
from time import sleep
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import serial

BotId = 1
#GameServicePath = Path.cwd()/'..'/'..'/'build'/'GameService'/'GameService'
Hostname = 'localhost'
WindowWidth, WindowHeight = 1200, 600

FieldWidth, FieldHeight = 8, 8
UpdateRate = 1

EMPTY = 0
WALL = 1
BOT = 2
ITEM = 4

def getObs(observer):
    # get an observation
    field, bots, scores, gameTick = observer.getObservation()
    bot = bots[BotId]
    score = scores[BotId]
    print(f'score: {score}')
    others = [bots[key] for key in bots.keys() if key != BotId]
    return (field, bot, others)


def initWindow(fieldImg, gridImg):
    global window, windowImageField, windowImageGrid
    window = pyglet.window.Window(width=WindowWidth, height=WindowHeight)
    width, height, pitch = fieldImg.shape
    windowImageField = pyglet.image.ImageData(width, height, 'RGB', fieldImg.tobytes(), pitch=-width*pitch)
    width, height, pitch = gridImg.shape
    windowImageGrid = pyglet.image.ImageData(width, height, 'RGB', gridImg.tobytes(), pitch=width*pitch)


def updateWindow(fieldImg, gridImg):
    global window, windowImageField, windowImageGrid
    width, height, pitch = fieldImg.shape
    windowImageField.set_data(format='RGB', data=fieldImg.tobytes(), pitch=-width*pitch)
    windowImageField.blit(0, 0)

    width, height, pitch = gridImg.shape
    windowImageGrid.set_data(format='RGBA', data=gridImg.tobytes(), pitch=width*pitch)
    windowImageGrid.blit(WindowWidth//2, 0)

    pyglet.clock.tick()
    
    for window in pyglet.app.windows:
        window.switch_to()
        window.dispatch_events()
        window.dispatch_event('on_draw')
        window.flip()


def drawField(fieldObs, botObs, othersObs):
    img = np.array(DrawObservation((fieldObs, botObs, othersObs), WindowWidth//2, WindowHeight))
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
    return img


def convertToGrid(fieldObs, botObs, othersObs):
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
                
            #if fieldObs[y,x,4]: # coin
            #    grid[cy, cx] = ITEM
            if fieldObs[y,x,5]: # treasure chest
                grid[cy, cx] = ITEM
            elif fieldObs[y,x,8]: # spike trap
                grid[cy, cx] = WALL # pretend its a wall
            else:
                # ignore: empty chest, mimic chest, bottle, test tube
                pass

    # other bots are walls
    for other in othersObs:
        bx = int(other[0]*fieldWidth*2 + 0.5)  # round to nearest int
        by = int(other[1]*fieldHeight*2 + 0.5)
        bx = np.clip(bx, 0, grid.shape[1]-1)
        by = np.clip(by, 0, grid.shape[0]-1)
        grid[by, bx] = WALL

    # bot
    bx = int(botObs[0]*fieldWidth*2 + 0.5)  # round to nearest int
    by = int(botObs[1]*fieldHeight*2 + 0.5)
    bx = np.clip(bx, 0, grid.shape[1]-1)
    by = np.clip(by, 0, grid.shape[0]-1)
    grid[by, bx] = BOT


    return grid


def drawGrid(grid):
    img = grid / np.max(grid)
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

def findPath(targetPos, grid, distances):
    pos = targetPos

    path = [pos]
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
        else:
            break
        path.append(pos)
    path = list(reversed(path))
    return path

def sendBotCommand(cmd):
    with serial.Serial('/dev/rfcomm0', timeout=0) as s:
        print(f'>{cmd}')

        request = cmd.encode('utf-8')
        s.write(request)

        sleep(0.2)

        #sleep(0.3)
        #response = b''
        #while True:
        #    data = s.read(100)
        #    response += data
        #    if len(data) < 100: break
        #response = response.decode('utf-8')

        #print(f'<{response}')

        #return response

distanceFactor = 1.0e-3

def moveForward(distance):
    sendBotCommand('f:2\r')
    sleep(distance * distanceFactor)
    sendBotCommand('s\r')

def moveBackward(distance):
    sendBotCommand('b:2\r')
    sleep(distance * distanceFactor)
    sendBotCommand('s\r')


def turnLeft():
    sendBotCommand('w:300\r')

def turnLeftBig():
    sendBotCommand('w:999\r')

def turnRight():
    sendBotCommand('c:300\r')

def turnRightBig():
    sendBotCommand('c:999\r')


def main():
    #game = GameController(GameServicePath, Hostname)
    #bot = BotController(BotId, Hostname)
    observer = Observer('192.168.0.100')

    # Start a simulation
    #game.createGame(FieldWidth, FieldHeight)
    #bot.reset(FieldWidth, FieldHeight)
    #game.startGame()


    fieldObs, botObs, otherObs = getObs(observer)

    grid = convertToGrid(fieldObs, botObs, otherObs)

    fieldImg = drawField(fieldObs, botObs, otherObs)
    gridImg = drawGrid(grid)

    initWindow(fieldImg, gridImg)
    updateWindow(fieldImg, gridImg)
    lastUpdate = datetime.now()

    shuffle = True

    try:
        while True:
            fieldObs, botObs, otherObs = getObs(observer)

            grid = convertToGrid(fieldObs, botObs, otherObs)

            botPos = np.argwhere(grid == BOT)[0]

            distances = findDistances(grid, botPos)
            reachablePositions = np.argwhere(~np.isinf(distances))
            reachableDistances = distances[reachablePositions[:,0], reachablePositions[:,1]]
            
            itemPositions = np.argwhere(grid == ITEM)
            itemDistances = distances[itemPositions[:,0], itemPositions[:,1]]

            closestItemPos = itemPositions[np.argmin(itemDistances)]
            farthestReachablePos = reachablePositions[np.argmax(reachableDistances)]

            path = findPath(closestItemPos, grid, distances)

            grid = grid.astype(np.float)

            # Color the path in the grid
            start = BOT
            end = ITEM
            rng = (end - start)
            start = start + (rng/4)
            end = end - (rng/4)
            step = (end - start) / len(path)
            c = start
            for i,pos in enumerate(path):
                grid[pos[0], pos[1]] = c
                c += step

            now = datetime.now()
            if (now - lastUpdate).total_seconds() > UpdateRate:
                fieldImg = drawField(fieldObs, botObs, otherObs)
                gridImg = drawGrid(grid)
                updateWindow(fieldImg, gridImg)
                lastUpdate = now

            if len(path) > 1:
                targetPos = np.array(path[1])
            else:
                targetPos = closestItemPos

            botPos = botObs[[1,0]]
            botAngle = botObs[2] * 2 * np.pi

            targetPos = targetPos / [grid.shape[0]-1, grid.shape[1]-1]

            deltaPos = targetPos - botPos

            targetAngle = np.arctan2(deltaPos[0], deltaPos[1])
            deltaAngle = targetAngle - botAngle
            if deltaAngle < -np.pi: deltaAngle += 2*np.pi
            if deltaAngle > np.pi: deltaAngle -= 2*np.pi

            print(botPos, '->', targetPos, ' delta angle', deltaAngle)

            bigThreshold = 0.5
            smallThreshold = 0.1
            if deltaAngle > bigThreshold:
                turnRightBig()
            elif deltaAngle < -bigThreshold:
                turnLeftBig()
            elif deltaAngle > smallThreshold:
                turnRight()
            elif deltaAngle < -smallThreshold:
                turnLeft()

            if np.abs(deltaAngle) <= smallThreshold:
                moveForward(1500)
            #elif shuffle:
            #    moveForward(1)
            #    shuffle = False
            #elif not shuffle:
            #    moveBackward(1)
            #    shuffle = True
            

            #sleep(0.02)

    finally:
        pass#game.close()


if __name__ == "__main__":
     main()
