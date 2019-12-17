import json
import numpy as np
import math
import time
import socket
from datetime import datetime


ServerAddress = '127.0.0.1'
TeamName = 'Ralph'




def rotate():
    #with open('/dev/rfcomm2', 'wb') as f:
    #    f.write(b'c:999\r\n')
    print('nop')

def forward():
    #with open('/dev/rfcomm2', 'wb') as f:
    #    f.write(b'f:1\r\n')
    print('nop')


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((ServerAddress, 9735))

    buffer = ''
    lastAction = datetime.min

    for _ in range(1000):
        newLinePos = buffer.find('\n')
        while newLinePos < 0:
            data = s.recv(16384).decode('utf-8')
            buffer += data
            newLinePos = buffer.find('\n')

        line = buffer[:newLinePos]
        buffer = buffer[newLinePos+1:]
        gameState = json.loads(line)

        me = None
        bots = gameState['bots']
        print(bots)
        for bot in bots:
            if bot['name'] == TeamName:
                me = bot
                break
        
        myForward = me['forward']
        myAngle = math.degrees(math.atan2(myForward[1], myForward[0]))
        
        # t, r, b, l
        cellWalls = {
            0: [0,0,0,0],
            1: [0,0,0,1],
            2: [0,1,0,1],
            3: [1,0,0,1],
            4: [1,1,0,1],
        }

        field = []
        for column in gameState['data']:
            fieldColumn = []
            for entry in column:
                cellType = int(entry['type'])
                cellOrient = int(entry['orientation'])

                cell = cellWalls[cellType]

                while cellOrient > 0:
                    cell = [cell[3], cell[0], cell[1], cell[2]]
                    cellOrient -= 90

                fieldColumn.append(cell)

            fieldColumn = np.array(fieldColumn)
            field.append(fieldColumn)
        field = np.array(field)

        myPosX = (me['position'][0] * field.shape[1])
        myPosY = (me['position'][1] * field.shape[0])

        myOffsetX, myCellX = math.modf(myPosX)
        myOffsetY, myCellY = math.modf(myPosY)
        myCellX = int(myCellX)
        myCellY = int(myCellY)
        myOffsetX -= 0.5
        myOffsetY -= 0.5
        myOffsetDistance = math.sqrt(myOffsetX**2 + myOffsetY**2)

        myCell = field[myCellX, myCellX]
       
        myOrientation = int(round(myAngle / 90)) * 90
        if myOrientation < 0: myOrientation += 360


        wall = None
        if myOrientation == 0:
            wall = myCell[1]
        elif myOrientation == 90:
            wall = myCell[2]
        elif myOrientation == 180:
            wall = myCell[3]
        elif myOrientation == 270:
            wall = myCell[0]

        #print(me)
        #print(myAngle)
        print('cell', myCellX, myCellY, 'offset', myOffsetX, myOffsetY, myOffsetDistance, 'orientation', myOrientation, 'my cell', myCell, 'wall', wall)
        if (datetime.now() - lastAction).total_seconds() > 5:
            if wall == 1:
                print('rotating...')
                rotate()
            elif wall == 0:
                print("forward...")
                forward()
            lastAction = datetime.now()
