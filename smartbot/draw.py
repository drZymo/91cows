from PIL import Image, ImageDraw
import numpy as np

def DrawObservation(obs, width, height):
    field, bots = obs

    nrRows, nrColumns = field.shape[0], field.shape[1]
    cellWidth = width / nrColumns
    cellHeight = height / nrRows

    im = Image.new('L', (width, height), color=0)
    draw = ImageDraw.Draw(im)

    def DrawCell(cellX, cellY, cellWidth, cellHeight, cell):
        l, t = cellX * cellWidth, (nrRows - 1 - cellY) * cellHeight
        r, b = l + cellWidth - 1, t + cellHeight - 1
        cx, cy = (l + r) // 2, (t + b) // 2

        qw, qh = cellWidth // 8, cellHeight // 8

        draw.rectangle((l, t, r, b), fill=0)
        draw.rectangle((l+1, t+1, r-1, b-1), fill=None, outline=64)

        if cell[0]: # top wall
            draw.line((l, t, r, t), fill=192)
        if cell[1]: # right wall
            draw.line((r, t, r, b), fill=192)
        if cell[2]: # bottom wall
            draw.line((l, b, r, b), fill=192)
        if cell[3]: # left wall
            draw.line((l, b, l, t), fill=192)

        if cell[4]: # coin
            draw.ellipse((cx-qw,cy-qh,cx+qw,cy+qh), fill=128, outline=192)
        if cell[5]: # treasure chest
            draw.rectangle((cx-qw,cy-qh,cx+qw,cy+qh), fill=128, outline=192)
        if cell[6]: # empty chest
            draw.rectangle((cx-qw,cy-qh,cx+qw,cy+qh), fill=None, outline=192)
        if cell[7]: # mimic chest
            draw.rectangle((cx-qw,cy-qh,cx+qw,cy+qh), fill=64, outline=192)
        if cell[8]: # spike trap
            draw.polygon((cx-qw,cy+qh, cx-(qw//2),cy-qh, cx,cy+qh, cx+(qw//2),cy-qh, cx+qw,cy+qh), fill=128, outline=192)
        if cell[9]: # bottle
            draw.ellipse((cx-(qw//2),cy-qh,cx+(qw//2),cy+qh), fill=128, outline=192)
        if cell[10]: # test tube
            draw.ellipse((cx-(qw//2),cy-qh,cx+(qw//2),cy+qh), fill=None, outline=192)

    def DrawField(field):
        for y,row in enumerate(field):
            for x,cell in enumerate(row):
                DrawCell(x, y, cellWidth, cellHeight, cell)

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
            line = line * [1, -1] + [0, 1]
            xy = line * [width, height]
            draw.line(xy.flatten().tolist(), fill=255)

    DrawField(field)

    for bot in bots:
        DrawBot(bot)

    del draw
    return im.tobytes()
