from .handBody_connections import *

def ray_casting(edges, xp, yp):
    count = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp-y1)/(y2-y1))*(x2-x1):
            count += 1
    
    return count%2 == 1

def get_body_connections_points(lmList):
        body_connections_points = ()
        for lines in BODY_CONNECTIONS:
            i, j = lines
            body_connections_points += (((lmList[i][1], lmList[i][2]), (lmList[j][1], lmList[j][2])), )

        return body_connections_points