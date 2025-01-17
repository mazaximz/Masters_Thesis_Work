#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:41:20 2019

@author: nipevalj
"""

# Copyright 2000 softSurfer, 2012 Dan Sunday
# This code may be freely used and modified for any purpose
# providing that this copyright notice is included with it.
# SoftSurfer makes no warranty for this code, and cannot be held
# liable for any real or imagined damage resulting from its use.
# Users of this code must verify correctness for their application.
 

# a Point is defined by its coordinates {int x, y;}
#===================================================================
 

# isLeft(): tests if a point is Left|On|Right of an infinite line.
#    Input:  three points P0, P1, and P2
#    Return: >0 for P2 left of the line through P0 and P1
#            =0 for P2  on the line
#            <0 for P2  right of the line
#    See: Algorithm 1 "Area of Triangles and Polygons"
def isLeft( P0, P1, P2 ):

    return ( (P1.x - P0.x) * (P2.y - P0.y)
            - (P2.x -  P0.x) * (P1.y - P0.y) );

#===================================================================


# cn_PnPoly(): crossing number test for a point in a polygon
#      Input:   P = a point,
#               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
#      Return:  0 = outside, 1 = inside
# This code is patterned after [Franklin, 2000]
def cn_PnPoly( P, V, n ):

    cn = 0;    # the  crossing number counter

    # loop through all edges of the polygon
    for i in range(n):    # edge from V[i]  to V[i+1]
       if (((V[i].y <= P.y) and (V[i+1].y > P.y))     # an upward crossing
        or ((V[i].y > P.y) and (V[i+1].y <=  P.y))): # a downward crossing
            # compute  the actual edge-ray intersect x-coordinate
            vt = (float)(P.y  - V[i].y) / (V[i+1].y - V[i].y);
            if (P.x <  V[i].x + vt * (V[i+1].x - V[i].x)): # P.x < intersect
                 cn += 1;   # a valid crossing of y=P.y right of P.x
    
    return (cn&1);    # 0 if even (out), and 1 if  odd (in)


#===================================================================


# wn_PnPoly(): winding number test for a point in a polygon
#      Input:   P = a point,
#               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]//      
#      Return:  wn = the winding number (=0 only when P is outside)

def wn_PnPoly( P, V, n ):

    wn = 0;    # the  winding number counter

    # loop through all edges of the polygon
    for i in range(n):    # edge from V[i] to  V[i+1]
        if (V[i].y <= P.y):          # start y <= P.y
            if (V[i+1].y  > P.y):      # an upward crossing
                 if (isLeft( V[i], V[i+1], P) > 0):  # P left of  edge
                     wn += 1;            # have  a valid up intersect
        
        else:                         # start y > P.y (no test needed)
            if (V[i+1].y  <= P.y):     # a downward crossing
                 if (isLeft( V[i], V[i+1], P) < 0):  # P right of  edge
                     wn -= 1;            # have  a valid down intersect
    
    return wn;

#===================================================================