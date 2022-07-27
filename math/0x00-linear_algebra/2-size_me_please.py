#!/usr/bin/env python3
def shape(ele, shp):
    if not isinstance(ele, list):
        return shp
    shp.append(len(ele))
    return shape(ele[0], shp)


def matrix_shape(matrix):
    return shape(matrix, [])
