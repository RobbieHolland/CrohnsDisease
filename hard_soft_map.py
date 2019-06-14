import numpy as np
import functools
from multiprocessing import Pool
import multiprocessing as mp

def map_f(coord, point):
    return np.sum(np.square(coord - point))

def generate_map(coord, dims):
    hard_soft_map = np.zeros(dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                hard_soft_map[i, j, k] = map_f(coord, np.array([i, j, k]))
    hard_soft_map = np.sqrt(hard_soft_map)
    return hard_soft_map / np.sum(hard_soft_map)

def generate_map2(coord, dims):
    xx, yy, zz = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]
    w = np.exp(-((xx - coord[0])**2 + (yy - coord[1])**2 + (zz - coord[2])**2)/(2 * 20))
    w = np.sqrt(w)
    w = w / np.sum(w)
    return w

def generate_batch_maps(batch_coords, dims):
    batch_coords = np.array(np.array(dims) / 2 + batch_coords * dims).astype(int)
    mappable_generate_map = functools.partial(generate_map2, dims=dims)

    with Pool(processes=mp.cpu_count()) as pool:
        print(f'Creating {len(batch_coords)} hard attention maps \r', end='')
        result = pool.map(mappable_generate_map, batch_coords)
        return np.stack(result, axis=0)

# print(generate_batch_maps(np.array([[.5, .5, .5], [0, 0, 0]]), (10, 20, 30))[0, 4, 5, 6])
