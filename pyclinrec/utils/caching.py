import struct

import numpy as np


def array_to_redis(redis, array, key):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    h, w = array.shape
    shape = struct.pack(">II", h, w)
    encoded = shape + array.tobytes()

    # Store encoded data in Redis
    redis.set(key, encoded)
    return


def redis_to_array(redis, key):
    """Retrieve Numpy array from Redis key 'n'"""
    encoded = redis.get(key)
    h, w = struct.unpack(">II", encoded[:8])
    return np.frombuffer(encoded[8:]).reshape(h, w)


def redis_key_exists(r, n):
    """Check if Redis key 'n' exists"""
    return r.exists(n)
