import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = 'unet.hdf5'


def print_structure(file_name):
    """
    Prints out the structure of HDF5 file.

    Args:
      file_name (str) : Path to the file to analyze
    """
    f = h5py.File(file_name)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(file_name))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        print "#" * 200

        if len(f.items()) == 0:
            return

        for layer, g in f.items():
            print "-" * 200
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))
                print(len(value))
                if (key == "layer_names"):
                    layer_names = value
            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print p_name
                print("      {}: {}".format(p_name, param))
                for key, value in param.attrs.items():
                    print ("{}:{}".format(key, value))
                    for key in param.keys():
                        weights = param[key]
                        print(weights)
    finally:
        f.close()
    return layer_names
