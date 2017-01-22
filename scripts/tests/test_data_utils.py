import unittest
from ..data_utils import train_val_generator


class DataTestCase(unittest.TestCase):
    def test_train_val_generator(self):
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_file = os.path.join(dir_path, "test_data.hdf5")

        test_file = "/Users/csong/Developer/workspace/ggo-data/ca123_64"
        from ..utils import listdir_fullpath
        import os
        if os.path.isdir(test_file):
            test_file = listdir_fullpath(test_file)

        under_test = train_val_generator(file=test_file, batch_size=10000, train_size=1, val_size=0, img_rows=64,
                                         img_cols=64, iter=10000, train_or_val="train", accept_partial_batch=True)

        under_test.next();
        under_test.next();


if __name__ == '__main__':
    unittest.main()
