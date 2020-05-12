# -*- coding: utf-8 -*-
import os
from six.moves import urllib
import tarfile

# download the BSD300 datasets
def download_bsd300(dest="data"):
    output_image_dir = os.path.join(dest, "BSDS300/images")
    if not os.path.exists(output_image_dir):
        if not os.path.exists(dest):
            os.mkdir(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)
        data = urllib.request.urlopen(url)
        file_path = os.path.join(dest, os.path.basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())
        print("decompression data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)
        print("datasets finished!")
    else:
        print("The datasets BSDS300 already exists!")


download_bsd300()