import sys 


import git
import sklearn.datasets
from PIL import Image

PATH_TO_ROOT = git.Repo(".", search_parent_directories=True).working_dir
sys.path.append(PATH_TO_ROOT)


random_state = 2024 #REPLACE WITH YAML VALUE

#import toy dataset
def toy_data(n_samples: int, random_state: int = random_state):
    pass

#import CMU face dataset
def load_as_png(image_path: str):
    im = Image.open("data/dataset/cmu+face+images/faces/"+image_path)
    im.show()

if __name__ == "__main__":
    load_as_png("an2i/an2i_left_angry_open_2.pgm")

