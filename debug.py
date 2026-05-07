import pdb
from generate.dataset.images import Images
from generate.checkpoint import Checkpoint


if __name__ == "__main__":
    pdb.run("Images('/tmp').generate_images(2,1000,10,1000,lambda x: print(x),20,Checkpoint.get_checkpoint('/tmp/check.json'))")
