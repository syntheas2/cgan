from cgan.__main__ import main as train_main
from cgan.sampling import main as sampling_main

def train():
    train_main()


def sample():
    sampling_main()


if __name__ == '__main__':
    # train_gmm()
    # train()
    sample()
    