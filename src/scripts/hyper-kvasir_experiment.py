import argparse
from utils import get_logger

def get_argument_parser():
    parser = argparse.ArgumentParser("Hyper Kvasir")
    parser.add_argument("--output", "-o", type=str, help="path to save ")
    parser.add_argument("--checkpoint", "-k", type=str, default=None, )
    args = parser.parse_args()
    return args
    
def main():
    args = get_argument_parser()
    
if __name__ == "__main__":
    main()