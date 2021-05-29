import argparse

from models import faceEstimator, handEstimator, objectDetector

def _setup_parser():

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--model", type=str)

    return parser

def main():
    
    parser = _setup_parser()

    args = parser.parse_args()

    assert args.model in ("faceEstimator", "handEstimator", "objectDetector"), \
        "'--model' argument has to be one of 'faceEstimator', 'handEstimator' and 'objectDetector'"

    if args.model == "faceEstimator":
        faceEstimator()
    elif args.model == "handEstimator":
        handEstimator()
    elif args.model == "objectDetector":
        objectDetector()

if __name__ == "__main__":
    main()