import argparse
import sys
from logger import logger

class Parser:
    def __init__(self):
        pass

    def init_framework(self):
        if not hasattr(self,'framework'):
            from framework import Framework
            self.framework = Framework()

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train','predict'])
        args = parser.parse_args(sys.argv[1:2])
        if args.command == 'train':
            self.train()
        if args.command == 'predict':
            self.predict()
        print args.command

    def train(self):
        logger.debug("Training Mode")
        self.init_framework()
        self.framework.train_generator()

    def predict(self):
        parser = argparse.ArgumentParser(prog = sys.argv[0]+" predict", description = 'Prediction Mode')
        parser.add_argument('dataset', choices=['train','test'], help='Video dataset for prediction')
        parser.add_argument('-c', '--count', type = int, default = 10)
        args = parser.parse_args(sys.argv[2:])

        logger.debug("Prediction Mode")
        self.init_framework()
        if args.dataset == 'train':
            _ids = self.framework.get_trainids(args.count)
        elif args.dataset == 'test':
            _ids = self.framework.get_testids(args.count)
        else:
            assert False
        self.framework.predict_model(_ids = _ids)

if __name__ == "__main__":
    Parser().parse()
