import argparse
import sys
from logger import logger
from rpc import register_server, get_rpc, PORT

class Parser:
    def __init__(self):
        pass

    def init_framework(self):
        if not hasattr(self,'framework'):
            from framework import Framework
            self.framework = Framework()

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train','predict','server'])
        args = parser.parse_args(sys.argv[1:2])
        if args.command == 'train':
            self.train()
        if args.command == 'predict':
            self.predict()
        if args.command == 'server':
            self.server()
        print args.command

    def train(self):
        logger.debug("Training Mode")
        self.init_framework()
        self.framework.train_generator()

    def predict(self):
        parser = argparse.ArgumentParser(prog = sys.argv[0]+" predict", description = 'Prediction Mode')
        parser.add_argument('dataset', choices=['train','test','save_all_test'], help='Video dataset for prediction')
        parser.add_argument('-c', '--count', type = int, default = 10)
        args = parser.parse_args(sys.argv[2:])

        logger.debug("Prediction Mode")
        self.init_framework()
        if args.dataset == 'train':
            _ids = self.framework.get_trainids(args.count)
        elif args.dataset == 'test':
            _ids = self.framework.get_testids(args.count)
        elif args.dataset == 'save_all_test':
            self.framework.save_all(_ids = self.framework.get_testids())
            return
        else:
            assert False
        self.framework.predict_model(_ids = _ids)

    def server(self):
        logger.debug("Server Mode")
        parser = argparse.ArgumentParser(prog = sys.argv[0]+" server", description = 'Server Mode')
        parser.add_argument('-s', '--start', help='Start RPC Server', action='store_true')
        parser.add_argument('-pids', '--predict_ids',type=int, help='Obtain Results for given IDs', nargs='+')
        parser.add_argument('-pfs', '--predict_fnames', help='Obtain Results for given files', nargs='+')
        args = parser.parse_args(sys.argv[2:])
        if args.start:
            self.init_framework()
            register_server(self.framework)
        elif args.predict_ids:
            proxy = get_rpc()
            result = proxy.predict_ids( args.predict_ids )
            print result
        elif args.predict_fnames:
            proxy = get_rpc()
            result = proxy.predict_fnames( args.predict_fnames )
            print result
        else:
            parser.print_help()

if __name__ == "__main__":
    Parser().parse()
