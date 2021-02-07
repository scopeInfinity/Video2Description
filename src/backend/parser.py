import argparse
import sys

from common.logger import logger
from common.rpc import register_server, get_rpc, PORT

class Parser:
    def __init__(self):
        pass

    def init_framework(self, model_fname = None, train_mode = False):
        if not hasattr(self,'framework'):
            from backend.framework import Framework
            if model_fname is not None:
                self.framework = Framework(model_load = model_fname, train_mode = train_mode)
            else:
                self.framework = Framework(train_mode = train_mode)

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=['train','predict','server','predict_all_model'])
        args = parser.parse_args(sys.argv[1:2])
        if args.command == 'train':
            self.train()
        if args.command == 'predict':
            self.predict()
        if args.command == 'server':
            self.server()
        if args.command == 'predict_all_model':
            self.predict_all_model()
        print(args.command)

    def train(self):
        logger.debug("Training Mode")
        self.init_framework(train_mode = True)
        self.framework.train_generator()

    def predict_all_model(self):
        import glob, os
        from backend.framework import Framework, MFNAME
 
        logger.debug("PredictAllModel Mode")
        result_dir = 'CombinedResults'
        os.system('mkdir -p %s' % result_dir)
        for fname in glob.glob(MFNAME+"_*"):
            save_file = result_dir + "/result_"+os.path.basename(fname)+"_.txt"
            if os.path.exists(save_file):
                continue
            logger.debug("Working on model %s " % fname)
            self.framework = Framework(model_load = fname)
            self.framework.save_all(_ids = self.framework.get_testids(), save = save_file)
        logger.debug("Done")

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
        parser.add_argument('-i', '--init-only', help='Prepares early caches for faster execution', action='store_true')
        parser.add_argument('-s', '--start', help='Start RPC Server', action='store_true')
        parser.add_argument('-m', '--model', help='Model file')
        parser.add_argument('-pids', '--predict_ids',type=int, help='Obtain Results for given IDs', nargs='+')
        parser.add_argument('-pfs', '--predict_fnames', help='Obtain Results for given files', nargs='+')
        parser.add_argument('-cf', '--close_framework', help='Close Server Framework', action='store_true')
        args = parser.parse_args(sys.argv[2:])
        if args.init_only:
            self.init_framework()
            print("[RPC][Server][Init][Done]")
        elif args.start:
            model_fname = None
            if args.model:
                model_fname = args.model
            self.init_framework(model_fname)
            register_server(self.framework)
        elif args.predict_ids:
            proxy = get_rpc()
            result = proxy.predict_ids( args.predict_ids )
            print(result)
        elif args.predict_fnames:
            proxy = get_rpc()
            result = proxy.predict_fnames( args.predict_fnames )
            print(result)
        elif args.close_framework:
            proxy = get_rpc()
            proxy.close_framework()
            print("[RPC][Send][close_framework]")
        else:
            parser.print_help()

if __name__ == "__main__":
    Parser().parse()
