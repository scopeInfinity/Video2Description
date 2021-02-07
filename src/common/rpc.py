import threading
import traceback

from six.moves.xmlrpc_client import ServerProxy
from six.moves.xmlrpc_server import SimpleXMLRPCServer

from common.config import get_rpc_config
from common.logger import logger


CONFIG = get_rpc_config()
SERVER_RUNAS = CONFIG["RPC_SERVER_RUNAS"]
PORT = CONFIG["RPC_PORT"]
SERVER_IP = CONFIG["RPC_ENDPOINT"]

lock = threading.Lock()

def rpc_decorator(f):
    def new_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Exception raised in rpc %s, %s\n%s" % (f, e, tb))
            raise e
    return new_f

def close_framework():
    exit()

def register_server(framework):
    print('Preparing for Register Server')
    server = SimpleXMLRPCServer((SERVER_RUNAS, PORT))
    print('Listening to %d' % PORT)
    server.register_function(rpc_decorator(framework.predict_fnames), 'predict_fnames')
    server.register_function(rpc_decorator(framework.predict_ids), 'predict_ids')
    server.register_function(rpc_decorator(framework.get_weights_status), 'get_weights_status')
    server.register_function(rpc_decorator(close_framework), 'close_framework')
    print("[RPC][Server][Started]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        raise
    except Exception:
        raise
    finally:
        print("[RPC][Server][Closing]")
        server.server_close()


def get_rpc():
    with lock:
        if hasattr(get_rpc, 'proxy'):
            return get_rpc.proxy
        get_rpc.proxy = ServerProxy("http://%s:%d/" % (SERVER_IP, PORT))
        return get_rpc.proxy
