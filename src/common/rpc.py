import threading
import xmlrpclib

from SimpleXMLRPCServer import SimpleXMLRPCServer

from common.config import get_rpc_config


CONFIG = get_rpc_config()
SERVER_RUNAS = CONFIG["RPC_SERVER_RUNAS"]
PORT = CONFIG["RPC_PORT"]
SERVER_IP = CONFIG["RPC_ENDPOINT"]

lock = threading.Lock()

def close_framework():
    exit()

def register_server(framework):
    print('Preparing for Register Server')
    server = SimpleXMLRPCServer((SERVER_RUNAS, PORT))
    print('Listening to %d' % PORT)
    server.register_function(framework.predict_fnames, 'predict_fnames')
    server.register_function(framework.predict_ids, 'predict_ids')
    server.register_function(framework.get_weights_status, 'get_weights_status')
    server.register_function(close_framework, 'close_framework')
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
        get_rpc.proxy = xmlrpclib.ServerProxy("http://%s:%d/" % (SERVER_IP, PORT))
        return get_rpc.proxy
