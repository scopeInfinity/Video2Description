import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer

PORT = 8000 # RPC
SERVER_IP = '127.0.0.1'

def register_server(framework):
    print 'Preparing for Register Server'
    server = SimpleXMLRPCServer(("0.0.0.0", PORT))
    print 'Listening to %d' % PORT
    server.register_function(framework.predict_fnames, 'predict_fnames')
    server.register_function(framework.predict_ids, 'predict_ids')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        raise
    except Exception:
        raise
    finally:
        print "Exiting"
        server.server_close()

def get_rpc():
    proxy = xmlrpclib.ServerProxy("http://%s:%d/" % (SERVER_IP, PORT))
    return proxy
