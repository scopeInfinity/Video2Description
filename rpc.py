import xmlrpclib
from SimpleXMLRPCServer import SimpleXMLRPCServer

PORT = 8000 # RPC

def register_server(framework):
    print 'Preparing for Register Server'
    server = SimpleXMLRPCServer(("localhost", PORT))
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
    proxy = xmlrpclib.ServerProxy("http://localhost:%d/" % PORT)
    return proxy
