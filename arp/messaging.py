import zmq

def pub_sock(context, port, addr="*"):
  sock = context.socket(zmq.PUB)
  sock.bind("tcp://%s:%d" % (addr, port))
  return sock

def sub_sock(context, port, poller=None, addr="127.0.0.1"):
  sock = context.socket(zmq.SUB)
  sock.connect("tcp://%s:%d" % (addr, port))
  sock.setsockopt(zmq.SUBSCRIBE, "")
  if poller is not None:
    poller.register(sock, zmq.POLLIN)
  return sock

def drain_sock(sock, wait_for_one=False):
  ret = []
  while 1:
    try:
      if wait_for_one and len(ret) == 0:
        dat = sock.recv()
      else:
        dat = sock.recv(zmq.NOBLOCK)
      ret.append(dat)
    except zmq.error.Again:
      break
  return ret

# TODO: print when we drop packets?
def recv_sock(sock, wait=False):
  dat = None
  while 1:
    try:
      if wait and dat is None:
        dat = sock.recv()
      else:
        dat = sock.recv(zmq.NOBLOCK)
    except zmq.error.Again:
      break
  return dat
