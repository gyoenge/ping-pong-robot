import socket
import struct
import queue
import threading

class LC:
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.max_pos = 141
        self.init_pos = 71
        self.curr_pos = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def init(self) :
        return self.move(-self.max_pos)
    
    def move(self, message):
        if self.curr_pos + message > self.max_pos:
            message = self.max_pos - self.curr_pos
        elif self.curr_pos + message < 0:
            message = -self.curr_pos

        try:
            network_message = struct.pack('!i', message)
            sent = self.sock.sendto(network_message, (self.address, self.port))
            self.curr_pos += message
            return self.curr_pos
        except :
            print("Linear-control-error")
            
    def reset(self) :
        return self.move(self.init_pos-self.curr_pos)
    
    def close(self) :
        self.sock.close()
        
class LCThread(threading.Thread):
    def __init__(self, address, port):
        threading.Thread.__init__(self)
        self.lc = LC(address, port)
        self.lc.init()
        self.queue = queue.Queue()

    def add_move(self, action, message=None):
        self.queue.put((action, message))

    def run(self):
        while True:
            action, message = self.queue.get()
            if action == "move" :
                if message is None:
                    break
                self.lc.move(message)
                self.queue.task_done()
            elif action == "reset" :
                self.lc.reset()
                self.queue.task_done()
            elif action == "init" :
                self.lc.init()
                self.queue.task_done()
            elif action == "close" :
                self.lc.close()
                self.queue.task_done()

    def stop(self):
        self.add_move(None)
