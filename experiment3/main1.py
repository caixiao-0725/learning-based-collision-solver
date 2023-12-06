import socket
import time
import numpy as np
import trimesh
import torch

objSrc = 'E:/siggraph2024/siggraph-2024/assets/obj/47.obj'

np.set_printoptions(threshold=np.inf,suppress=True)

class MyWrappr():
    def __init__(self):
        self.mesh = trimesh.load_mesh(objSrc)
        self.vertexs = torch.tensor(self.mesh.vertices,dtype=torch.float32,requires_grad=False)
        self.vertsNum = self.vertexs.shape[0]
        self.faces = torch.tensor(self.mesh.faces,dtype=torch.int32,requires_grad=False)
        self.facesNum = self.faces.shape[0]

        
        vertexs_str = np.array2string(self.vertexs.numpy(),precision=6).replace('[', '').replace(']', '')
        faces_str = np.array2string(self.faces.numpy()).replace('[', '').replace(']', '')

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('127.0.0.1', 8080))
        self.server.listen(1)
        self.connection, self.address = self.server.accept()

        self.connection.send(bytes(vertexs_str, encoding="ascii"))
        time.sleep(0.1)

        self.connection.send(bytes(faces_str, encoding="ascii"))

    def reset(self):
        return 

    def step(self):
        #reward = ccd(self.triangle0,self.triangle1,action)

        send_str = "I'm fine, thx!"
        self.connection.send(bytes(send_str, encoding="ascii"))
        print("send:   {}".format(send_str))

        recv_str = self.connection.recv(1024)
        recv_str = recv_str.decode("ascii")
        print("receive:{}".format(recv_str))



if __name__ == '__main__':
    env = MyWrappr()

