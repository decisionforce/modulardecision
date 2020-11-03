import subprocess
import time
import signal
import argparse
import os
import re

class CarlaDockerMonitor(object):
    def __init__(self):
        self.server_processes = []

    def _init_server(self, host, port=2000, card=0, town=3):
        host = str(host)
        port_start = str(port)
        port_end = str(port+2)
        carla_cmd = 'docker run --net=host --rm -d -it -e NVIDIA_VISIBLE_DEVICES=%s  --runtime nvidia carlasim/carla:0.9.5' %(card)
        carla_arg = ' ./CarlaUE4.sh /Game/Carla/Maps/Town0%s -benchmark -carla-server -fps=20 -world-port=%s -windowed -ResX=4 -ResY=4 -carla-no-hud' %(town, port)
        carla_cmd += carla_arg
        self.server_process = subprocess.Popen(carla_cmd,shell=True,preexec_fn=os.setsid)
        self.server_processes.append(self.server_process)
        print('service group id:%s'%os.getpgid(self.server_process.pid))
        print('server launch return code:%s'%self.server_process.poll())
        if self.server_process.poll()!=None:
            print("IP %s lanching failed"%host)
            raise ValueError('Carla Server launching failed')
        print('############################## Monitor succesfully opened carla with command #######')
        print(carla_cmd)

    def _close_server(self):
        try:
            for process in self.server_processes:
                os.killpg(os.getpgid(process.pid),signal.SIGTERM)
        except OSError:
            print('Process does not exist')

def container_numbers():
    result = os.popen('docker ps').read()
    numbers = len(re.findall('carlasim/carla', result))
    return int(numbers)

def check_network():
    result = os.popen('docker network ls').read()
    network_num = re.findall('carlanet', result)
    if len(network_num) > 0:
        return True
    else:
        return False

def docker_network_args(args):
    existing_numbers = container_numbers()
    idx = existing_numbers
    network_args = []
    while True:
        ip_suffix = 20 + idx
        ip = '172.16.0.'+str(ip_suffix)
        port = args.port + idx*3
        network_port = idx // 2
        card = idx #// 2
        idx = idx+1
        network_args.append([ip, port, network_port, card])

        new_containers = idx-existing_numbers
        if new_containers == args.ip_num:
            break
    return network_args

def create_docker_network():
    cmd = 'docker network create --subnet=172.31.0.0/16 carlanet'
    server_process = subprocess.Popen(cmd,shell=True,preexec_fn=os.setsid)
    if server_process.poll()!=None:
        print("Docker-network create lanching failed")
        raise ValueError('Carla Server launching failed')

def main(args):
    carlanet_exist = check_network()
    if not carlanet_exist:
        create_docker_network()
    network_args = docker_network_args(args)
    carla_monitor = CarlaDockerMonitor()
    for network_arg in network_args:
        ip, port, network_port, card = network_arg
        carla_monitor._init_server(host=ip, port=port, card=card, town=args.town)
        time.sleep(5.0)
        print("INIT SUCCESS")
        print("")
    return

def argsparser():
    parser = argparse.ArgumentParser("Docker starter")
    parser.add_argument('--ip_num', help='total ip numbers', type=int, default=1)
    parser.add_argument('--port', help='starting port', type=int, default=2000)
    parser.add_argument('--town', help='town numbers', type=int, default=3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
