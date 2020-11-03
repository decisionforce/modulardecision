import subprocess
import time
import signal
import argparse
import os
import re

class CarlaDockerMonitor(object):
    def __init__(self):
        self.server_processes = []

    def _init_server(self, host, port=2000, card=0):
        host = str(host)
        port_start = str(port)
        port_end = str(port+2)
        carla_cmd = 'docker run --rm -d -it -p %s:%s-%s:%s-%s -e NVIDIA_VISIBLE_DEVICES=%s --runtime nvidia johnnywong/carla094' %(host, port_start, port_end, port_start, port_end, card)
        carla_arg = ' /bin/bash -c \'cp CarlaUE4/Binaries/Linux/CarlaUE4 CarlaUE4/Binaries/Linux/CarlaUE4_copy && mv CarlaUE4/Binaries/Linux/CarlaUE4_copy CarlaUE4/Binaries/Linux/CarlaUE4  && ./CarlaUE4.sh /Game/Carla/Maps/Town03 -benchmark -carla-server -fps=20 -world-port=2000 -windowed -ResX=100 -ResY=100 -carla-no-hud\''
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
    numbers = len(re.findall('johnnywong/carla094', result))
    return int(numbers)

def extra_network_args(args):
    existing_numbers = container_numbers()
    idx = existing_numbers
    network_args = []
    while True:
        ip_suffix = 25 + idx
        ip = '172.20.23.'+str(ip_suffix)
        #port = 2000+idx*3
        port = 2000
        network_port = idx // 2
        card = idx // 2
        idx = idx+1
        network_args.append([ip, port, network_port, card])

        new_containers = idx-existing_numbers
        if new_containers == args.ip_num:
            break
    return network_args

def extra_ip(network_port, ip):
    # subprocess.call(['tclsh', 'sudo.tcl']) # if password is required
    cmd = 'sudo ifconfig eth0:%s %s netmask 255.255.255.0 up' % (network_port, ip)
    server_process = subprocess.Popen(cmd,shell=True,preexec_fn=os.setsid)
    print('extra ip :%s'%ip)
    if server_process.poll()!=None:
        print("IP %s lanching failed"%ip)
        raise ValueError('Carla Server launching failed')

def restart_networking():
    cmd = 'sudo /etc/init.d/networking restart'
    server_process = subprocess.Popen(cmd,shell=True,preexec_fn=os.setsid)
    if server_process.poll()!=None:
        print("Restart networking failed")
        raise ValueError('Restart networking failed')

def main(args):
    network_args = extra_network_args(args)
    carla_monitor = CarlaDockerMonitor()
    restart_networking() 
    for network_arg in network_args:
        ip, port, network_port, card = network_arg
        extra_ip(network_port, ip)
        carla_monitor._init_server(host=ip, port=port, card=card)
    return

def argsparser():
    parser = argparse.ArgumentParser("Docker starter")
    parser.add_argument('--ip_num', help='total ip numbers', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    main(args)
