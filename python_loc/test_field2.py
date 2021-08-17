#!/usr/bin/env python

import time
import math
import socket
import struct
import select
import threading
import enum
import os
import re

import dwm1001_ble
import nano33ble

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize
from simple_pid import PID
import config as cfg

from scipy.ndimage.filters import uniform_filter1d
from scipy.signal.signaltools import wiener
from scipy.signal import savgol_filter
#X_f, Y_f, Z_f = wiener(np.array([X_lse, Y_lse, Z_lse]))
from scipy.ndimage import gaussian_filter1d



dwm_fd      = None
nano33_fd   = None
parrot_sock = None
plot_sock   = None

dwm_loc     = None
parrot_data = None

X_lse = []
Y_lse = []
Z_lse = []
T = []

X_filtered = []
Y_filtered = []
Z_filtered = []

moving_window = 21

hist_len_sec = 5

total_pos = 0
total_calc = 0

PID_CONTROL_RATE_HZ = 2

class dwm_source(enum.Enum):
    BLE = 0,
    SOCK = 1,

DWM_DATA_SOURCE = dwm_source.BLE
#DWM_DATA_SOURCE = dwm_source.SOCK

class len_log:
    def __init__(self, anch):
        self.anch = anch
        self.data = []
        self.T = []

    def add_to_filter(self, l, ts):
        self.data.append(l)
        self.T.append(ts)

        apply_filter = 2
        if apply_filter == 1 and len(self.data) > moving_window:
            self.data = list(uniform_filter1d(self.data, size=moving_window, mode="reflect"))
        if apply_filter == 2:
            self.data = list(gaussian_filter1d(self.data, 6))

        #print(self.data)
        while (len(self.T) > 0) and (ts - self.T[0] > hist_len_sec):
            self.data.pop(0)
            self.T.pop(0)

    def get_log(self):
        return self.data

    def get_last_filtered(self):
        apply_filter = 1
        res = self.data[-1]

        return res

anch_len_log = {}
for anch in cfg.ANCHORS.keys():
    anch_len_log[anch] = len_log(anch)
#
# Welford's online algorithm
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
#
class welford_state:
    count = 0
    mean = 0
    m2 = 0

    def avg_welford(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.m2 += delta * delta2

        return self.mean;

class avg_rate:
    welford = welford_state()
    ts = 0.0

    def __call__(self):
        rate = 0
        now = time.time()
        if self.ts:
            rate = 1.0 / (now - self.ts)
            # Disable average, sometimes we have very high @rate
            # so the whole average goes mad, better to see rate in
            # momentum
            #rate = self.welford.avg_welford(rate)
        self.ts = now

        return rate

class drone_navigator(threading.Thread):
    # PID tuning files, format is: float float float [float, float]
    pid_tuning_file = "./pid.tuning"

    # Default PID config
    default_pid_components = (10, 30, 0.1)
    default_pid_limits = (-100, 100)

    # Thread data
    data = None
    lock =  threading.Lock()

    # XXX
    pitch = 0
    roll = 0
    rate = 0.0

    def __init__(self, target_x, target_y):
        # Create commands sock
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        components = self.default_pid_components

        # Set desired landing coordinates
        x_pid = PID(Kp=components[0], Ki=components[1], Kd=components[2],
                    setpoint=target_x,
                    proportional_on_measurement=False)
        y_pid = PID(Kp=components[0], Ki=components[1], Kd=components[2],
                    setpoint=target_y,
                    proportional_on_measurement=False)

        # Control coeff limits
        x_pid.output_limits = self.default_pid_limits
        y_pid.output_limits = self.default_pid_limits

        self.start_time = time.time()

        self.sock = sock
        self.x_pid = x_pid
        self.y_pid = y_pid

        super().__init__()

    def _send_command(self, roll, pitch, yaw, throttle):
        # 4x signed chars
        buf = struct.pack("bbbb", int(roll), int(pitch), int(yaw), int(throttle))
        self.sock.sendto(buf, (cfg.UDP_COMMANDS_IP, cfg.UDP_COMMANDS_PORT))

    def _pid_tuning(self, pid, tuning_file):
        tunings = self.default_pid_components
        limits = self.default_pid_limits

        if os.path.exists(tuning_file):
            with open(tuning_file, "r") as file:
                line = file.readline()
                components = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(components) >= 3:
                    # To floats
                    components = [float(f) for f in components]
                    tunings = components[0:3]
                    if pid.Kp != tunings[0] or pid.Ki != tunings[1] or \
                       pid.Kd != tunings[2]:
                        pid.reset()
                    if len(components) >= 5:
                        limits = components[3:5]

        pid.tunings = tunings
        pid.output_limits = limits

    def run(self):
        while True:
            time.sleep(1/PID_CONTROL_RATE_HZ)

            self.lock.acquire()
            data = self.data
            self.lock.release()

            if data is None:
                continue

            (x, y) = data

            self._pid_tuning(self.x_pid, self.pid_tuning_file)
            self._pid_tuning(self.y_pid, self.pid_tuning_file)

            control_x = self.x_pid(x)
            control_y = self.y_pid(y)

            # Parrot accepts in signed percentage, i.e. [-100, 100] range
            roll = int(control_x)
            pitch = int(control_y)
            self._send_command(roll, pitch, 0, 0)

            self.roll = roll
            self.pitch = pitch
            self.rate = rate

    def navigate_drone(self, x, y):
        self.lock.acquire()
        self.data = (x, y)
        self.lock.release()

def create_dwm_sock():
    # Create sock and bind
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((cfg.MCAST_GRP, cfg.MCAST_PORT))

    # Join group
    mreq = struct.pack("4sl", socket.inet_aton(cfg.MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    sock.setblocking(0)

    return sock

def create_dwm_ble():
    manager = dwm1001_ble.DwmDeviceManager()
    device = dwm1001_ble.DwmDevice(mac_address=cfg.TAG_MAC, manager=manager)

    device.connect()
    manager.start()

    global dwm_device
    dwm_device = device

    return device.eventfd

def create_nano33_ble():
    manager = nano33ble.Nano33DeviceManager()
    device = nano33ble.Nano33Device(mac_address=cfg.NANO33_MAC, manager=manager)

    device.connect()
    manager.start()

    global nano33_device
    nano33_device = device

    return device.eventfd

def create_dwm_fd():
    if DWM_DATA_SOURCE == dwm_source.SOCK:
        return create_dwm_sock()
    return create_dwm_ble()

def create_parrot_sock():
    # Create parrot sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((cfg.UDP_TELEMETRY_IP, cfg.UDP_TELEMETRY_PORT))
    sock.setblocking(0)

    return sock

def create_plot_sock():
    # Create plot sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    return sock

def send_plot_data(sock, x, y, z, parrot_alt, ts, rate, nr_anchors, navigator, loc):
    x_pid = navigator.x_pid
    y_pid = navigator.y_pid

    x2585_len = -1.0
    x262d_len = -1.0
    x28b9_len = -1.0
    x260f_len = -1.0

    for anch in loc["anchors"]:
        dist = anch["dist"]["dist"]
        addr = anch["dist"]["addr"]

        if addr == 0x2585:
            x2585_len = dist
        if addr == 0x262d:
            x262d_len = dist
        if addr == 0x28b9:
            x28b9_len = dist
        if addr == 0x260f:
            x260f_len = dist

    # 1 double, 17 floats, 3 int32, 4 floats
    buf = struct.pack("dfffffffffffffffffiiiffff",
                      ts, x, y, z, parrot_alt, rate,
                      x_pid.Kp, x_pid.Ki, x_pid.Kd,
                      x_pid.components[0], x_pid.components[1], x_pid.components[2],
                      y_pid.Kp, y_pid.Ki, y_pid.Kd,
                      y_pid.components[0], y_pid.components[1], y_pid.components[2],
                      navigator.roll, navigator.pitch, nr_anchors,
                      x2585_len, x262d_len, x28b9_len, x260f_len)
    sock.sendto(buf, (cfg.UDP_PLOT_IP, cfg.UDP_PLOT_PORT))

def receive_dwm_location_from_sock(sock):
    # Location header
    fmt = "iiihhiii"
    sz = struct.calcsize(fmt)
    buf = sock.recv(sz, socket.MSG_PEEK)

    # Unpack location header
    (x, y, z, pos_qf, pos_valid, ts_sec, ts_usec, nr_anchors) = struct.unpack(fmt, buf)

    print("ts:%ld.%06ld [%d,%d,%d,%u] " % (ts_sec, ts_usec, x, y, z, pos_qf), end='')

    location = {
        'calc_pos': {
            'x':  float(x) / 1000,
            'y':  float(y) / 1000,
            'z':  float(z) / 1000,
            'qf': pos_qf,
            'valid': pos_valid,
        },
        'ts': float("%ld.%06ld" % (ts_sec, ts_usec)),
        'anchors': [],
    }

    # Skip size of the location header
    off = sz

    # Read the whole location packet
    fmt = "iiihhihh"
    sz = sz + struct.calcsize(fmt) * nr_anchors
    buf = sock.recv(sz)

    # For each anchor
    for i in range(0, nr_anchors):
        (x, y, z, pos_qf, pos_valid, dist, addr, dist_qf) = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)

        print("#%u) a:0x%08x [%d,%d,%d,%u] d=%u,qf=%u " % (i, addr, x, y, z, pos_qf, dist, dist_qf), \
              end='')

        anchor = {
            'pos': {
                'x':  float(x) / 1000,
                'y':  float(y) / 1000,
                'z':  float(z) / 1000,
                'qf': pos_qf,
                'valid': pos_valid,
            },
            'dist': {
                'dist': float(dist) / 1000,
                'addr': addr,
                'qf': dist_qf
            },
        }

        location['anchors'].append(anchor)

    print('')

    return location

def receive_dwm_location(dwm_fd):
    if DWM_DATA_SOURCE == dwm_source.SOCK:
        return receive_dwm_location_from_sock(dwm_fd)

    global dwm_device
    loc = dwm_device.get_location()

    # FIXME: extend the BLE anchors with coords from config
    # FIXME: well, this is ugly
    for anchor in loc['anchors']:
        anchor_coords = cfg.ANCHORS[anchor['dist']['addr']]
        anchor['pos']['x'] = anchor_coords[0]
        anchor['pos']['y'] = anchor_coords[1]
        anchor['pos']['z'] = anchor_coords[2]

    return loc

def receive_nano33_data():
    global nano33_device
    return nano33_device.get_data()

def receive_parrot_data_from_sock(sock):
    fmt = "iiffff"
    sz = struct.calcsize(fmt)
    buf = sock.recv(sz)
    sec, usec, alt, roll, pitch, yaw = struct.unpack(fmt, buf)

    parrot_data = {
        'ts':    float("%ld.%06ld" % (sec, usec)),
        'alt':   alt,
        'roll':  roll,
        'pitch': pitch,
        'yaw':   yaw
    }

    print("parrot_data: ts=%.6f alt=%f roll=%f pitch=%f yaw=%f" % \
        (parrot_data['ts'], parrot_data['alt'], parrot_data['roll'], parrot_data['pitch'], parrot_data['yaw']))

    return parrot_data

def is_dwm_location_reliable(loc):
    return len(loc['anchors']) >= 3

def get_dwm_location_or_parrot_data():
    global dwm_fd, nano33_fd, parrot_sock, dwm_loc, parrot_data

    if dwm_fd is None:
        dwm_fd = create_dwm_fd()
    if nano33_fd is None:
        nano33_fd = create_nano33_ble()
    if parrot_sock is None:
        parrot_sock = create_parrot_sock()

    dwm_received = False

    # Suck everything from the socket, we need really up-to-date data
    while (True):
        # Wait inifinitely if we don't have reliable DWM location
        timeout = 0 if dwm_received else None

        rd, wr, ex = select.select([dwm_fd, nano33_fd, parrot_sock], [], [], timeout)
        if 0 == len(rd):
            break

        if dwm_fd in rd:
            loc = receive_dwm_location(dwm_fd)
            print("ts:%.6f [%.2f,%.2f,%.2f,%u] " % (loc['ts'], loc['calc_pos']['x'], loc['calc_pos']['y'], loc['calc_pos']['z'],
                                               loc['calc_pos']['qf']), end='')
            i = 0
            for anch in loc['anchors']:
                print("#%u) a:0x%04x [%.2f,%.2f,%.2f,%.2f] d=%.2f,qf=%u " % (i, anch['dist']['addr'],
                                                                   anch['pos']['x'], anch['pos']['y'], anch['pos']['z'],
                                                                   anch['pos']['qf'], anch['dist']['dist'], anch['dist']['qf']), \
                      end='')
                i += 1
            print('')

            if is_dwm_location_reliable(loc):
                dwm_received = True
                dwm_loc = loc
        if parrot_sock in rd:
            parrot_data = receive_parrot_data_from_sock(parrot_sock)
        if nano33_fd in rd:
            acc, gyro, mag = receive_nano33_data()
            print("acc = {x=%f y=%f z=%f ts=%d}, gyro = {x=%f y=%f z=%f ts=%d}, mag = {x=%f y=%f z=%f ts=%d}" % \
              (acc[0], acc[1], acc[2], acc[3],
               gyro[0], gyro[1], gyro[2], gyro[3],
               mag[0], mag[1], mag[2], mag[3]))

    return dwm_loc, parrot_data

def find_anchor_by_addr(location, addr):
    for anchor in location['anchors']:
        if anchor['dist']['addr'] == addr:
            return anchor

    return None

def func1(X, loc):
    sum = 0
    for anch in loc["anchors"]:
        anchor_pos = np.array([anch["pos"]["x"], anch["pos"]["y"], anch["pos"]["z"]], dtype=np.float64)
        dist = anch["dist"]["dist"]
        sum += (np.linalg.norm(X - anchor_pos) - dist) ** 2
    return sum

# grad is probably wrong, check it later
# btw it works fine without it
# this shit probably good if we have some anchors with non zero z coordinate
def grad_func1(X, la, lb, lc, ld):
    na = np.linalg.norm(X - A)
    nb = np.linalg.norm(X - B)
    nc = np.linalg.norm(X - C)
    nd = np.linalg.norm(X - D)

    ret = 2 * (1 - la / na) * (X - A) + 2 * (1 - lb / nb) * (X - B) + \
          2 * (1 - lc / nc) * (X - C) + 2 * (1 - ld / nd) * (X - D)

    return ret

assigned = False
def calc_pos(X0, loc):

    # all are the experiments
    #res = least_squares(func, X0, loss='soft_l1', jac=jac, bounds=([-3, -3, 0.0], [3, 3, 3]), args=(la, lb, lc, ld), verbose=1)

    #to make smooth path, but in general this is shit i think
    # lowb = X0-0.2
    # if lowb[2] < 0:
    #     lowb[2] = 0
    # upb = X0+0.2
    lowb = [-np.inf, -np.inf, 0]
    upb = [np.inf, np.inf, np.inf]

    start = time.time()
    #res = least_squares(func1, X0, loss='cauchy', f_scale=0.001, bounds=(lowb, upb),
    res = least_squares(func1, X0, bounds=(lowb, upb),
                        #args=(la, lb, lc, ld), verbose=1)
                        args=[loc], verbose=0)

    ##also decent and fast
    # res = minimize(func1, X0, method="L-BFGS-B", bounds=[(-math.inf, math.inf), (-math.inf, math.inf), (0, math.inf)],
    #                #options={'ftol': 1e-4, 'disp': True}, args=(la, lb, lc, ld))
    #                options={'ftol': 1e-4,'eps' : 1e-4, 'disp': False}, args=loc)

    # res = minimize(func1, X0, method="SLSQP", bounds=[(-math.inf, math.inf), (-math.inf, math.inf), (0, math.inf)],
    #           options={ 'ftol': 1e-5, 'disp': True}, args=loc)
    #           #options={'ftol': 1e-5,'eps' : 1e-8, 'disp': False}, args=loc)

    stop = time.time()
    print("calc time {}".format(stop - start))
    # experiments
    #res = minimize(func1, X0, method='BFGS', options={'xatol': 1e-8, 'disp': True}, args=(la, lb, lc, ld))
    #res = optimize.shgo(func1, bounds=[(-10, 10), (-10, 10), (0, 10)], args=(la, lb, lc, ld),n=200, iters=5, sampling_method='sobol')
    #res = minimize(func1, X0, method='BFGS', options={'disp': True}, args=(la, lb, lc, ld))
    return res.x

def calc_pos_pyramid(loc):
    start = time.time()

    a = 0.64
    la = -1
    lb = -1
    lc = -1
    for anch in loc["anchors"]:
        if anch['dist']['addr'] == 0x28b9:
            la = anch['dist']['dist']
        if anch['dist']['addr'] == 0x262d:
            lb = anch['dist']['dist']
        if anch['dist']['addr'] == 0x260f:
            lc = anch['dist']['dist']

    try:
        if (a**2 + lc**2 - la**2 < 0) and (a**2 + la**2 - lb**2 < 0) and \
                (a**2 + la**2 - lc**2 > 0) and (a**2 + lb**2 - la**2 > 0):

            print("case1, la %f lb %f lc %f" % (la, lb, lc))
            on = 1 / (2 * a) * (lb**2 - la**2 - a**2)
            om = a + 1 / (2 * a) * (la**2 - lc**2 - a**2)
            oh = math.sqrt(lc**2 - on**2 - (om - a)**2)
            res = [-on, om, oh]

        elif (a**2 + lc**2 - la**2 < 0) and (a**2 + la**2 - lc**2 > 0) and \
                (a**2 + la**2 - lb**2 > 0) and (a**2 + lb**2 - la**2 > 0):

            print("case 2, la %f lb %f lc %f" % (la, lb, lc))
            on = 1 / (2 * a) * (la**2 - lb**2 + a**2)
            om = a + 1 / (2 * a) * (la**2 - lc**2 - a**2)
            oh = math.sqrt(lc**2 - on**2 - (om - a)**2)
            res = [on, om, oh]

        elif (a**2 + lb**2 - la**2 < 0) and (a**2 + lc**2 - la**2 < 0) and \
                (a**2 + la**2 - lb**2 > 0) and (a**2 + la**2 - lc**2 > 0):

            print("case 3, la %f lb %f lc %f" % (la, lb, lc))
            on = a + 1 / (2 * a) * (la**2 - lb**2 - a**2)
            om = a + 1 / (2 * a) * (la**2 - lc**2 - a**2)
            oh = math.sqrt(lc**2 - on**2 - om**2)
            res = [on, om, oh]

        elif (a**2 + la**2 - lb**2 < 0) and (a**2 + lb**2 - la**2 > 0) and \
                (a**2 + la**2 - lc**2 > 0) and (a**2 + lc**2 - la**2 > 0):

            print("case 4, la %f lb %f lc %f" % (la, lb, lc))
            on = 1 / (2 * a) * (lb**2 - la**2 - a**2)
            om = 1 / (2 * a) * (la**2 - lc**2 + a**2)
            oh = math.sqrt(la**2 - on**2 - om**2)
            res = [-on, om, oh]

        elif (a**2 + lb**2 - la**2 > 0) and (a**2 + la**2 - lb**2 > 0) and \
                (a**2 + lc**2 - la**2 > 0) and (a**2 + la**2 - lc**2 > 0):

            print("case 5, la %f lb %f lc %f" % (la, lb, lc))
            on = 1 / (2 * a) * (la**2 - lb**2 + a**2)
            om = 1 / (2 * a) * (la**2 - lc**2 + a**2)
            oh = math.sqrt(la**2 - on**2 - om**2)
            res = [on, om, oh]

        elif (a**2 + lb**2 - la**2 < 0) and (a**2 + la**2 - lb**2 > 0) and \
                (a**2 + lc**2 - la**2 > 0) and (a**2 + la**2 - lc**2 > 0):

            print("case 6, la %f lb %f lc %f" % (la, lb, lc))
            on = a + 1 / (2 * a) * (la**2 - lb**2 - a**2)
            om = 1 / (2 * a) * (la**2 - lc**2 + a**2)
            oh = math.sqrt(lb**2 - om**2 - (on - a)**2)
            res = [on, om, oh]

        elif (a**2 + la**2 - lb**2 < 0) and (a**2 + la**2 - lc**2 < 0) and \
                (a**2 + lb**2 - la**2 > 0) and (a**2 + lc**2 - la**2 > 0):

            print("case 7, la %f lb %f lc %f" % (la, lb, lc))
            on = 1 / (2 * a) * (lb**2 - la**2 - a**2)
            om = 1 / (2 * a) * (lc**2 - la**2 - a**2)
            oh = math.sqrt(la**2 - on**2 - om**2)
            res = [-on, -om, oh]

        elif (a**2 + la**2 - lc**2 < 0) and (a**2 + lc**2 - la**2 > 0) and \
                (a**2 + la**2 - lb**2 > 0) and (a**2 + lb**2 - la**2 > 0):

            print("case 8, la %f lb %f lc %f" % (la, lb, lc))
            on = 1 / (2 * a) * (la**2 - lb**2 + a**2)
            om = 1 / (2 * a) * (lc**2 - la**2 - a**2)
            oh = math.sqrt(la**2 - on**2 - om**2)
            res = [on, -om, oh]

        elif (a**2 + lb**2 - la**2 < 0) and (a**2 + la**2 - lc**2 < 0) and \
                (a**2 + la**2 - lb**2 > 0) and (a**2 + lc**2 - la**2 > 0):

            print("case 9, la %f lb %f lc %f" % (la, lb, lc))
            on = a + 1 / (2 * a) * (la**2 - lb**2 - a**2)
            om = 1 / (2 * a) * (lc**2 - la**2 - a**2)
            oh = math.sqrt(lb**2 - (on - a)**2 - om**2)
            res = [on, -om, oh]
        else:
            print("No case matched")
            res = [-1, -1, -1]
    except:
        print("failed to calc")
        res = [-1, -1 , -1]


    stop = time.time()
    print("pyramid calc time {}".format(stop - start))

    return res

avg_rate = avg_rate()
navigator = drone_navigator(cfg.LANDING_X, cfg.LANDING_Y)
plot_sock = create_plot_sock()

navigator.start()

def filter_dist(loc):
    for anch in loc["anchors"]:
        addr = anch["dist"]["addr"]

        dist = anch["dist"]["dist"]
        anch_len_log[addr].add_to_filter(dist, ts)

        print("dist before filtering %.4f" % dist)
        anch["dist"]["dist"] = anch_len_log[addr].get_last_filtered()
        print("dist after filtering %.4f" % anch["dist"]["dist"])

while True:
    print(">> get location from anchors")

    loc, parrot_data = get_dwm_location_or_parrot_data()

    print(">> got calculated position from the engine")

    x = loc['calc_pos']['x']
    y = loc['calc_pos']['y']
    z = loc['calc_pos']['z']
    qf = loc['calc_pos']['qf']
    ts = loc["ts"]

    parrot_alt = 0

    print(">> get distances")
    filter_dist(loc)

    if not assigned:
        X0 = np.abs(np.array([x, y, z]))
        assigned = True

    received_anch = []
    for anch in loc["anchors"]:
        received_anch.append(anch['dist']['addr'])

    if (0x28b9 in received_anch) and \
        (0x262d in received_anch) and \
        (0x260f in received_anch):
            X_calc_pyramid = calc_pos_pyramid(loc)
    else:
        print("did not receive 3 anchors")
        continue

    #X_calc = calc_pos(X0, loc)
    X_calc = X_calc_pyramid

    X0 = X_calc
    X_lse.append(X_calc[0])
    Y_lse.append(X_calc[1])
    #Z_lse.append(X_calc[2])
    if parrot_data is not None and (ts - parrot_data["ts"] < 2):
        parrot_alt = parrot_data["alt"]
        Z_lse.append(parrot_data["alt"])
    else:
        Z_lse.append(X_calc[2])
    T.append(ts)

    while ts - T[0] > hist_len_sec:
        X_lse.pop(0)
        Y_lse.pop(0)
        Z_lse.pop(0)
        T.pop(0)

    apply_filter = 2

    if apply_filter:
        moving_window = 15

        if len(X_lse) < moving_window:
            continue

        if apply_filter == 1:
            X_filtered = savgol_filter(X_lse, moving_window, 5, mode="nearest")
            Y_filtered = savgol_filter(Y_lse, moving_window, 5, mode="nearest")
            Z_filtered = savgol_filter(Z_lse, moving_window, 5, mode="nearest")
        else:
            X_filtered = uniform_filter1d(X_lse, size=moving_window, mode="reflect")
            Y_filtered = uniform_filter1d(Y_lse, size=moving_window, mode="reflect")
            Z_filtered = uniform_filter1d(Z_lse, size=moving_window, mode="reflect")
    else:
        X_filtered = X_lse
        Y_filtered = Y_lse
        Z_filtered = Z_lse

    xf = X_filtered[-1]
    yf = Y_filtered[-1]
    zf = Z_filtered[-1]

    f_pos = func1(np.array([x, y, z]), loc)
    c_pos = func1([xf, yf, zf], loc)
    print("POS: %.2f %.2f %.2f" % (x, y , z), " C : %.2f %.2f %.2f" % (xf, yf, zf),
          " func(pos): %.4f" % f_pos, " func1(X_calc): %.4f" % c_pos)

    f_pos_norm = np.linalg.norm(f_pos)
    c_pos_norm = np.linalg.norm(c_pos)
    total_pos += f_pos_norm
    total_calc += c_pos_norm

    print("norm f(pos): ", f_pos_norm, " norm f(X_calc): ", c_pos_norm)
    print("total pos norm: ", total_pos, " total calc norm: ", total_calc)

    # Calculate update rate
    rate = avg_rate()

    # PID control
    navigator.navigate_drone(xf, yf)

    # Send all math output to the plot
    ts = time.time()

    send_plot_data(plot_sock, xf, yf, zf, parrot_alt, ts, rate,
                   len(loc['anchors']), navigator, loc)
