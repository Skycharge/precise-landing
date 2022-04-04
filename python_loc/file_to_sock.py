#!/usr/bin/env python3

"""File to socket

Usage:
  file_to_sock.py --file <file> [--trajectory]

Options:
  -h --help          Show this screen
  --file <file>      File with logged data, e.g. data/field-session-2/log.6.leastsq.manual
  --trajectory       File data is a true trajectory with x,y,z
"""

from docopt import docopt
import struct
import time
import socket
import ctypes

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5555
MULTICAST_TTL = 2

PARROT_IP = "127.0.0.1"
PARROT_PORT = 5556

last_ts = 0
last_ts_s = 0
last_ts_us = 0

speed = 2

def send_parrot_data(line):
    fields = line[line.find("alt") : -1].split(" ")
    fmt = "iiffff"

    alt = float(fields[1])
    roll = float(fields[3])
    pitch = float(fields[5])
    yaw = float(fields[7])

    buff = ctypes.create_string_buffer(512)
    struct.pack_into(fmt, buff, 0, last_ts_s, last_ts_us, alt, roll, pitch, yaw)
    parot_sock.sendto(buff, (PARROT_IP, PARROT_PORT))

def send_dwm_data(loc):
    nr_anchors = len(loc['anchors'])

    fmt = "iiihhiii"
    buff = ctypes.create_string_buffer(512)

    pos = loc['pos']

    # (x, y, z, pos_qf, pos_valid, ts_sec, ts_usec, nr_anchors)
    struct.pack_into(fmt, buff, 0,
                     *pos['coords'], pos['qf'], pos['valid'],
                     loc['ts_s'], loc['ts_us'], nr_anchors)
    off = struct.calcsize(fmt)

    for anchor in loc['anchors']:
        fmt = "iiihhihh"
        coords = anchor["pos"]["coords"]

        # (x, y, z, pos_qf, pos_valid, dist, addr, dist_qf)
        struct.pack_into(fmt, buff, off,
                         *coords,
                         anchor["pos"]["qf"], anchor["pos"]["valid"],
                         anchor["dist"]["dist"], anchor["addr"],
                         anchor["dist"]["qf"])
        off += struct.calcsize(fmt)

    sock.sendto(buff, (MCAST_GRP, MCAST_PORT))

if __name__ == '__main__':
    args = docopt(__doc__)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
    parot_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data_file = open(args['--file'], 'r')

    while True:
        line = data_file.readline()
        if not line:
            break

        if line.startswith("## get parrot data:"):
            send_parrot_data(line)
            continue

        if not line.startswith("ts:"):
            continue
        line = line.replace(" ", "")

        ts = line[line.find(":")+1 : line.find("[")]
        ts_s , ts_us = ts.split(".")
        ts_s = int(ts_s)
        ts_us = int(ts_us)

        vals = line[line.find("[")+1 : line.find("]")]
        vals = vals.split(",")

        # Be aware that we have all saved coords and distances in mm
        loc = {
            'pos': {
                'coords': [int(vals[0]), int(vals[1]), int(vals[2])],
                'qf': int(vals[3]),
                'valid': True
            },
            'ts_s': ts_s,
            'ts_us': ts_us,
        }

        anchor_cnt = 0
        start = 0
        anchors = []

        while (anchor_cnt < 4):
            start = line.find("a:", start)
            if start == -1:
                break

            addr = line[start+2 : line.find("[", start)]
            addr = int(addr, base=16)

            anchor_pos = line[line.find("[", start)+1 : line.find("]", start)]
            x, y, z, pos_qf = anchor_pos.split(",")
            pos_qf = int(pos_qf)

            dist = line[line.find("d=", start)+2 : line.find(",qf=", start)]
            dist = int(dist)

            dist_qf = line[line.find("qf=", start)+3 : line.find("#", start)]
            dist_qf = int(dist_qf)

            # Be aware that we have all saved coords and distances in mm
            anchor = {
                'addr': addr,
                'pos': {
                    'coords':  [int(x), int(y), int(z)],
                    'qf': loc['pos']['qf'],
                    'valid': loc['pos']['valid'],
                },
                'dist': {
                    'dist': int(dist),
                    'qf': dist_qf
                },
            }

            anchors.append(anchor)
            start += 1
            anchor_cnt += 1

        loc['anchors'] = anchors
        send_dwm_data(loc)

        if last_ts != 0:
            delay = (float(ts) - last_ts) / speed
            print("cur time: {} delay: {}".format(float(ts), delay))
            time.sleep(delay)
            last_ts = float(ts)
            last_ts_s = ts_s
            last_ts_us = ts_us

        else:
            last_ts = float(ts)
