import socket
import struct
import select

PARROT_IP = "127.0.0.1"
PARROT_PORT = 5556

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5555

# Create DWM1001 sock and bind
dwm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
dwm_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
dwm_sock.bind((MCAST_GRP, MCAST_PORT))

# Join group
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
dwm_sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
dwm_sock.setblocking(0)


# Create parrot sock
parrot_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
parrot_sock.bind((PARROT_IP, PARROT_PORT))
parrot_sock.setblocking(0)

while True:
    readable, writable, exceptional = select.select([dwm_sock, parrot_sock], \
                                                    [], [])
    if dwm_sock in readable:
        # Location header
        fmt = "iiihhiii"
        sz = struct.calcsize(fmt)
        buf = dwm_sock.recv(sz, socket.MSG_PEEK)

        (x, y, z, pos_qf, pos_valid, ts_sec, ts_usec, nr_anchors) = struct.unpack(fmt, buf)

        print("ts:%ld.%06ld [%d,%d,%d,%u] " % (ts_sec, ts_usec, x, y, z, pos_qf), end='')

        # Skip size of the location header
        off = sz

        # Read the whole location packet
        fmt = "iiihhihh"
        sz = sz + struct.calcsize(fmt) * nr_anchors
        buf = sock.recv(sz)

        for i in range(0, nr_anchors):
            (x, y, z, pos_qf, pos_valid, dist, addr, dist_qf) = struct.unpack_from(fmt, buf, off)
            off += struct.calcsize(fmt)

            print("#%u) a:0x%08x [%d,%d,%d,%u] d=%u,qf=%u " % (i, addr, x, y, z, pos_qf, dist, dist_qf), \
                  end='')

        print()

    elif parrot_sock in readable:
        fmt = "iiffff"
        sz = struct.calcsize(fmt)
        buf = parrot_sock.recv(sz)

        sec, usec, alt, roll, pitch, yaw = struct.unpack(fmt, buf)
        print("ts:%ld.%06ld alt %f roll %f pitch %f yaw %f" % (sec, usec, alt, roll, pitch, yaw))
