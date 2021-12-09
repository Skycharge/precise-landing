# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# Copyright (c) 2008, Enthought, Inc.
# License: BSD Style.
import math

from numpy import arange, pi, cos, sin
import numpy as np
import socket
import struct
import select
import config as cfg

from traits.api import HasTraits, Range, Instance, Float,\
        on_trait_change, observe
from traitsui.api import View, Item, Group

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
                MlabSceneModel
from mayavi import mlab

from traits_futures.api import CallFuture, submit_call, TraitsExecutor

from plot import recv_math_output

def receive_nano_data(sock):
    # Wait for data
    select.select([sock], [], [], None)

    # 1 double, 17 floats, 3 int32, 4 unsigned shorts, 4 floats, 3 floats
    fmt = "ffffff"
    sz = struct.calcsize(fmt)

    #Suck everything, drop all packets except the last one
    buf = None
    dropped = -1
    while True:
        try:
            #buf = sock.recv(8192)
            buf = sock.recv(sz)
            dropped += 1
            continue
        except:
            # EAGAIN
            break

    #data = struct.unpack(fmt, buf[-3 * sz:])
    data = struct.unpack(fmt, buf)
    return *data, dropped

def plot3d_axes(fig):
    xx = yy = zz = np.arange(0.0, 5.0, 0.5)
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)
    mlab.plot3d(yx, yy, yz, line_width=0.01, tube_radius=0.01)
    mlab.plot3d(zx, zy, zz, line_width=0.01, tube_radius=0.01)
    mlab.plot3d(xx, xy, xz, line_width=0.01, tube_radius=0.01)

    mlab.text3d(3, 0, 0, "X", orient_to_camera=True, scale=0.2)
    mlab.text3d(0, 3, 0, "Y", orient_to_camera=True, scale=0.2)
    mlab.text3d(0, 0, 3, "Z", orient_to_camera=True, scale=0.2)
    mlab.text3d(0, 0, 0, '0', orient_to_camera=True, scale=0.2)

def plot_table():
    x, y = np.mgrid[0:0.6:2j, 0:0.6:2j]
    z = np.zeros(x.shape)
    mlab.mesh(x, y , z, color=(0, 1, 0))


class MyModel(HasTraits):
    #: The executor to submit tasks to.
    traits_executor = Instance(TraitsExecutor)

    #: The future object returned on task submission.
    future = Instance(CallFuture)
    nano_future = Instance(CallFuture)

    #x_angle = Range(0, 360, 0, mode='slider')
    # y_angle = Range(0, 360, 0, mode='slider')
    # z_angle = Range(0, 360, 0, mode='slider')
    pitch = Float(0)
    roll = Float(0)
    yaw = Float(0)


    x_pos = Float(0)
    y_pos = Float(0)
    z_pos = Float(0)

    scene = Instance(MlabSceneModel, ())

    plot = Instance(PipelineBase)

    def __init__(self, traits_executor):
        HasTraits.__init__(self)
        self.traits_executor = traits_executor
        self.sock = sock
        self.roll = 0
        self.yaw = 0
        self.pitch = 0
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0

        self.future = submit_call(self.traits_executor, recv_math_output, sock)

    @observe("future:done")
    def _report_result(self, event):
        future = event.object
#         (acc_x, acc_y, acc_z, yaw, pitch, roll, dropped) = future.result
        (ts, x, y, z, parrot_alt, rate, xKp, xKi, xKd, xp, xi, xd,
         yKp, yKi, yKd, yp, yi, yd, roll, pitch, nr_anchors,
         addr1, addr2, addr3, addr4, dist1, dist2, dist3, dist4,
         acc_x, acc_y, acc_z, yaw, pitch, roll,
         dropped) = future.result

        print("received input")
        self.yaw = math.degrees(yaw)
        self.pitch = math.degrees(pitch)
        self.roll = math.degrees(roll)

        self.x_pos = x
        self.y_pos = y
        self.z_pos = z
        self.update_plot()
        # self.future = submit_call(self.traits_executor, receive_nano_data, sock)

        self.future = submit_call(self.traits_executor, recv_math_output, sock)

    def plot_cube(self):
        self.plot = self.scene.mlab.points3d(self.x_pos, self.y_pos, self.z_pos,
                                             color=(0, 0, 1),  mode='cube', scale_factor=0.3, opacity=1)
        self.plot.actor.actor.origin = [self.x_pos, self.y_pos, self.z_pos]
        #self.plot.actor.actor.position = [self.x_pos, self.y_pos, self.z_pos]
        #1
        #self.plot.actor.actor.orientation = [self.pitch, self.roll, self.yaw]
        #2
        # self.plot.actor.actor.rotate_x(self.roll)
        # self.plot.actor.actor.rotate_y(self.pitch)
        # self.plot.actor.actor.rotate_z(self.yaw)

        #3
        self.plot.actor.actor.trait_set(orientation=[self.pitch, self.roll, self.yaw])

        #4
        # self.plot.glyph.glyph_source._trfm.transform.rotate_x(self.roll)
        # self.plot.glyph.glyph_source._trfm.transform.rotate_y(self.pitch)
        # self.plot.glyph.glyph_source._trfm.transform.rotate_z(self.yaw)


        ms = self.plot.mlab_source
        return


        # gs = self.plot.glyph.glyph_source
        # gs.glyph_source = gs.glyph_dict['axes']

    # When the scene is activated, or when the parameters are changed, we
    # update the plot.
    @on_trait_change('scene.activated')
    def init_plot(self):
        mlab.clf()
        mlab.view(90, 90)
        #self.scene.anti_aliasing_frames = 0
        # from tvtk.api import tvtk
        # fig = mlab.gcf()
        # fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
        self.plot_cube()
        plot3d_axes(None)
        plot_table()


    # @on_trait_change('x_angle,y_angle,z_angle,x_pos,y_pos,z_pos')
    def update_plot(self):
        #mlab.clf()
        ##################

        scene = self.scene.mlab.gcf()


        disable_render = scene.scene.disable_render
        scene.scene.disable_render = True
        mlab.clf()

        #scene.children[:] = []
        # scene._mouse_pick_dispatcher.clear_callbacks()
        #scene.scene.disable_render = disable_render
###############
        f = mlab.gcf()
        # cam = f.scene.camera

        plot3d_axes(None)
        #x, y, z, t = curve(self.x_angle, self.n_longitudinal)
        #fig = self.scene.mlab.gcf()
        # if self.plot is None:
        self.plot_cube()
        # self.plot.actor.actor.rotate_x(self.x_angle)
    # else:
    #     self.plot.actor.actor.origin = [self.x_pos, self.y_pos, self.z_pos]
        #self.plot.actor.actor.position = [self.x_pos, self.y_pos, self.z_pos]
        # self.plot.actor.actor.orientation = [self.y_deg, self.z_deg, self.x_deg]
                # self.plot.mlab_source.trait_set(x=x, y=y, z=z)
        plot_table()
        scene.scene.disable_render = disable_render

        mlab.draw()
        # f.scene.reset_zoom()


        #mlab.orientation_axes()
        ########################################


    # The layout of the dialog created
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=500, width=500, show_label=False),
                Group(
                        'pitch', 'roll', 'yaw', 'x_pos', 'y_pos', 'z_pos'
                     ),
                resizable=True,
                )
if __name__ == '__main__':
    # Create plot sock
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((cfg.UDP_PLOT_IP, cfg.UDP_PLOT_PORT))
    sock.setblocking(False)

    traits_executor = TraitsExecutor()

    my_model = MyModel(traits_executor=traits_executor)
    my_model.configure_traits()




