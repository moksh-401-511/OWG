import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from third_party.grconvnet.network.utils.dataset_processing.grasp import detect_grasps

warnings.filterwarnings("ignore")


def plot_results(
        fig,
        rgb_img,
        grasp_q_img,
        grasp_angle_img,
        depth_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output of a network
    :param fig: Figure to plot the output
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    #plt.ion()
    #plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')

    if depth_img is not None:
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(depth_img, cmap='gray')
        ax.set_title('Depth')
        ax.axis('off')

    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(rgb_img)

    # color_list = ['r', 'b', 'g', 'y', 'y', 'y', 'y', 'y']
    # grasp_list = ['best', 'second', 'third', 'rest']
    # i = 0
    for g in gs:        
        g.length=40
        g.plot(ax)

    ax.set_title('Grasp')
    ax.axis('off')

    ax = fig.add_subplot(2, 3, 4)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Quality')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 5)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 3, 6)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)

    # plt.pause(0.1)
    # fig.canvas.draw()
    #plt.close()
    return fig


def plot_grasp(
        fig,
        grasps=None,
        save=False,
        rgb_img=None,
        grasp_q_img=None,
        grasp_angle_img=None,
        no_grasps=1,
        grasp_width_img=None
):
    """
    Plot the output grasp of a network
    :param fig: Figure to plot the output
    :param grasps: grasp pose(s)
    :param save: Bool for saving the plot
    :param rgb_img: RGB Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    if grasps is None:
        grasps = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    plt.ion()
    plt.clf()

    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in grasps:
        g.plot(ax)
    ax.set_title('Grasp')
    ax.axis('off')

    # plt.pause(0.1)
    # fig.canvas.draw()
        

def save_results(rgb_img, grasp_q_img, grasp_angle_img, depth_img=None, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a network
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of network
    :param grasp_angle_img: Angle output of network
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of network
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    ax.set_title('RGB')
    ax.axis('off')
    #fig.savefig('example_imgs/rgb.png')

    if depth_img is not None and depth_img.any():
        fig = plt.figure(figsize=(10, 10))
        plt.ion()
        plt.clf()
        ax = plt.subplot(111)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')
        #fig.savefig('example_imgs/depth.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')
    fig.savefig('example_imgs/grasp.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Quality')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('example_imgs/quality.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('example_imgs/angle.png')

    fig = plt.figure(figsize=(10, 10))
    plt.ion()
    plt.clf()
    ax = plt.subplot(111)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')
    ax.axis('off')
    plt.colorbar(plot)
    fig.savefig('example_imgs/width.png')

    fig.canvas.draw()
    plt.close(fig)
