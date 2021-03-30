import numpy as np
import matplotlib.pyplot as plt
# import open3d as o3d

from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.common.utils import boxes_to_sensor
from pynuscenes.utils.nuscenes_utils import global_to_vehicle
import pprint
import os

def showImage(array):
    """
    Creates an image of the heatmap and displays it as greyscale
    """
    f, axarr = plt.subplots(1,1)
    scaled = np.uint8(np.interp(array, (0, array.max()), (0,255)))
    axarr.imshow(scaled, cmap="gray")
            
    # axarr.scatter(maxima[:,1], maxima[:,0], c=maxima[:,2], cmap='Paired', marker='o', s=(72./f.dpi)**2)

    plt.show()

def showFrame3d(self, frame, results, camData):
    """
    Draws the radar pointcloud and the lidar pointcloud in open3d
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame['lidar']['pointcloud'].points[:3,:].T)
    pcd.paint_uniform_color(np.array([1,0,1])) ## blue
    results_pcd = o3d.geometry.PointCloud()
    # print(results.shape)
    results_pcd.points = o3d.utility.Vector3dVector(results[:,:3])
    results_pcd.paint_uniform_color(np.array([0,0,1]))

    rpcd = o3d.geometry.PointCloud()
    rpcd.points = o3d.utility.Vector3dVector(frame['radar']['pointcloud'].points[:3,:].T)
    rpcd.paint_uniform_color(np.array([1,0,0])) ## Red

    cpcd = o3d.geometry.PointCloud()
    ctArray = copy.deepcopy(camData) #np.array(self.centerTrack[frame['sample_token']])
    ctArray[:,2] = 0.5
    cpcd.points = o3d.utility.Vector3dVector(ctArray)
    cpcd.paint_uniform_color(np.array([0,1,0])) ## green
    
    vis.add_geometry(pcd)
    vis.add_geometry(rpcd)
    vis.add_geometry(cpcd)
    vis.add_geometry(results_pcd)
    ctr = vis.get_view_control()

    # ctr.set_up(np.array([1,0,0]))
    ctr.set_zoom(.2)
    ctr.translate(-40,10)
    
    vis.run()
    vis.destroy_window()
    
def saveFrame(array, frame, results, cameraPoints):
    raise Exception("stop. already generated figures -Landon")
    # _, ax = plt.subplots(1, 2, figsize=(9, 9))
    plt.figure(0,figsize=(4,4))
    lidar = frame['lidar']['pointcloud'].points[:3,:]
    # radar = frame['radar']['pointcloud'].points[:3,:]
    
    # radar = view_points(radar, np.eye(4), normalize=False)
    # scaled = np.uint8(np.interp(array, (0, array.max()), (0,255)))
    # plt.scatter(radar[0,:], radar[1,:], c=[0,0,1], s=.5)
    # plt.imshow(scaled, cmap='binary', origin='upper', extent=[-50,50,-50,50])
    # plt.xlim(-50,50)
    # plt.ylim(-50,50)
    # plt.axis('off')
    # plt.savefig("heatmap.png", format='png', bbox_inches='tight')
    # plt.savefig("heatmap.pdf", format='pdf', bbox_inches='tight')

    # plt.figure(1,figsize=(4,4))
    # lidar = view_points(lidar, np.eye(4), normalize=False)
    # plt.scatter(lidar[0,:], lidar[1,:], c=[.75,.75,.75],s=.2)
    # plt.scatter(results[:,0], results[:,1], c=[0,0,1], s=3)
    # plt.scatter(cameraPoints[:,0], cameraPoints[:,1], c=[1,0,0], s=3)
    # plt.plot(0,0,'x', color='black')
    # plt.xlim(-50,50)
    # plt.ylim(-50,50)
    # plt.axis('off')
    # plt.savefig("maxima_w_cam.png", format='png', bbox_inches='tight')
    # plt.savefig("maxima_w_cam.pdf", format='pdf', bbox_inches='tight')

    lidar = view_points(lidar, np.eye(4), normalize=False)
    plt.scatter(lidar[0,:], lidar[1,:], c=[.75,.75,.75],s=.2)
    plt.scatter(results[:,0], results[:,1], c=[0,.75,0], s=3)
    plt.plot(0,0,'x', color='black')
    plt.xlim(-50,50)
    plt.ylim(-50,50)
    plt.axis('off')
    plt.savefig("fusion_results.png", format='png', bbox_inches='tight')
    plt.savefig("fusion_results.pdf", format='pdf', bbox_inches='tight')

    # plt.show()


def showFrame(frame, results, velFactor=3, seq=0):
    # plt.figure(0,figsize=(4,4))

    fig, ax = plt.subplots(1,1,figsize=(4,4))
    lidar = frame['lidar']['pointcloud'].points[:3,:]
    view_points(lidar, np.eye(4), normalize=False)

    ax.scatter(lidar[0,:], lidar[1,:], c=[.75,.75,.75],s=.2)
    ax.plot(0,0,'x', color='black')
    ax.set_ylim(-50,50)
    ax.set_xlim(-50,50)
    ax.axis('off')
    # boxes = [global_to_vehicle(box['box_3d'],frame['ref_pose_record']) for box in frame['anns']]
    for box in frame['anns']:
        box['box_3d'].render(ax, view=np.eye(4), colors=('g','g','g'), linewidth=1)
        ax.arrow(box['box_3d'].center[0], box['box_3d'].center[1], box['box_3d'].velocity[0]/velFactor, box['box_3d'].velocity[1]/velFactor, head_width=.2, head_length=.4, fc='green', ec='green', lw=.5)
    
    ax.scatter(results[:,0], results[:,1], c=[0,0,1], s=.2)
    for i in range(results.shape[0]):
        ax.arrow(results[i,0], results[i,1], results[i,3]/velFactor, results[i,2]/velFactor, head_width=.2, head_length=.4, fc='red', ec='red', lw=.5)


    fig.savefig(os.path.join('figures', str(seq).zfill(3) + '.pdf'), format='pdf', bbox_inches='tight')

def _getVelComp(vx,vy):
    angle = np.atan2(vx,vy)
    magnitude = np.sqrt(vx**2,vy**2)
    return angle,magnitude
