import csv
import numpy as np
import matplotlib.pyplot as plt

data_root = 'KITTI_sequence_1'
est_pose_path = 'build/pose.csv'
gt_pose_path = data_root + '/poses.txt'

est_poses = []
est_path  = []
with open(est_pose_path, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        print(row)
        est_pose = np.array([[float(row[i]) for i in range(len(row))]])
        est_pose = np.reshape(est_pose, (4, 4))
        est_poses.append(est_pose)
        est_path.append([est_pose[0, 3], est_pose[2, 3]])

gt_poses = []
gt_path  = []
with open(gt_pose_path, 'r') as f:
    for line in f.readlines():
        gt_pose = np.fromstring(line, dtype = np.float64, sep = ' ')
        gt_pose = np.reshape(gt_pose, (3, 4))
        gt_poses.append(gt_pose)
        gt_path.append([gt_pose[0, 3], gt_pose[2, 3]])

gt_path = np.array(gt_path)
est_path = np.array(est_path)

plt.plot(gt_path[:, 0], gt_path[:, 1], '.r')
plt.plot(est_path[:, 0], est_path[:, 1], '.b')

plt.xlim([-50, 50])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['ground truth', 'estimation'])
plt.show()
