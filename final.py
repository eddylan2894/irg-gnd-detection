import pyrealsense2 as rs
import numpy as np
import cv2
import pcl
import math

M_PI = 3.141592

def points_to_pcl(points): 
	cloud = pcl.PointCloud()
	vtx = np.asanyarray(points.get_vertices())
	point_array = np.zeros((len(vtx), 3), np.float32)
	
	for i in range(len(vtx)):
		point_array[i][0] = np.float(vtx[i][0])
		point_array[i][1] = np.float(vtx[i][1])
		point_array[i][2] = np.float(vtx[i][2])
	if False:
		point_array = point_array.reshape(240, 424,3)
		cloud.from_3d_array(point_array)
	else:
		cloud.from_array(point_array)
	print(cloud.height, cloud.width)
	return cloud

def region_growing_segmentation(cloud, organized=False):
	if not organized:
		print("Making Kd Tree")
		tree = cloud.make_kdtree()
   
	ne = cloud.make_NormalEstimation()
	if not organized:
		ne.set_SearchMethod(tree)
	ne.set_RadiusSearch(0.05)
	print("Computing normals")
	# normals
	cloud_normals = ne.compute()

	print("Applying RegionGrowing")
	reg = cloud.make_RegionGrowing(ksearch=5)
	reg.set_MinClusterSize(1000)
	reg.set_NumberOfNeighbours(5)
	reg.set_SmoothnessThreshold(5.0 / 180.0 * M_PI)
	reg.set_CurvatureThreshold(2.0)
	reg.set_InputNormals(cloud_normals)
	print("Extracting")
	cluster_indices = reg.Extract()
	return cloud_normals, cluster_indices

def mark_pixels(image, width, indices):
	MARK_COLOR = np.array([255, 255, 255])
	res_image = image.copy()
	for idx in indices:
		i = idx % width
		j = idx // width
		res_image[j, i] = MARK_COLOR
	return res_image

def compute_mean_normal(indices, cloud_normals, count):
	s = np.zeros(4)
	for i in np.random.choice(indices, count):
		s += cloud_normals[i]
	return s / count

def choose_flat_plane(cluster_indices, cloud_normals, threshold=0.95, num_points=200):
	idx = -1
	mean_normals = []
	for i, indices in enumerate(cluster_indices):
		mn = compute_mean_normal(indices, cloud_normals, num_points)
		if abs(mn[1]) > threshold:
			idx = i
			break
		mean_normals.append(mn)
	if idx == -1:
		print("no ground found")
		print(mean_normals)
		arr = np.asarray(mean_normals)
		y_values = np.abs(arr[:, 1].squeeze())
		idx = np.argmax(y_values)
		print("use idx: ",idx)
	return idx
def fit_plane(plane_points):
	A = plane_points[:, :2]
	# https://data100.datahub.berkeley.edu/user/yclan2/notebooks/fa19/hw/hw7/hw7.ipynb
	A = np.hstack([A, np.ones([len(A), 1])])
	# A.shape
	b = plane_points[:, 2].reshape(-1, 1)

	fit = np.linalg.inv(A.T @ A) @ A.T @ b
	fit = np.squeeze(fit)
	plane_coef = np.insert(fit, 2, -1)
	return plane_coef

def shortest_distance(point, plane_coef):
	# point: (x, y, z), plane:ax+by+cz+d=0
	x1, y1, z1 = point
	a, b, c, d = plane_coef
	d = abs((a * x1 + b * y1 + c * z1 + d))
	e = math.sqrt(a * a + b * b + c * c)
	return d / e

def find_close_points(cloud, plane_coef, threshold=0.02):
	indices = []
	for i, point in enumerate(cloud):
		distance = shortest_distance(point, plane_coef)
		if distance < threshold:
			indices.append(i)
	return indices

def image_post_process(img, k_size=30):
	kernel = np.ones((k_size,k_size), np.uint8) 
	img_dilation = cv2.dilate(img, kernel, iterations=3) 
	img_processed = cv2.erode(img_dilation, kernel, iterations=3) 
	return img_processed
def main():
	# Declare pointcloud object, for calculating pointclouds and texture mappings
	pc = rs.pointcloud()
	# We want the points object to be persistent so we can display the last cloud when a frame drops
	points = rs.points()
	pipe = rs.pipeline()
	cfg = rs.config()
	rs.config.enable_device_from_file(cfg, './20191211_160154.bag', repeat_playback = False)
	cfg.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 6) # color camera
	cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6) # depth camera
	# pipe.start(cfg)
	profile = pipe.start(cfg)
	playback = profile.get_device().as_playback()
	playback.set_real_time(False)
	# define Filters
	thres_filter = rs.threshold_filter()
	depth_to_disparity = rs.disparity_transform(True)
	spat_filter = rs.spatial_filter()
	temp_filter = rs.temporal_filter()
	# hole_fill_filter = rs.hole_filling_filter()
	disparity_to_depth = rs.disparity_transform(False)
	
	i = 0
	while True:
		try:
			frames = pipe.wait_for_frames()
		except:
			break
		print(i)		
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()

		# process depth image
		if True:
			depth_frame = thres_filter.process(depth_frame)
			depth_frame = depth_to_disparity.process(depth_frame)
			depth_frame = spat_filter.process(depth_frame)
			depth_frame = temp_filter.process(depth_frame)
			# depth_frame = hole_fill_filter.process(depth_frame)
			depth_frame = disparity_to_depth.process(depth_frame)

		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

		points = pc.calculate(depth_frame)
		cloud = points_to_pcl(points)
		cloud_normals, cluster_indices = region_growing_segmentation(cloud)
		
		idx = choose_flat_plane(cluster_indices, cloud_normals)
		chosen_plane_indices = cluster_indices[idx]
		arr = cloud.to_array()
		plane_points = arr[chosen_plane_indices]
		plane_coef = fit_plane(plane_points)
		plane_filtered_indices = find_close_points(cloud, plane_coef)
		processed = mark_pixels(depth_colormap, 424, plane_filtered_indices)

		# Dilation
		img_processed = image_post_process(processed)
		cv2.imwrite("./images/proc_"+str(i)+".png", img_processed)
		i += 1
def save_images():
	pipe = rs.pipeline()
	cfg = rs.config()
	rs.config.enable_device_from_file(cfg, './20191211_160154.bag', repeat_playback = False)
	cfg.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 6) # color camera
	cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6) # depth camera
	# pipe.start(cfg)
	profile = pipe.start(cfg)
	playback = profile.get_device().as_playback()
	playback.set_real_time(False)
	
	i = 0
	while True:
		try:
			frames = pipe.wait_for_frames()
		except:
			break
		print(i)		
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		cv2.imwrite("./images/rgb/rgb_"+str(i)+".png", color_image)
		cv2.imwrite("./images/depth/depth_"+str(i)+".png", depth_colormap)
		i += 1
if __name__ == "__main__":
	# main()
	save_images()