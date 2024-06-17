import collections
import trimesh as tm
import open3d as o3d
from open3d.visualization import rendering
import os
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import wandb
sys.path.append(os.path.join(os.getcwd(), 'lib')) # HACK add the lib folder
from lib.config import CONF
from lib.sepdataset import Answer
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random
from scipy.stats import norm as scipynorm

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

all_annotations = \
    json.load(open(os.path.join(CONF.PATH.SQA, 'SQA_train.json'), 'r')) + \
    json.load(open(os.path.join(CONF.PATH.SQA, 'SQA_val.json'), 'r')) + \
    json.load(open(os.path.join(CONF.PATH.SQA, 'SQA_test.json'), 'r'))

qid2annoid = {}
for i in range(len(all_annotations)):
    qid2annoid[all_annotations[i]["question_id"]] = i

answer_counter_list = json.load(open(os.path.join(CONF.PATH.SQA, "answer_counter.json")))
answer_counter = collections.Counter(sorted(answer_counter_list))
answer_max_size = len(answer_counter)
answer_counter = dict([x for x in answer_counter.most_common()[:answer_max_size] if x[1] >= 1])
answer_cands = sorted(answer_counter.keys())
answer_vocab = Answer(answer_cands)

def create_arrow():
    cone = tm.creation.cone(radius=0.15, height=0.30, sections=None, transform=None)
    rotate_around_y = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    cone = cone.apply_transform(rotate_around_y)
    cylinder = tm.creation.cylinder(radius=0.09, height=0.45, sections=None, segment=None, transform=None)
    cylinder = cylinder.apply_transform(rotate_around_y)
    mv_2_head = np.array([[1, 0, 0, -0.15], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    cone = cone.apply_transform(mv_2_head)
    arrow = tm.util.concatenate([cone, cylinder])
    return arrow

def update_arrow_pos_rot_color(arrow, position, rotation, color, situation_loss_tag):
    if '__quat__' in situation_loss_tag:
        quaternion = rotation
        temp = R.from_quat(quaternion)
    elif '__angle__' in situation_loss_tag:
        angle = np.arctan2(rotation[0], rotation[1])
        temp = R.from_rotvec(np.array([0, 0, angle]))
    else:
        raise ValueError("Unknown situation loss tag: {}".format(situation_loss_tag))
    rot_mat_3x3 = temp.as_matrix()
    rotation_matrix = np.array([[rot_mat_3x3[0][0], rot_mat_3x3[0][1], rot_mat_3x3[0][2], 0],
                                [rot_mat_3x3[1][0], rot_mat_3x3[1][1], rot_mat_3x3[1][2], 0],
                                [rot_mat_3x3[2][0], rot_mat_3x3[2][1], rot_mat_3x3[2][2], 0],
                                [0, 0, 0, 1]])
    transformation_matrix = np.array([[1, 0, 0, position[0]],
                                        [0, 1, 0, position[1]],
                                        [0, 0, 1, position[2]],
                                        [0, 0, 0, 1]])
    this_arrow = arrow.apply_transform(rotation_matrix)
    this_arrow = this_arrow.apply_transform(transformation_matrix)
    this_arrow.visual.vertex_colors = np.zeros((100, 4))
    this_arrow.visual.vertex_colors[:, 0] = color[0]
    this_arrow.visual.vertex_colors[:, 1] = color[1]
    this_arrow.visual.vertex_colors[:, 2] = color[2]
    this_arrow.visual.vertex_colors[:, 3] = color[3]
    
    return this_arrow


def visualize_scene_wandb(data_dict, scene_ids, stamp, phase, global_iter_id, use_answer, situation_loss_tag):
    scene_id = scene_ids[0]
    ply_root = CONF.PATH.SCANNET_SCANS
    
    # setup open3d renderer
    w, h = 640, 480
    render = rendering.OffscreenRenderer(w, h)
    render.scene.set_background([0.0, 0.0, 0.0, 0.0])
    render.scene.view.set_post_processing(False)
    render.scene.show_axes(False)
    render.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)
    render.scene.scene.enable_sun_light(True)
    aspect = h/w
    s = 1.4
    render.scene.camera.set_projection(rendering.Camera.Projection.Ortho, -s, s, -s*aspect, s*aspect, 0.1, 200)
    render.scene.camera.look_at([0, 0, 0], [0, 0, 50], [0, 0, 1]) # [0, 0, 0], [-30, -30, 100], [0, 0, 1]

    # load text info (answer, situation, question)
    answer = data_dict["answers"][0][0]
    if use_answer:
        answer_pd = data_dict["answer_scores"][0]
        answer_pd_idx = np.argmax(answer_pd.detach().cpu().numpy())
        answer_pd_str = answer_vocab.itos(answer_pd_idx)
    situation = data_dict["situation"][0]
    question = data_dict["question"][0]
    question_id = data_dict["question_id"][0].item()
    visualization_root = os.path.join(CONF.PATH.OUTPUT, stamp, 'visualization_{}_{}'.format(phase, str(global_iter_id)))
    os.makedirs(os.path.join(visualization_root, 'png'), exist_ok=True)
    os.makedirs(os.path.join(visualization_root, 'obj'), exist_ok=True)
    output_file_obj = os.path.join(visualization_root, 'obj', str(question_id) + ".obj")
    output_file_png = os.path.join(visualization_root, 'png', str(question_id) + ".png")

    # load scene mesh
    ply_path = os.path.join(ply_root, scene_id, scene_id+"_vh_clean_2.ply")
    meta_file = open(os.path.join(ply_root, scene_id, scene_id+".txt")).readlines()
    axis_align_matrix = None
    for line in meta_file:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4)) if axis_align_matrix is not None else np.eye(4)
    scene = tm.load(ply_path)
    scene = scene.apply_transform(axis_align_matrix)
    # translate scene minumim of x and y to 0
    scene_min = np.array(scene.bounds[0]) # (x, y, z)
    scene_min[2] = 0
    scene = scene.apply_translation(-scene_min) 
    
    # load GT arrow mesh
    arrow = create_arrow() # ground truth arrow
    position = data_dict["auxiliary_task"][0][:3].tolist()
    rotation = data_dict["auxiliary_task"][0][3:].tolist()
    arrow = update_arrow_pos_rot_color(arrow, position, rotation, 
                                    color=[0, 205, 108, 255], 
                                    situation_loss_tag=situation_loss_tag)

    # load Pred arrow mesh
    arrow_pd = create_arrow() 
    position_pd = data_dict["aux_scores"][0][:3].tolist()
    rotation_pd = data_dict["aux_scores"][0][3:].tolist()
    arrow_pd = update_arrow_pos_rot_color(arrow_pd, position_pd, rotation_pd, 
                                            color=[0, 154, 222, 255],  # 255, 31, 91
                                            situation_loss_tag=situation_loss_tag)

    whole_scene = tm.util.concatenate([scene, arrow, arrow_pd])
    if CONF.VISUALIZATION.SAVEOBJ:
        whole_scene.export(output_file_obj)

    # prepare the scene visualization
    o3d_mesh = whole_scene.as_open3d
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(whole_scene.visual.vertex_colors[:, :3] / 255)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    # scale the mesh apporximately to 1
    bound = o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound()
    max_bound = np.max(bound)
    o3d_mesh.scale(1.8 / max_bound, center=[0, 0, 0])
    # define material
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.shader = "defaultUnlit" # "defaultLit" 

    # translate the o3d_mesh to the center of the bounding box
    o3d_center = o3d_mesh.get_center()
    o3d_mesh.translate(-o3d_center)
    render.scene.add_geometry('3d_scene', o3d_mesh, mtl, True)

    # prepare the activation points visualization
    points = data_dict["scene_positions"][0].detach().cpu().numpy() # (256, 2)
    activation = data_dict["att_feat_ori"][0].detach().cpu() # (256, 768)
    activation -= activation.mean(dim=0, keepdim=True)
    activation = activation.norm(dim=-1).numpy() # (256, 1)
    activation = (activation - activation.min()) / (activation.max() - activation.min())
    # normalize, and add a z-axis value (the max of o3d_mesh z-axis value) to the points
    points *= 1.8 / max_bound
    pcd = o3d.geometry.PointCloud()
    point_z_height = o3d_mesh.get_max_bound()[2] 
    points = np.concatenate([points, np.ones((points.shape[0], 1)) * point_z_height], axis=1)
    pcd.points = o3d.utility.Vector3dVector(points)
    # define material
    mtl_points = o3d.visualization.rendering.MaterialRecord()
    mtl_points.shader = "defaultUnlit"
    # mtl_points.base_color = [1.0, 1.0, 1.0, 0.8]  # RGBA
    # mtl_points.shader = "defaultLitTransparency" 
    # mtl_points.base_roughness = 0.0
    # mtl_points.base_reflectance = 0.0
    # mtl_points.base_clearcoat = 1.0
    # mtl_points.thickness = 1.0
    # mtl_points.transmission = 1.0
    # mtl_points.absorption_distance = 10
    # mtl_points.absorption_color = [0.5, 0.5, 0.5]
    norm = plt.Normalize(activation.min(), activation.max())
    colors = plt.cm.viridis(norm(activation))
    colors = colors[:, :3]
    sphere_radius = 0.015

    pcd.translate(-o3d_center)
    
    spheres_mesh = point_cloud_to_spheres(pcd, colors, sphere_radius=0.015)
    render.scene.add_geometry('points', spheres_mesh, mtl_points, True)

    # render the image and add text
    img = render.render_to_image()
    img = np.asarray(img)

    if use_answer:
        info_text = 'QID: {}  GT Ans: {}  PD Ans: {}'.format(str(question_id), str(answer), str(answer_pd_str))
    else:
        info_text = 'QID: {}  GT Ans: {}'.format(str(question_id), str(answer))
    question_text = 'Q: {}'.format(question)
    situation_text = 'S: {}'.format(situation)
    image = Image.fromarray(img)
    font = ImageFont.truetype("./utils/NotoSans-Regular.ttf", 11)

    bbox0 = ImageDraw.Draw(image).textbbox((0, 0), info_text, font=font)
    bbox1 = ImageDraw.Draw(image).textbbox((0, 0), question_text, font=font)
    bbox2 = ImageDraw.Draw(image).textbbox((0, 0), situation_text, font=font)
    info_height = bbox0[3] - bbox0[1]
    question_height = bbox1[3] - bbox1[1]
    situation_height = bbox2[3] - bbox2[1]

    # Create a new image with space at the bottom for the two lines of text
    new_image_height = image.height + int(info_height) + int(question_height) + int(situation_height) + 20
    new_image = Image.new("RGB", (image.width, new_image_height), (0, 0, 0))
    new_image.paste(image, (0, 0))

    # Add the texts to the new image
    draw = ImageDraw.Draw(new_image)
    draw.text((10, image.height + 5), info_text, font=font, fill=(255, 255, 255))
    draw.text((10, image.height + int(info_height) + 10), question_text, font=font, fill=(255, 255, 255))
    draw.text((10, image.height + int(info_height) + int(question_height) + 15), situation_text, font=font, fill=(255, 255, 255))

    # Save the new image
    new_image.save(output_file_png)
    wandb.log({"{}/visualization/{}".format(phase, str(question_id)): wandb.Image(new_image)}, step=global_iter_id)

    render.scene.remove_geometry("3d_scene")
    render.scene.remove_geometry("points")


def point_cloud_to_spheres(pcd, colors, sphere_radius=0.05):
    # List to hold all the sphere meshes
    sphere_list = []
    # For each point in the point cloud, translate the sphere to that point's location and color it
    for point, color in zip(pcd.points, colors):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere_copy = sphere.translate(point)
        sphere_copy.paint_uniform_color(color)
        sphere_list.append(sphere_copy)
    
    # Combine all the spheres into a single mesh for efficient rendering
    combined_mesh = o3d.geometry.TriangleMesh()
    for s in sphere_list:
        combined_mesh += s
    
    return combined_mesh


def visualize_scene_test(data_dict, scene_id, situation_loss_tag, visualization_root, idx, activation, feat_name, points, VIS_CONF, render):
    ply_root = CONF.PATH.SCANNET_SCANS
    render.scene.camera.look_at(VIS_CONF.LOOK_AT[0], VIS_CONF.LOOK_AT[1], VIS_CONF.LOOK_AT[2]) 

    # load text info (answer, situation, question)
    answer = data_dict["answers"][0][0]
    answer_pd = data_dict["answer_scores"][0]
    answer_pd_idx = np.argmax(answer_pd.detach().cpu().numpy())
    answer_pd_str = answer_vocab.itos(answer_pd_idx)

    situation = data_dict["situation"][0]
    question = data_dict["question"][0]
    question_id = data_dict["question_id"][0].item()
    output_file_png = os.path.join(visualization_root, str('{:04d}_{}.png'.format(idx, feat_name)))
    output_file_png_minus = os.path.join(visualization_root, str('{:04d}_{}_minus.png'.format(idx, feat_name)))
    output_file_obj = os.path.join(visualization_root, str('{:04d}_{}.obj'.format(idx, feat_name)))
    output_file_video = os.path.join(visualization_root, str('{:04d}_{}'.format(idx, feat_name)))
    

    # load scene mesh
    ply_path = os.path.join(ply_root, scene_id, scene_id+"_vh_clean_2.ply")
    meta_file = open(os.path.join(ply_root, scene_id, scene_id+".txt")).readlines()
    axis_align_matrix = None
    for line in meta_file:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4)) if axis_align_matrix is not None else np.eye(4)
    scene = tm.load(ply_path)
    scene = scene.apply_transform(axis_align_matrix)
    # translate scene minumim of x and y to 0
    scene_min = np.array(scene.bounds[0]) # (x, y, z)
    scene_min[2] = 0
    scene = scene.apply_translation(-scene_min) 

    # load GT arrow mesh
    arrow = create_arrow() 
    position = data_dict["auxiliary_task"][0][:3].tolist()
    rotation = data_dict["auxiliary_task"][0][3:].tolist()
    arrow = update_arrow_pos_rot_color(arrow, position, rotation, 
                                    color=[255, 31, 91, 255], # Green: 0, 205, 108  Red: 255, 31, 91
                                    situation_loss_tag=situation_loss_tag)

    # load Pred arrow mesh
    arrow_pd = create_arrow() 
    position_pd = data_dict["aux_scores"][0][:3].tolist()
    rotation_pd = data_dict["aux_scores"][0][3:].tolist()
    arrow_pd = update_arrow_pos_rot_color(arrow_pd, position_pd, rotation_pd, 
                                            color=[0, 154, 222, 255],  #  Blue: 0, 154, 222
                                            situation_loss_tag=situation_loss_tag)

    if VIS_CONF.GT_SVEC and VIS_CONF.PRED_SVEC:
        whole_scene = tm.util.concatenate([scene, arrow, arrow_pd])
    elif VIS_CONF.GT_SVEC:
        whole_scene = tm.util.concatenate([scene, arrow])
    else:
        whole_scene = tm.util.concatenate([scene])
    
    if CONF.VISUALIZATION.SAVEOBJ:
        whole_scene.export(output_file_obj)

    # prepare the scene visualization
    o3d_mesh = whole_scene.as_open3d
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(whole_scene.visual.vertex_colors[:, :3] / 255)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    # scale the mesh apporximately to 1
    bound = o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound()
    max_bound = np.max(bound)
    o3d_mesh.scale(1.8 / max_bound, center=[0, 0, 0])
    # define material
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.shader = "defaultUnlit" 
    # translate the o3d_mesh to the center of the bounding box
    o3d_center = o3d_mesh.get_center()
    o3d_mesh.translate(-o3d_center)
    render.scene.add_geometry('3d_scene', o3d_mesh, mtl, True)
    # if required, prepare the activation points visualization
    if VIS_CONF.POINT:
        # normalize, and add a z-axis value (the max of o3d_mesh z-axis value) to the points
        points *= 1.8 / max_bound
        pcd = o3d.geometry.PointCloud()
        point_z_height = o3d_mesh.get_max_bound()[2] 
        points = np.concatenate([points, np.ones((points.shape[0], 1)) * point_z_height], axis=1)
        pcd.points = o3d.utility.Vector3dVector(points)
        # define material
        mtl_points = o3d.visualization.rendering.MaterialRecord()
        mtl_points.shader = "defaultUnlit"
        # mtl_points.base_color = [1.0, 1.0, 1.0, 0.8]  # RGBA
        # mtl_points.shader = "defaultLitTransparency" 
        # mtl_points.base_roughness = 0.0
        # mtl_points.base_reflectance = 0.0
        # mtl_points.base_clearcoat = 1.0
        # mtl_points.thickness = 1.0
        # mtl_points.transmission = 1.0
        # mtl_points.absorption_distance = 10
        # mtl_points.absorption_color = [0.5, 0.5, 0.5]
        if VIS_CONF.ACTIVATION:
            norm = plt.Normalize(activation.min(), activation.max())
            colors = plt.cm.viridis(norm(activation)) # viridis   cividis
            colors = colors[:, :3] # Remove alpha channel if it exists
            sphere_radius = 0.015
        else:
            colors = np.ones((points.shape[0], 3)) * 0.8
            sphere_radius=0.01
        
        pcd.translate(-o3d_center)
        spheres_mesh = point_cloud_to_spheres(pcd, colors, sphere_radius=sphere_radius)
        render.scene.add_geometry('points', spheres_mesh, mtl_points, True)

    # render the image and add text
    img = render.render_to_image()
    img = np.asarray(img)

    info_text = 'QID: {}  GT Ans: {}  PD Ans: {}'.format(str(question_id), str(answer), str(answer_pd_str))
    question_text = 'Q: {}'.format(question)
    situation_text = 'S: {}'.format(situation)
    image = Image.fromarray(img)
    font = ImageFont.truetype("./utils/NotoSans-Regular.ttf", 11)

    bbox0 = ImageDraw.Draw(image).textbbox((0, 0), info_text, font=font)
    bbox1 = ImageDraw.Draw(image).textbbox((0, 0), question_text, font=font)
    bbox2 = ImageDraw.Draw(image).textbbox((0, 0), situation_text, font=font)
    info_height = bbox0[3] - bbox0[1]
    question_height = bbox1[3] - bbox1[1]
    situation_height = bbox2[3] - bbox2[1]

    # Create a new image with space at the bottom for the two lines of text
    new_image_height = image.height + int(info_height) + int(question_height) + int(situation_height) + 20
    new_image = Image.new("RGB", (image.width, new_image_height), (0, 0, 0))
    new_image.paste(image, (0, 0))

    # Add the texts to the new image
    draw = ImageDraw.Draw(new_image)
    draw.text((10, image.height + 5), info_text, font=font, fill=(255, 255, 255))
    draw.text((10, image.height + int(info_height) + 10), question_text, font=font, fill=(255, 255, 255))
    draw.text((10, image.height + int(info_height) + int(question_height) + 15), situation_text, font=font, fill=(255, 255, 255))

    # Save the new image
    new_image.save(output_file_png)

    # generate a new image with 180 degree view rotation
    render.scene.camera.look_at(VIS_CONF.LOOK_AT[0], [-VIS_CONF.LOOK_AT[1][0], VIS_CONF.LOOK_AT[1][1], VIS_CONF.LOOK_AT[1][2]], VIS_CONF.LOOK_AT[2]) 
    
    if VIS_CONF.GENERATE_MINUS:
        img = render.render_to_image()
        img = np.asarray(img)
        image = Image.fromarray(img)
        image.save(output_file_png_minus)

    # generate video
    if VIS_CONF.VIDEO:
        os.makedirs(output_file_video, exist_ok=True)
        # create linear interpolation of 360 degree
        yaw_seq = np.linspace(0, 360, 49)[:-1]
        for frame in tqdm.tqdm(range(48)):
            # turn elev and yaw to camera location with R
            cam_x = VIS_CONF.LOOK_AT[1][0] * np.cos(np.deg2rad(yaw_seq[frame]))
            cam_y = VIS_CONF.LOOK_AT[1][0] * np.sin(np.deg2rad(yaw_seq[frame]))
            cam_z = VIS_CONF.LOOK_AT[1][2]
            render.scene.camera.look_at([0, 0, 0], [cam_x, cam_y, cam_z], [0, 0, 1]) 

            # render the image and add text
            img = render.render_to_image()
            img = np.asarray(img)

            image = Image.fromarray(img)
            image.save(os.path.join(output_file_video, '%03d.png'%(frame)))


    render.scene.remove_geometry("3d_scene")
    if VIS_CONF.POINT:
        render.scene.remove_geometry("points")



def visualize_scene_test_good_res_video(data_dict, scene_id, situation_loss_tag, visualization_root, idx, activation, feat_name, points, VIS_CONF, render):
    ply_root = CONF.PATH.SCANNET_SCANS
    render.scene.camera.look_at(VIS_CONF.LOOK_AT[0], VIS_CONF.LOOK_AT[1], VIS_CONF.LOOK_AT[2]) 

    # load text info (answer, situation, question)
    answer = data_dict["answers"][0][0]
    answer_pd = data_dict["answer_scores"][0]
    answer_pd_idx = np.argmax(answer_pd.detach().cpu().numpy())
    answer_pd_str = answer_vocab.itos(answer_pd_idx)

    situation = data_dict["situation"][0]
    question = data_dict["question"][0]
    question_id = data_dict["question_id"][0].item()
    output_file_png = os.path.join(visualization_root, str('{:04d}_{}.png'.format(idx, feat_name)))
    output_file_png_minus = os.path.join(visualization_root, str('{:04d}_{}_minus.png'.format(idx, feat_name)))
    output_file_obj = os.path.join(visualization_root, str('{:04d}_{}.obj'.format(idx, feat_name)))
    output_file_video = os.path.join(visualization_root, str('{:04d}_{}'.format(idx, feat_name)))
    

    # load scene mesh
    ply_path = os.path.join(ply_root, scene_id, scene_id+"_vh_clean_2.ply")
    meta_file = open(os.path.join(ply_root, scene_id, scene_id+".txt")).readlines()
    axis_align_matrix = None
    for line in meta_file:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4)) if axis_align_matrix is not None else np.eye(4)
    scene = tm.load(ply_path)
    scene = scene.apply_transform(axis_align_matrix)
    # translate scene minumim of x and y to 0
    scene_min = np.array(scene.bounds[0]) # (x, y, z)
    scene_min[2] = 0
    scene = scene.apply_translation(-scene_min) 

    # load GT arrow mesh
    arrow = create_arrow() 
    position = data_dict["auxiliary_task"][0][:3].tolist()
    rotation = data_dict["auxiliary_task"][0][3:].tolist()
    arrow = update_arrow_pos_rot_color(arrow, position, rotation, 
                                    color=[255, 31, 91, 255], # Green: 0, 205, 108  Red: 255, 31, 91
                                    situation_loss_tag=situation_loss_tag)

    # load Pred arrow mesh: as updated GT arrow
    arrow_pd = create_arrow() 
    position_pd = data_dict["auxiliary_task"][0][:3].tolist()
    rotation_pd = data_dict["auxiliary_task"][0][3:].tolist()

    dist = np.linalg.norm(points[:, :2] - np.array(position_pd[:2]), axis=1)
    idx = np.argsort(dist)
    points_sorted = points[idx]
    random_idx = np.random.randint(0, 5)
    # print("Top 36 points: ", points_sorted[:36])
    # print("random_idx: ", random_idx)
    position_new = points_sorted[random_idx].tolist()
    position_new = position_new + [position_pd[2]]

    delta_theta = scipynorm.rvs(scale=0.15)
    q_noise = R.from_rotvec(delta_theta * np.array([0, 0, 1])).as_quat()
    # print("delta_theta: {:.3f}".format(delta_theta), ' In degree: {:.3f}'.format(delta_theta * 180 / np.pi))
    if len(rotation_pd) == 4:
        q_new = R.from_quat(np.array(rotation_pd)) * R.from_quat(q_noise)
        rotation_new = q_new.as_quat().tolist()
    if len(rotation_pd) == 2: # sin cos
        angle = np.arctan2(rotation_pd[0], rotation_pd[1])
        q_old = R.from_rotvec(np.array([0, 0, angle]))
        q_new = q_old * R.from_quat(q_noise)
        rotAngle = q_new.as_rotvec()[-1]
        rotation_new = [np.sin(rotAngle), np.cos(rotAngle)]

    arrow_pd = update_arrow_pos_rot_color(arrow_pd, position_new, rotation_new, 
                                            color=[0, 154, 222, 255],  #  Blue: 0, 154, 222
                                            situation_loss_tag=situation_loss_tag)

    if VIS_CONF.GT_SVEC and VIS_CONF.PRED_SVEC:
        whole_scene = tm.util.concatenate([scene, arrow, arrow_pd])
    elif VIS_CONF.GT_SVEC:
        whole_scene = tm.util.concatenate([scene, arrow])
    else:
        whole_scene = tm.util.concatenate([scene])
    
    if CONF.VISUALIZATION.SAVEOBJ:
        whole_scene.export(output_file_obj)

    # prepare the scene visualization
    o3d_mesh = whole_scene.as_open3d
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(whole_scene.visual.vertex_colors[:, :3] / 255)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    # scale the mesh apporximately to 1
    bound = o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound()
    max_bound = np.max(bound)
    o3d_mesh.scale(1.8 / max_bound, center=[0, 0, 0])
    # define material
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.shader = "defaultUnlit" 
    # translate the o3d_mesh to the center of the bounding box
    o3d_center = o3d_mesh.get_center()
    o3d_mesh.translate(-o3d_center)
    render.scene.add_geometry('3d_scene', o3d_mesh, mtl, True)
    # if required, prepare the activation points visualization
    if VIS_CONF.POINT:
        # normalize, and add a z-axis value (the max of o3d_mesh z-axis value) to the points
        points *= 1.8 / max_bound
        pcd = o3d.geometry.PointCloud()
        point_z_height = o3d_mesh.get_max_bound()[2] 
        points = np.concatenate([points, np.ones((points.shape[0], 1)) * point_z_height], axis=1)
        pcd.points = o3d.utility.Vector3dVector(points)
        # define material
        mtl_points = o3d.visualization.rendering.MaterialRecord()
        # mtl_points.shader = "defaultUnlit"
        mtl_points.shader = 'defaultLitTransparency'
        mtl_points.base_color = [1.0, 1.0, 1.0, 0.6]
        mtl_points.base_roughness = 0.0
        mtl_points.base_reflectance = 0.0
        mtl_points.base_clearcoat = 1.0
        mtl_points.thickness = 1.0
        mtl_points.transmission = 1.0
        mtl_points.absorption_distance = 10
        mtl_points.absorption_color = [0.5, 0.5, 0.5]

        if VIS_CONF.ACTIVATION:
            norm = plt.Normalize(activation.min(), activation.max())
            colors = plt.cm.viridis(norm(activation)) # viridis   cividis
            colors = colors[:, :3] # Remove alpha channel if it exists
            sphere_radius = 0.015
        else:
            colors = np.ones((points.shape[0], 3)) * 0.8
            sphere_radius=0.01
        
        pcd.translate(-o3d_center)
        grid_step = 0.32 * 1.8 / max_bound
        combined_plane_mesh = o3d.geometry.TriangleMesh()
        points -= np.array(o3d_center)
        points[:, 2] += 0.5
        # remove duplicated point in points, and corresponding color
        points, idx = np.unique(points, axis=0, return_index=True)
        colors = colors[idx]
        for point, color in zip(points, colors):
            plane_mesh = o3d.geometry.TriangleMesh.create_box(width=grid_step, height=grid_step, depth=grid_step*0.01)
            plane_mesh.translate([point[0] - grid_step/2, point[1] - grid_step/2, point[2] - grid_step/2])
            plane_mesh.compute_vertex_normals()
            plane_mesh.compute_triangle_normals()
            plane_mesh.paint_uniform_color(color)
            combined_plane_mesh += plane_mesh

        render.scene.add_geometry('points', combined_plane_mesh, mtl_points, True)

    # render the image and add text

    img = render.render_to_image()
    img = np.asarray(img)

    info_text = 'QID: {}  GT Ans: {}  PD Ans: {}'.format(str(question_id), str(answer), str(answer_pd_str))
    question_text = 'Q: {}'.format(question)
    situation_text = 'S: {}'.format(situation)
    image = Image.fromarray(img)
    font = ImageFont.truetype("./utils/NotoSans-Regular.ttf", 11)

    bbox0 = ImageDraw.Draw(image).textbbox((0, 0), info_text, font=font)
    bbox1 = ImageDraw.Draw(image).textbbox((0, 0), question_text, font=font)
    bbox2 = ImageDraw.Draw(image).textbbox((0, 0), situation_text, font=font)
    info_height = bbox0[3] - bbox0[1]
    question_height = bbox1[3] - bbox1[1]
    situation_height = bbox2[3] - bbox2[1]

    # Create a new image with space at the bottom for the two lines of text
    new_image_height = image.height + int(info_height) + int(question_height) + int(situation_height) + 20
    new_image = Image.new("RGB", (image.width, new_image_height), (0, 0, 0))
    new_image.paste(image, (0, 0))

    # Add the texts to the new image
    draw = ImageDraw.Draw(new_image)
    draw.text((10, image.height + 5), info_text, font=font, fill=(255, 255, 255))
    draw.text((10, image.height + int(info_height) + 10), question_text, font=font, fill=(255, 255, 255))
    draw.text((10, image.height + int(info_height) + int(question_height) + 15), situation_text, font=font, fill=(255, 255, 255))

    # Save the new image
    new_image.save(output_file_png)

    render.scene.camera.look_at(VIS_CONF.LOOK_AT[0], [-VIS_CONF.LOOK_AT[1][0], VIS_CONF.LOOK_AT[1][1], VIS_CONF.LOOK_AT[1][2]], VIS_CONF.LOOK_AT[2]) 
    
    if VIS_CONF.GENERATE_MINUS:
        img = render.render_to_image()
        img = np.asarray(img)
        image = Image.fromarray(img)
        image.save(output_file_png_minus)

    # generate video
    if VIS_CONF.VIDEO:
        os.makedirs(output_file_video, exist_ok=True)
        # create linear interpolation of 360 degree
        yaw_seq = np.linspace(0, 360, 49)[:-1]
        for frame in tqdm.tqdm(range(48)):
            # turn elev and yaw to camera location with R
            cam_x = VIS_CONF.LOOK_AT[1][0] * np.cos(np.deg2rad(yaw_seq[frame]))
            cam_y = VIS_CONF.LOOK_AT[1][0] * np.sin(np.deg2rad(yaw_seq[frame]))
            cam_z = VIS_CONF.LOOK_AT[1][2]
            render.scene.camera.look_at([0, 0, 0], [cam_x, cam_y, cam_z], [0, 0, 1]) 

            # render the image and add text
            img = render.render_to_image()
            img = np.asarray(img)

            image = Image.fromarray(img)
            image.save(os.path.join(output_file_video, '%03d.png'%(frame)))

    render.scene.remove_geometry("3d_scene")
    if VIS_CONF.POINT:
        render.scene.remove_geometry("points")
