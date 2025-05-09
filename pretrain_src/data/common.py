import os
import math
import json
import numpy as np
import networkx as nx

import torch

def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: list or nparray int, shape=(N, )
    Returns:
        masks: nparray, shape=(N, L), padded=0
    """
    seq_lens = np.array(seq_lens)
    if max_len is None:
        max_len = max(seq_lens)
    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=np.bool)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks

# headings:36, elevations:36, angle_feat_size:4,返回36*4的矩阵，每一行代表一个角度的特征
def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]  # 4*36
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)  # 36*4
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

# 获得一周的角度，间隔20度，每个角度包含三个俯角：-30，0，30，一共12*3=36个角度，每个角度用两位表示
def get_view_rel_angles(baseViewId=0):
    rel_angles = np.zeros((36, 2), dtype=np.float32)

    base_heading = (baseViewId % 12) * math.radians(30)  # 一周12个角度，每个角度30度
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        rel_angles[ix, 0] = heading - base_heading
        rel_angles[ix, 1] = elevation - base_elevation

    return rel_angles

# 加载图，返回图、最短距离、最短路径
def load_nav_graphs(connectivity_dir):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    scans = [x.strip() for x in open(os.path.join(connectivity_dir, 'scans.txt')).readlines()]
    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()  # 建立一个没有节点和边的空图
            positions = {}
            data = json.load(f)  # 读取json图文件, data是一个list, 包含131个位置，每个位置有6张图
            for i, item in enumerate(data):
                if item['included']:  # 如果这个位置包含在数据集中
                    for j,conn in enumerate(item['unobstructed']):  # 遍历131个点，查看该位置是否能到这些点
                        if conn and data[j]['included']:  # 如果这个点是unobstructed的，并且这个点也包含在数据集中则可以形成一条边
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]) # 保存这个点的位置，使用pose中的三个元素
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))  # 向图中添加一条边
            nx.set_node_attributes(G, values=positions, name='position')  # 为图中的每个节点添加属性，属性名为position，属性值为该点位置
            graphs[scan] = G  # 将该图添加到图字典中

    shortest_distances = {}
    shortest_paths = {}
    for scan, G in graphs.items():  # compute all shortest paths
        shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    return graphs, shortest_distances, shortest_paths

def softmax(logits, dim=1):
    # logits: (n, d)
    tmp = np.exp(logits)
    return tmp / np.sum(tmp, axis=dim, keepdims=True)


def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist
    
def normalize_angle(x):
    '''convert radians into (-pi, pi]'''
    pi2 = 2 * math.pi
    x = x % pi2 # [0, 2pi]
    x = np.where(x > math.pi, x - pi2, x)
    return x