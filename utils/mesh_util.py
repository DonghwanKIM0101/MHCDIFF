import os.path as osp
import _pickle as cPickle
import numpy as np
import torch
import trimesh
import tinyobjloader
import open3d as o3d

import torch.nn.functional as F
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance

from utils.render_utils import face_vertices


def evaluate_registration(verts_gt, verts_pr, verts_smpl):
    # To evaluate Hi4D
    height_pr = verts_pr[...,1].max() - verts_pr[...,1].min()
    height_gt = verts_gt[...,1].max() - verts_gt[...,1].min()
    scale = height_gt / height_pr
    verts_pr *= scale
    verts_smpl *= scale

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(verts_gt)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts_pr)

    source = pcd
    target = pcd_gt
    threshold = 5
    trans_init = np.asarray(
                [[1, 0, 0,  0],
                [0, 1, 0,  0],
                [0, 0,  1, 0],
                [0, 0, 0, 1]])
    
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)

    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 200))

    verts_pr = np.matmul(reg_p2p.transformation[:3,:3], verts_pr.T).T + reg_p2p.transformation[:3,3]
    verts_smpl = np.matmul(reg_p2p.transformation[:3,:3], verts_smpl.T).T + reg_p2p.transformation[:3,3]

    return verts_pr, verts_smpl


def obj_loader(path):
    # Create reader.
    reader = tinyobjloader.ObjReader()

    # Load .obj(and .mtl) using default configuration
    ret = reader.ParseFromFile(path)

    if ret == False:
        print("Failed to load : ", path)
        return None

    # note here for wavefront obj, #v might not equal to #vt, same as #vn.
    attrib = reader.GetAttrib()
    verts = np.array(attrib.vertices).reshape(-1, 3)

    shapes = reader.GetShapes()
    tri = shapes[0].mesh.numpy_indices().reshape(-1, 9)
    faces = tri[:, [0, 3, 6]]

    return verts, faces


def rescale_smpl(fitted_path, scale=100, translate=(0, 0, 0)):

    fitted_body = trimesh.load(fitted_path, process=False, maintain_order=True, skip_materials=True)
    resize_matrix = trimesh.transformations.scale_and_translate(scale=(scale), translate=translate)

    fitted_body.apply_transform(resize_matrix)

    return np.array(fitted_body.vertices)


def read_smpl_constants(folder):
    """Load smpl vertex code"""
    smpl_vtx_std = np.loadtxt(osp.join(folder, 'vertices.txt'))
    min_x = np.min(smpl_vtx_std[:, 0])
    max_x = np.max(smpl_vtx_std[:, 0])
    min_y = np.min(smpl_vtx_std[:, 1])
    max_y = np.max(smpl_vtx_std[:, 1])
    min_z = np.min(smpl_vtx_std[:, 2])
    max_z = np.max(smpl_vtx_std[:, 2])

    smpl_vtx_std[:, 0] = (smpl_vtx_std[:, 0] - min_x) / (max_x - min_x)
    smpl_vtx_std[:, 1] = (smpl_vtx_std[:, 1] - min_y) / (max_y - min_y)
    smpl_vtx_std[:, 2] = (smpl_vtx_std[:, 2] - min_z) / (max_z - min_z)
    smpl_vertex_code = np.float32(np.copy(smpl_vtx_std))
    """Load smpl faces & tetrahedrons"""
    smpl_faces = np.loadtxt(osp.join(folder, 'faces.txt'), dtype=np.int32) - 1
    smpl_face_code = (
        smpl_vertex_code[smpl_faces[:, 0]] + smpl_vertex_code[smpl_faces[:, 1]] +
        smpl_vertex_code[smpl_faces[:, 2]]
    ) / 3.0
    smpl_tetras = np.loadtxt(osp.join(folder, 'tetrahedrons.txt'), dtype=np.int32) - 1

    return smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras


def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights


def build_triangles(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device=vertices.device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]


def cal_sdf(verts, faces, normals, points):
    # functions modified from ICON
    
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    
    Bsize = points.shape[0]
    
    triangles = build_triangles(verts, faces)
    normals_tri = build_triangles(normals, faces)
    
    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    
    closest_normals = torch.gather(normals_tri, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    bary_weights = barycentric_coordinates_of_projection(
        points.view(-1, 3), closest_triangles)
    
    pts_normals = (closest_normals*bary_weights[:, None]).sum(1).unsqueeze(0)
    
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_occ = check_sign(verts, faces[0], points).float()
    pts_signs = 2.0 * (pts_occ - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)
    pts_occ = pts_occ.unsqueeze(-1)

    return pts_occ.view(Bsize, -1, 1), pts_sdf.view(Bsize, -1, 1), pts_normals.view(Bsize, -1, 3)


def compute_normal_batch(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]

    vert_norm = torch.zeros(bs * nv, 3).type_as(vertices)
    tris = face_vertices(vertices, faces)
    face_norm = F.normalize(
        torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0]), dim=-1
    )

    faces = (faces + (torch.arange(bs).type_as(faces) * nv)[:, None, None]).view(-1, 3)

    vert_norm[faces[:, 0]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 1]] += face_norm.view(-1, 3)
    vert_norm[faces[:, 2]] += face_norm.view(-1, 3)

    vert_norm = F.normalize(vert_norm, dim=-1).view(bs, nv, 3)

    return vert_norm


def projection(points, calib):
    if torch.is_tensor(points):
        calib = torch.as_tensor(calib) if not torch.is_tensor(calib) else calib
        return torch.mm(calib[:3, :3], points.T).T + calib[:3, 3]
    else:
        return np.matmul(calib[:3, :3], points.T).T + calib[:3, 3]


class SMPLX():
    def __init__(self):

        self.current_dir = osp.join(osp.dirname(__file__), "../data/smpl_related")

        self.smpl_verts_path = osp.join(self.current_dir, "smpl_data/smpl_verts.npy")
        self.smpl_faces_path = osp.join(self.current_dir, "smpl_data/smpl_faces.npy")
        self.smplx_verts_path = osp.join(self.current_dir, "smpl_data/smplx_verts.npy")
        self.smplx_faces_path = osp.join(self.current_dir, "smpl_data/smplx_faces.npy")
        self.cmap_vert_path = osp.join(self.current_dir, "smpl_data/smplx_cmap.npy")

        self.smplx_to_smplx_path = osp.join(self.current_dir, "smpl_data/smplx_to_smpl.pkl")

        self.smplx_eyeball_fid = osp.join(self.current_dir, "smpl_data/eyeball_fid.npy")
        self.smplx_fill_mouth_fid = osp.join(self.current_dir, "smpl_data/fill_mouth_fid.npy")

        self.smplx_faces = np.load(self.smplx_faces_path)
        self.smplx_verts = np.load(self.smplx_verts_path)
        self.smpl_verts = np.load(self.smpl_verts_path)
        self.smpl_faces = np.load(self.smpl_faces_path)

        self.smplx_eyeball_fid = np.load(self.smplx_eyeball_fid)
        self.smplx_mouth_fid = np.load(self.smplx_fill_mouth_fid)

        self.smplx_to_smpl = cPickle.load(open(self.smplx_to_smplx_path, 'rb'))

        self.model_dir = osp.join(self.current_dir, "models")
        self.tedra_dir = osp.join(self.current_dir, "../tedra_data")

    def cmap_smpl_vids(self, type):

        # keys:
        # closest_faces -   [6890, 3] with smplx vert_idx
        # bc            -   [6890, 3] with barycentric weights

        cmap_smplx = torch.as_tensor(np.load(self.cmap_vert_path)).float()
        if type == 'smplx':
            return cmap_smplx
        elif type == 'smpl':
            bc = torch.as_tensor(self.smplx_to_smpl['bc'].astype(np.float32))
            closest_faces = self.smplx_to_smpl['closest_faces'].astype(np.int32)

            cmap_smpl = torch.einsum('bij, bi->bj', cmap_smplx[closest_faces], bc)

            return cmap_smpl
