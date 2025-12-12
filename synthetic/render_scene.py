import os
import numpy as np
import torch
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    TexturesVertex,
)
from pytorch3d.renderer import look_at_view_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def next_index():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("rgb_") and f.endswith(".png")]
    if not files:
        return 1
    nums = []
    for f in files:
        try:
            nums.append(int(f[4:8]))
        except:
            pass
    return max(nums) + 1 if nums else 1


def rot_x(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float32)

def rot_y(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], dtype=torch.float32)

def apply_rt(verts, R=None, t=None):
    if R is not None:
        verts = verts @ R.t()
    if t is not None:
        verts = verts + t
    return verts


def create_cube_VFC(size=1.0, color=(0.6,0.6,0.9), R=None, t=None):
    s = size
    verts = torch.tensor([
        [-s,-s,-s],[ s,-s,-s],[ s, s,-s],[-s, s,-s],
        [-s,-s, s],[ s,-s, s],[ s, s, s],[-s, s, s],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [2,3,7],[2,7,6],
        [1,2,6],[1,6,5],
        [0,3,7],[0,7,4],
    ], dtype=torch.int64)
    verts = apply_rt(verts, R, t)
    V = verts.shape[0]
    cols = torch.tensor(color, dtype=torch.float32).view(1,1,3).repeat(1,V,1).squeeze(0)
    return verts, faces, cols


def create_torus_VFC(Rmaj=1.5, rmin=0.5, nu=64, nv=32, color=(0.9,0.6,0.3), R=None, t=None):
    u = torch.linspace(0, 2*np.pi, nu, dtype=torch.float32)[:-1]
    v = torch.linspace(0, 2*np.pi, nv, dtype=torch.float32)[:-1]
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    x = (Rmaj + rmin*torch.cos(vv)) * torch.cos(uu)
    y = (Rmaj + rmin*torch.cos(vv)) * torch.sin(uu)
    z = rmin * torch.sin(vv)
    verts = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=1)
    faces = []
    for i in range(nu-1):
        for j in range(nv-1):
            a = i*(nv-1) + j
            b = ((i+1)%(nu-1))*(nv-1) + j
            c = ((i+1)%(nu-1))*(nv-1) + (j+1)%(nv-1)
            d = i*(nv-1) + (j+1)%(nv-1)
            faces.append([a,b,c]); faces.append([a,c,d])
    faces = torch.tensor(faces, dtype=torch.int64)
    verts = apply_rt(verts, R, t)
    V = verts.shape[0]
    cols = torch.tensor(color, dtype=torch.float32).view(1,1,3).repeat(1,V,1).squeeze(0)
    return verts, faces, cols


def create_cylinder_VFC(radius=0.7, height=2.0, n=64, color=(0.6,0.9,0.6), R=None, t=None):
    ang = torch.linspace(0, 2*np.pi, n+1, dtype=torch.float32)[:-1]
    x = radius*torch.cos(ang); y = radius*torch.sin(ang)
    z0 = -height/2.0; z1 = height/2.0
    ring0 = torch.stack([x, y, torch.full_like(x, z0)], dim=1)
    ring1 = torch.stack([x, y, torch.full_like(x, z1)], dim=1)
    c0 = torch.tensor([[0,0,z0]], dtype=torch.float32)
    c1 = torch.tensor([[0,0,z1]], dtype=torch.float32)
    verts = torch.cat([ring0, ring1, c0, c1], dim=0)
    faces = []
    for i in range(n):
        a0 = i
        a1 = (i+1) % n
        b0 = i + n
        b1 = ((i+1)%n) + n
        faces.append([a0,a1,b1]); faces.append([a0,b1,b0])
    c0i = 2*n
    c1i = 2*n+1
    for i in range(n):
        a = i; b = (i+1)%n
        faces.append([c0i, b, a])
        a = i + n; b = ((i+1)%n) + n
        faces.append([c1i, a, b])
    faces = torch.tensor(faces, dtype=torch.int64)
    verts = apply_rt(verts, R, t)
    V = verts.shape[0]
    cols = torch.tensor(color, dtype=torch.float32).view(1,1,3).repeat(1,V,1).squeeze(0)
    return verts, faces, cols


def build_mesh(verts, faces, cols):
    tex = TexturesVertex(verts_features=cols.unsqueeze(0).to(device))
    return Meshes(verts=[verts.to(device)], faces=[faces.to(device)], textures=tex)


def render_and_save(renderer, mesh, rgb_path, depth_path):
    rgb = renderer(mesh)
    rgb_image = (rgb[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8)
    fragments = renderer.rasterizer(mesh)
    depth = fragments.zbuf[0].detach().cpu().numpy()[..., 0]
    Image.fromarray(rgb_image).save(rgb_path)
    np.save(depth_path, depth)
    print(f"Zapisano: {rgb_path}")
    print(f"Zapisano: {depth_path}")


def render_scene():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    idx = next_index()

    Rcam, Tcam = look_at_view_transform(dist=8.0, elev=20.0, azim=30.0)
    cameras = FoVPerspectiveCameras(device=device, R=Rcam, T=Tcam, fov=40.0)
    lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])
    raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(cameras=cameras, lights=lights, device=device),
    )

    v,f,c = create_cube_VFC(size=1.0, color=(0.6,0.6,0.9))
    mesh = build_mesh(v,f,c)
    render_and_save(renderer, mesh,
                    os.path.join(OUTPUT_DIR, f"rgb_{idx:04}.png"),
                    os.path.join(OUTPUT_DIR, f"depth_gt_{idx:04}.npy"))
    idx += 1

    v,f,c = create_torus_VFC(Rmaj=1.6, rmin=0.45, nu=72, nv=36, color=(0.9,0.6,0.3),
                             R=rot_x(torch.tensor(0.6)), t=torch.tensor([0.0,0.0,0.0]))
    mesh = build_mesh(v,f,c)
    render_and_save(renderer, mesh,
                    os.path.join(OUTPUT_DIR, f"rgb_{idx:04}.png"),
                    os.path.join(OUTPUT_DIR, f"depth_gt_{idx:04}.npy"))
    idx += 1

    v1,f1,c1 = create_cylinder_VFC(radius=0.7, height=2.0, n=64, color=(0.6,0.9,0.6),
                                   R=None, t=torch.tensor([0.0,0.0,0.0]))
    Rcube = rot_y(torch.tensor(np.deg2rad(45.0)))
    v2,f2,c2 = create_cube_VFC(size=0.9, color=(0.9,0.5,0.5),
                               R=Rcube, t=torch.tensor([1.8, 0.0, 1.2]))
    verts = torch.cat([v1, v2], dim=0)
    faces = torch.cat([f1, f2 + v1.shape[0]], dim=0)
    cols  = torch.cat([c1, c2], dim=0)
    mesh = build_mesh(verts, faces, cols)
    render_and_save(renderer, mesh,
                    os.path.join(OUTPUT_DIR, f"rgb_{idx:04}.png"),
                    os.path.join(OUTPUT_DIR, f"depth_gt_{idx:04}.npy"))


def main():
    render_scene()


if __name__ == "__main__":
    main()
