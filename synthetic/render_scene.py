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
def save_depth_png(depth: np.ndarray, png_path: str):
    depth = np.nan_to_num(depth, nan=0.0)
    d_min = depth.min()
    d_max = depth.max()
    if d_max - d_min < 1e-8:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    import cv2
    cv2.imwrite(png_path, depth_norm)

def checker_texture(verts, base=(0.8, 0.8, 0.8), scale=8.0):
    v = verts.clone()

    v_min = v.min(dim=0).values
    v_max = v.max(dim=0).values
    v = (v - v_min) / (v_max - v_min + 1e-6)

    check = (
        (torch.floor(v[:, 0] * scale) +
         torch.floor(v[:, 1] * scale) +
         torch.floor(v[:, 2] * scale)) % 2
    ).float()

    col1 = torch.tensor(base) * 0.6
    col2 = torch.tensor(base) * 1.1

    colors = torch.where(
        check.unsqueeze(1) > 0,
        col1,
        col2,
    )
    return torch.clamp(colors, 0.0, 1.0)

def add_edge_shading(
    verts: torch.Tensor,
    colors: torch.Tensor,
    strength=0.35,
    axis_pairs=((0, 1), (0, 2), (1, 2)),
):
    """
    Fake krawędzie: przyciemnia okolice krawędzi bryły.
    Działa najlepiej dla prostopadłościanów i ostrosłupów.
    """
    v = verts.clone()

    # normalizacja współrzędnych
    v_min = v.min(dim=0).values
    v_max = v.max(dim=0).values
    v_norm = (v - v_min) / (v_max - v_min + 1e-6)

    edge_strength = torch.zeros(len(v), device=v.device)

    for a, b in axis_pairs:
        edge_strength += torch.minimum(
            v_norm[:, a], 1.0 - v_norm[:, a]
        )
        edge_strength += torch.minimum(
            v_norm[:, b], 1.0 - v_norm[:, b]
        )

    edge_strength = torch.clamp(edge_strength, 0.0, 1.0)
    edge_strength = 1.0 - edge_strength

    # przyciemnienie
    shaded = colors * (1.0 - strength * edge_strength.unsqueeze(1))
    return torch.clamp(shaded, 0.0, 1.0)

def make_renderer(cameras):
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(
        device=device,
        location=[[2.5, 5.0, 2.5]],   
        ambient_color=((0.25, 0.25, 0.25),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.2, 0.2, 0.2),),
    )


    return MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=HardPhongShader(
            cameras=cameras,
            lights=lights,
            device=device,
        ),
    )

def create_box_VFC(sx=2.0, sy=1.0, sz=0.2, color=(0.7, 0.7, 0.7), R=None, t=None):
    """
    Graniastosłup prostokątny (box) o pół-wymiarach sx, sy, sz.
    (czyli pełne wymiary: 2*sx x 2*sy x 2*sz)
    """
    verts = torch.tensor(
        [
            [-sx, -sy, -sz], [ sx, -sy, -sz], [ sx,  sy, -sz], [-sx,  sy, -sz],
            [-sx, -sy,  sz], [ sx, -sy,  sz], [ sx,  sy,  sz], [-sx,  sy,  sz],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # side
            [2, 3, 7], [2, 7, 6],  # side
            [1, 2, 6], [1, 6, 5],  # side
            [0, 3, 7], [0, 7, 4],  # side
        ],
        dtype=torch.int64,
    )
    verts = apply_rt(verts, R, t)
    V = verts.shape[0]
    cols = torch.tensor(color, dtype=torch.float32).view(1, 1, 3).repeat(1, V, 1).squeeze(0)
    return verts, faces, cols


def create_pyramid_VFC(base=0.8, height=0.9, color=(0.9, 0.8, 0.2), R=None, t=None):
    """
    Ostrosłup prawidłowy czworokątny:
    - podstawa w płaszczyźnie z=0, wierzchołek na z=height
    - 'base' to połowa długości boku podstawy (pełny bok = 2*base)
    """
    b = base
    verts = torch.tensor(
        [
            [-b, -b, 0.0],  # 0
            [ b, -b, 0.0],  # 1
            [ b,  b, 0.0],  # 2
            [-b,  b, 0.0],  # 3
            [0.0, 0.0, height],  # 4 apex
        ],
        dtype=torch.float32,
    )

    faces = torch.tensor(
        [
            [0, 1, 2], [0, 2, 3],  # base (2 trójkąty)
            [0, 1, 4],  # sides
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=torch.int64,
    )

    verts = apply_rt(verts, R, t)
    V = verts.shape[0]
    cols = torch.tensor(color, dtype=torch.float32).view(1, 1, 3).repeat(1, V, 1).squeeze(0)
    return verts, faces, cols

def create_walls(
    parts,
    cam_center_world,          # <- NOWE: pozycja kamery w świecie (Tensor [3])
    floor_angle_deg=45.0,
    floor_size=8.0,            # pół-wymiar podłogi
    floor_thick=0.08,          # pół-grubość
    wall_height=12.0,           # wyższe ściany
    wall_thick=0.08,           # pół-grubość
    R_part=None,               # R_up (Z->Y)
):
    """
    Podłoga + 2 ściany na "właściwym" rogu, wybieranym AUTOMATYCZNIE względem kamery.
    Ściany są pionowe i stykają się z górą podłogi.
    """

    if R_part is None:
        raise ValueError("create_walls: R_part jest wymagane")
    if cam_center_world is None:
        raise ValueError("create_walls: cam_center_world jest wymagane")

    wall_color = (0.95, 0.95, 0.65)

    # Układ środowiska (podłoga obrócona o 45°) + mapowanie Z->Y
    R_env = rot_y(torch.tensor(np.deg2rad(floor_angle_deg))) @ R_part

    # Lokalna oś X i Z podłogi w świecie
    # (kolumny macierzy obrotu to osie lokalne w świecie)
    floor_x_world = (torch.tensor([1.0, 0.0, 0.0]) @ R_env.t())
    floor_z_world = (torch.tensor([0.0, 0.0, 1.0]) @ R_env.t())

    # Rzut kamery na osie podłogi (znak mówi, z której strony stoi kamera)
    cam_x = torch.dot(cam_center_world, floor_x_world).item()
    cam_z = torch.dot(cam_center_world, floor_z_world).item()

    # Wybieramy krawędzie NAJDALSZE od kamery:
    # jeśli kamera jest po stronie +X, ściana ma być na -X (i odwrotnie)
    sx = -1.0 if cam_x > 0 else 1.0
    sz = -1.0 if cam_z > 0 else 1.0

    # ================= PODŁOGA =================
    floor_center = torch.tensor([0.0, floor_thick, 0.0])
    v_floor, f_floor, _ = create_box_VFC(
        sx=floor_size,
        sy=floor_size,
        sz=floor_thick,
        color=(1, 1, 1),  # dummy
        R=R_env,
        t=floor_center,
    )

    c_floor = checker_texture(
        v_floor,
        base=(0.85, 0.85, 0.85),
        scale=10.0,
    )

    c_floor = add_edge_shading(v_floor, c_floor, strength=0.25)
    parts.append((v_floor, f_floor, c_floor))



    # ================= ŚCIANY =================
    wall_center_y = 2.0 * floor_thick + wall_height / 2.0
    edge_offset = (floor_size - wall_thick)
    y_shift = floor_size   # przesunięcie o pół długości podłogi w X
    x_back = sx * floor_size
   # ŚCIANA "Z" (obejmuje całą krawędź podłogi wzdłuż X)
    wall_z_center = (
        torch.tensor([0.0, wall_center_y+y_shift, 0.0]) +
        (torch.tensor([0.0, x_back, 0.0]) @ R_env.t()) +  
        (torch.tensor([0.0, 0.0, (sz * (edge_offset + wall_thick))]) @ R_env.t())
    )

    v_back, f_back, c_back = create_box_VFC(
        sx=floor_size,            # długość (X)
        sy=wall_thick,            # grubość (Z)
        sz=wall_height / 2.0,     # wysokość (Y po R_part)
        color=wall_color,
        R=R_env,
        t=wall_z_center,
    )
    c_back = checker_texture(v_back, base=(0.9, 0.9, 0.7), scale=8.0)
    parts.append((v_back, f_back, c_back))



    # ŚCIANA "X" (obejmuje całą krawędź podłogi wzdłuż Z)
    wall_x_center = (
    torch.tensor([0.0, wall_center_y, 0.0]) +
    (torch.tensor([(sx * (edge_offset + wall_thick)), 0.0, 0.0]) @ R_env.t())
)

    v_side, f_side, c_side = create_box_VFC(
        sx=wall_thick,            # grubość
        sz=wall_height / 2.0,     # wysokość
        sy=floor_size,            # długość
        color=wall_color,
        R=R_env,
        t=wall_x_center,
    )
    c_side = checker_texture(v_side, base=(0.9, 0.9, 0.7), scale=8.0)
    parts.append((v_side, f_side, c_side))




def merge_parts(parts, add_walls=False, R_part=None, cam_center_world=None):
    if add_walls:
        if R_part is None:
            raise ValueError("merge_parts(add_walls=True) wymaga R_part")
        if cam_center_world is None:
            raise ValueError("merge_parts(add_walls=True) wymaga cam_center_world")

        create_walls(
            parts=parts,
            cam_center_world=cam_center_world,
            floor_angle_deg=45.0,
            R_part=R_part,
            wall_height=6.0,   
        )

    verts_all, faces_all, cols_all = [], [], []
    offset = 0
    for v, f, c in parts:
        verts_all.append(v)
        faces_all.append(f + offset)
        cols_all.append(c)
        offset += v.shape[0]

    return torch.cat(verts_all, 0), torch.cat(faces_all, 0), torch.cat(cols_all, 0)





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


def render_and_save(renderer, mesh, cameras, rgb_path, depth_path):
    rgb = renderer(mesh)
    rgb_image = (rgb[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8)

    fragments = renderer.rasterizer(mesh)

    # --- PRAWDZIWA DEPTH ---
    verts = mesh.verts_packed()              # (V, 3)
    faces = mesh.faces_packed()              # (F, 3)

    pix_to_face = fragments.pix_to_face[0, ..., 0]   # (H, W)
    bary = fragments.bary_coords[0, ..., 0, :]        # (H, W, 3)

    depth = torch.zeros_like(pix_to_face, dtype=torch.float32)

    valid = pix_to_face >= 0
    face_ids = pix_to_face[valid]
    tri = faces[face_ids]                    # (N, 3)
    tri_verts = verts[tri]                   # (N, 3, 3)

    pts_world = (tri_verts * bary[valid].unsqueeze(-1)).sum(dim=1)

    # transform world → camera
    pts_cam = cameras.get_world_to_view_transform().transform_points(pts_world)

    depth[valid] = pts_cam[:, 2]              

    Image.fromarray(rgb_image).save(rgb_path)

    depth_np = depth.cpu().numpy()
    np.save(depth_path, depth_np)

    png_path = depth_path.replace(".npy", ".png")
    save_depth_png(depth_np, png_path)



def two_tables_scene(renderer, cameras):
    """
    Scena: pokój + dwa stoliki
    - stół 1: romb (ostrosłup)
    - stół 2: sześcian
    """

    # ================= KAMERA =================
    Rcam = cameras.R[0]
    Tcam = cameras.T[0]
    cam_center_world = (-Rcam.t() @ Tcam)

    # ================= ORIENTACJA =================
    R_up = rot_x(torch.tensor(-np.pi / 2.0))

    parts = []

    # ======================================================
    # STÓŁ 1 – BLIŻEJ KAMERY (ROMB / OSTROSŁUP)
    # ======================================================
    parts_1 = []

    top_sx, top_sy, top_thick = 2.0, 1.2, 0.12
    leg_radius, leg_height = 0.12, 1.3
    top_y = leg_height + top_thick

    # blat
    v_top, f_top, c_top = create_box_VFC(
        sx=top_sx, sy=top_sy, sz=top_thick,
        color=(0.65, 0.45, 0.30),
        R=R_up,
        t=torch.tensor([0.0, top_y, 0.0]),
    )
    c_top = checker_texture(
        v_top,
        base=(0.65, 0.45, 0.30),
        scale=6.0,
    )

    c_top = add_edge_shading(v_top, c_top, 0.45)
    parts_1.append((v_top, f_top, c_top))

    # nogi
    inset = leg_radius * 1.6
    for x in [-(top_sx - inset), (top_sx - inset)]:
        for z in [-(top_sy - inset), (top_sy - inset)]:
            v_leg, f_leg, c_leg = create_cylinder_VFC(
                radius=leg_radius,
                height=leg_height,
                color=(0.25, 0.25, 0.28),
                R=R_up,
                t=torch.tensor([x, leg_height / 2.0, z]),
            )
            parts_1.append((v_leg, f_leg, c_leg))

    # romb (ostrosłup)
    v_pyr, f_pyr, c_pyr = create_pyramid_VFC(
        base=0.5,
        height=0.8,
        color=(0.85, 0.75, 0.15),
        R=R_up,
        t=torch.tensor([0.0, top_y + 0.15, 0.0]),
    )
    c_pyr = add_edge_shading(v_pyr, c_pyr, 0.6)
    parts_1.append((v_pyr, f_pyr, c_pyr))

    # przesunięcie stołu 1
    for v, f, c in parts_1:
        parts.append((apply_rt(v, t=torch.tensor([-1.5, 0.0, 0.5])), f, c))

    # ======================================================
    # STÓŁ 2 – DALEJ (SZEŚCIAN)
    # ======================================================
    parts_2 = []

    # blat
    v_top2, f_top2, c_top2 = create_box_VFC(
        sx=top_sx, sy=top_sy, sz=top_thick,
        color=(0.55, 0.55, 0.75),
        R=R_up,
        t=torch.tensor([0.0, top_y, 0.0]),
    )
    c_top2 = checker_texture(
        v_top,
        base=(0.65, 0.45, 0.30),
        scale=6.0,
    )

    c_top2 = add_edge_shading(v_top2, c_top2, 0.45)
    parts_2.append((v_top2, f_top2, c_top2))

    # nogi
    for x in [-(top_sx - inset), (top_sx - inset)]:
        for z in [-(top_sy - inset), (top_sy - inset)]:
            v_leg, f_leg, c_leg = create_cylinder_VFC(
                radius=leg_radius,
                height=leg_height,
                color=(0.25, 0.25, 0.28),
                R=R_up,
                t=torch.tensor([x, leg_height / 2.0, z]),
            )
            parts_2.append((v_leg, f_leg, c_leg))

    # sześcian
    v_cube, f_cube, c_cube = create_cube_VFC(
        size=0.5,
        color=(0.9, 0.5, 0.5),
        R=R_up,
        t=torch.tensor([0.0, top_y + 0.5, 0.0]),
    )
    c_cube = add_edge_shading(v_cube, c_cube, 0.6)
    parts_2.append((v_cube, f_cube, c_cube))

    # obrót + przesunięcie stołu 2
    R_spin = rot_y(torch.tensor(np.deg2rad(-40.0)))
    for v, f, c in parts_2:
        v2 = apply_rt(v, R=R_spin, t=torch.tensor([2.0, 0.0, -2.0]))
        parts.append((v2, f, c))

    # ======================================================
    # POKÓJ (PODŁOGA + ŚCIANY)
    # ======================================================
    v_all, f_all, c_all = merge_parts(
        parts,
        add_walls=True,
        R_part=R_up,
        cam_center_world=cam_center_world,
    )

    return build_mesh(v_all, f_all, c_all)

def table_cabinet_chair_scene(renderer, cameras):
    """
    Scena: pokój + stolik + szafa (przy ścianie X) + krzesło
    """

    # ================= KAMERA =================
    cam_center_world = cameras.get_camera_center()[0]

    # ================= ORIENTACJA =================
    R_up = rot_x(torch.tensor(-np.pi / 2.0))

    parts = []

    # ======================================================
    # STÓŁ
    # ======================================================
    top_sx, top_sy, top_thick = 2.0, 1.2, 0.12
    leg_radius, leg_height = 0.12, 1.3
    top_y = leg_height + top_thick

    # blat
    v_top, f_top, c_top = create_box_VFC(
        sx=top_sx, sy=top_sy, sz=top_thick,
        color=(0.65, 0.45, 0.30),
        R=R_up,
        t=torch.tensor([0.0, top_y, 0.0]),
    )
    c_top = add_edge_shading(v_top, c_top, 0.45)
    parts.append((v_top, f_top, c_top))

    # nogi
    inset = leg_radius * 1.6
    for x in [-(top_sx - inset), (top_sx - inset)]:
        for z in [-(top_sy - inset), (top_sy - inset)]:
            v_leg, f_leg, c_leg = create_cylinder_VFC(
                radius=leg_radius,
                height=leg_height,
                color=(0.25, 0.25, 0.28),
                R=R_up,
                t=torch.tensor([x, leg_height / 2.0, z]),
            )
            parts.append((v_leg, f_leg, c_leg))

    # ======================================================
    # SZAFA (GRANIASTOSŁUP PRZY ŚCIANIE X)
    # ======================================================
    cabinet_w, cabinet_d, cabinet_h = 1.0, 1.5, 2.2
    cabinet_y = cabinet_h / 2.0
    v_cab, f_cab, c_cab = create_box_VFC(
        sx=cabinet_w,
        sy=cabinet_d,
        sz=cabinet_h,
        color=(0.55, 0.55, 0.55),
        R=R_up,
        t=torch.tensor([3.0, cabinet_y, 0.0]),  # PRZY ŚCIANIE X
    )
    c_cab = add_edge_shading(v_cab, c_cab, 0.4)
    parts.append((v_cab, f_cab, c_cab))
         # ======================================================
    # 3 SZEŚCIANY TESTOWE – TEN SAM DEPTH (RED + BLUE)
    # ======================================================

    # pozycja kamery w świecie
    cam_center_world = cameras.get_camera_center()[0]

    # kierunek patrzenia kamery
    view_dir = -cam_center_world
    view_dir = view_dir / torch.norm(view_dir)

    # wektor w prawo (lokalny dla kamery)
    world_up = torch.tensor([0.0, 1.0, 0.0])
    right_dir = torch.cross(world_up, view_dir)
    right_dir = right_dir / torch.norm(right_dir)

    # ================= PARAMETRY =================
    cube_size = 0.4
    cube_y = cube_size

    d_shared = 4.0        # TA SAMA ODLEGŁOŚĆ od kamery (RED + BLUE)
    d_far = 6.5           # DALEJ (GREEN)
    side_offset = 1.5     # MAŁY OFFSET W LEWO

    # ================= POZYCJE =================
    # 🔵 NIEBIESKI – na osi kamery
    pos_blue = cam_center_world + view_dir * d_shared

    # 🔴 CZERWONY – ten sam depth, ale PO LEWEJ
    pos_red = cam_center_world + view_dir * d_shared - right_dir * side_offset

    # 🟢 ZIELONY – dalej
    pos_green = cam_center_world + view_dir * d_far

    # ================= DODANIE DO SCENY =================
    cube_defs = [
        (pos_red,   (0.9, 0.2, 0.2)),  # czerwony – lewo
        (pos_blue,  (0.2, 0.2, 0.9)),  # niebieski – środek
        (pos_green, (0.2, 0.9, 0.2)),  # zielony – daleko
    ]

    for pos, color in cube_defs:
        pos = torch.tensor([pos[0], cube_y, pos[2]])
        v_cube, f_cube, c_cube = create_cube_VFC(
            size=cube_size,
            color=color,
            R=R_up,
            t=pos,
        )
        c_cube = add_edge_shading(v_cube, c_cube, strength=0.5)
        parts.append((v_cube, f_cube, c_cube))

    # ======================================================
    # POKÓJ (ŚCIANY + PODŁOGA)
    # ======================================================
    v_all, f_all, c_all = merge_parts(
        parts,
        add_walls=True,
        R_part=R_up,
        cam_center_world=cam_center_world,
    )

    return build_mesh(v_all, f_all, c_all)

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
    render_and_save(renderer, mesh,cameras, 
                    os.path.join(OUTPUT_DIR, f"rgb_{idx:04}.png"),
                    os.path.join(OUTPUT_DIR, f"depth_gt_{idx:04}.npy"))
    idx += 1

    v,f,c = create_torus_VFC(Rmaj=1.6, rmin=0.45, nu=72, nv=36, color=(0.9,0.6,0.3),
                             R=rot_x(torch.tensor(0.6)), t=torch.tensor([0.0,0.0,0.0]))
    mesh = build_mesh(v,f,c)
    render_and_save(renderer, mesh,cameras,
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
    render_and_save(renderer, mesh,cameras,
                    os.path.join(OUTPUT_DIR, f"rgb_{idx:04}.png"),
                    os.path.join(OUTPUT_DIR, f"depth_gt_{idx:04}.npy"))
    # kamera dla 1 stolu
    Rcam_table, Tcam_table = look_at_view_transform(
        dist=10.5,   
        elev=20.0,    
        azim=35.0     
    )
    cam_center_world = (-Rcam_table[0].t() @ Tcam_table[0])
    cameras_table = FoVPerspectiveCameras(
        device=device,
        R=Rcam_table,
        T=Tcam_table,
        fov=45.0
    )

    renderer_table = make_renderer(cameras_table)

      # stół + ostrosłup

    top_sx, top_sy, top_thick = 2.2, 1.3, 0.12
    leg_radius = 0.12
    leg_height = 1.4
    R_up = rot_x(torch.tensor(-np.pi / 2.0))

    R_spin = rot_y(torch.tensor(np.deg2rad(35.0)))

    R_part = R_up

    legs_center_y = leg_height / 2.0
    top_center_y  = leg_height + top_thick

    parts = []

    # blat
    v_top, f_top, c_top = create_box_VFC(
        sx=top_sx, sy=top_sy, sz=top_thick,
        color=(0.65, 0.45, 0.30),
        R=R_part,
        t=torch.tensor([0.0, top_center_y, 0.0]),
    )
    c_top = checker_texture(
        v_top,
        base=(0.65, 0.45, 0.30),
        scale=6.0,
    )
    c_top = add_edge_shading(
    v_top,
    c_top,
    strength=0.6,
    )
    parts.append((v_top, f_top, c_top))

    # nogi 
    inset = leg_radius * 1.6
    xs = [-(top_sx - inset), (top_sx - inset)]
    zs = [-(top_sy - inset), (top_sy - inset)]

    for x in xs:
        for z in zs:
            v_leg, f_leg, c_leg = create_cylinder_VFC(
                radius=leg_radius,
                height=leg_height,
                n=64,
                color=(0.25, 0.25, 0.28),
                R=R_part,
                t=torch.tensor([x, legs_center_y, z]),
            )
            parts.append((v_leg, f_leg, c_leg))

    # ostrosłup na blacie
    pyr_base = 0.55
    pyr_height = 0.85
    pyramid_base_y = leg_height + 2.0 * top_thick

    v_pyr, f_pyr, _ = create_pyramid_VFC(
        base=pyr_base,
        height=pyr_height,
        color=(1, 1, 1),
        R=R_part,
        t=torch.tensor([0.0, pyramid_base_y, 0.0]),
    )

    c_pyr = checker_texture(
        v_pyr,
        base=(0.85, 0.75, 0.15),
        scale=4.0,
    )

    c_pyr = add_edge_shading(
        v_pyr,
        c_pyr,
        strength=0.65,
    )

    parts.append((v_pyr, f_pyr, c_pyr))


    # scalenie
    v_all, f_all, c_all = merge_parts(
        parts,
        add_walls=True,
        R_part=R_part,
        cam_center_world=cam_center_world,
    )

    # <-- KLUCZ: obrót CAŁEGO stołu po złożeniu (obraca też pozycje nóg)
    v_all = apply_rt(v_all, R=R_spin, t=None)

    mesh = build_mesh(v_all, f_all, c_all)

    idx = next_index()
    render_and_save(
        renderer_table,
        mesh,cameras,
        os.path.join(OUTPUT_DIR, f"rgb_{idx:04}.png"),
        os.path.join(OUTPUT_DIR, f"depth_gt_{idx:04}.npy"),
    )

    Rcam_tt, Tcam_tt = look_at_view_transform(
        dist=11.0,   # dalej
        elev=20.0,
        azim=0.0
    )

    cameras_two_tables = FoVPerspectiveCameras(
        device=device,
        R=Rcam_tt,
        T=Tcam_tt,
        fov=45.0
    )

    renderer_two_tables = make_renderer(cameras_two_tables)
    mesh = two_tables_scene(renderer_two_tables, cameras_two_tables)
    idx = next_index()
    render_and_save(
        renderer_two_tables,
        mesh,
        cameras_two_tables,
        os.path.join(OUTPUT_DIR, f"rgb_{idx:04}.png"),
        os.path.join(OUTPUT_DIR, f"depth_gt_{idx:04}.npy"),
    )
# ======================================================
# SCENA: STÓŁ + SZAFA + KRZESŁO
# ======================================================
    Rcam_ttt, Tcam_ttt = look_at_view_transform(
        dist=10.0,   # dalej
        elev=30.0,
        azim=0.0
    )

    cameras_table_chair = FoVPerspectiveCameras(
        device=device,
        R=Rcam_ttt,
        T=Tcam_ttt,
        fov=50.0
    )
    renderer_table_chair = make_renderer(cameras_table_chair)
    mesh = table_cabinet_chair_scene(renderer_table_chair, cameras_table_chair)

    idx = next_index()
    render_and_save(
        renderer_table_chair,
        mesh,
        cameras_table_chair,
        os.path.join(OUTPUT_DIR, f"rgb_{idx:04}.png"),
        os.path.join(OUTPUT_DIR, f"depth_gt_{idx:04}.npy"),
    )



def main():
    render_scene()


if __name__ == "__main__":
    main()
