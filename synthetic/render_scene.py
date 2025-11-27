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
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def create_cube(size=1.0):
    """
    Tworzy sześcian 3D o boku size, wycentrowany w (0,0,0).
    Zwraca mesh PyTorch3D.
    """

    verts = torch.tensor([
        [-size, -size, -size],
        [ size, -size, -size],
        [ size,  size, -size],
        [-size,  size, -size],
        [-size, -size,  size],
        [ size, -size,  size],
        [ size,  size,  size],
        [-size,  size,  size],
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0,1,2], [0,2,3],
        [4,5,6], [4,6,7],
        [0,1,5], [0,5,4],
        [2,3,7], [2,7,6],
        [1,2,6], [1,6,5],
        [0,3,7], [0,7,4],
    ], dtype=torch.int64)

    # kolor (błękitny)
    verts_rgb = torch.tensor([[0.6, 0.6, 0.9]]).repeat(verts.shape[0], 1)

    # DODAJEMY WYMIAR PARTII (batch)
    verts_rgb = verts_rgb.unsqueeze(0)  # (1, 8, 3)

    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )
    return mesh



def render_scene():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[render_scene] Tworzę scenę 3D...")

    # 1. Kamera (z góry patrząca w stronę obiektu)
    R, T = look_at_view_transform(
    dist=8.0,     # odległość kamery od środka sceny
    elev=20.0,    # kąt podniesienia kamery w górę
    azim=30.0     # obrót wokół osi Y
)

    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        fov=40.0
)

    # 2. Światło
    lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

    # 3. Obiekt – sześcian
    cube = create_cube(size=1.0)

    # 4. Ustawienia rasteryzacji
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # 5. Renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            cameras=cameras,
            lights=lights,
            device=device
        )
    )

    # Render RGB
    rgb = renderer(cube)
    rgb_image = (rgb[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

    # Render depth map z z-buffer
    fragments = renderer.rasterizer(cube)
    depth = fragments.zbuf[0].cpu().numpy()
    depth = depth[..., 0]     
    # Zapis RGB
    rgb_path = os.path.join(OUTPUT_DIR, "rgb_0001.png")
    Image.fromarray(rgb_image).save(rgb_path)

    # Zapis depth (metry!)
    depth_path = os.path.join(OUTPUT_DIR, "depth_gt_0001.npy")
    np.save(depth_path, depth)

    print("[render_scene] Zapisano:")
    print(" - RGB:", rgb_path)
    print(" - Depth GT:", depth_path)


def main():
    render_scene()


if __name__ == "__main__":
    main()
