"""生成 50m×50m×4m 房间场景（XML + PLY 网格）"""
import os
import struct
import numpy as np

# 房间尺寸（物理坐标，米）
L, W, H = 50.0, 50.0, 4.0

def write_ply(filepath, vertices, faces):
    """写 ASCII PLY 文件"""
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"{len(face)} {' '.join(map(str, face))}\n")

def make_room(out_dir):
    """生成 50m×50m×4m 房间的 6 个面"""
    os.makedirs(out_dir, exist_ok=True)
    meshes_dir = os.path.join(out_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    # floor (z=0)
    write_ply(os.path.join(meshes_dir, "floor.ply"),
              [(0,0,0), (L,0,0), (L,W,0), (0,W,0)],
              [(0,1,2), (0,2,3)])

    # ceiling (z=H)
    write_ply(os.path.join(meshes_dir, "ceiling.ply"),
              [(0,0,H), (L,0,H), (L,W,H), (0,W,H)],
              [(0,2,1), (0,3,2)])

    # wall_xp (x=L)
    write_ply(os.path.join(meshes_dir, "wall_xp.ply"),
              [(L,0,0), (L,W,0), (L,W,H), (L,0,H)],
              [(0,2,1), (0,3,2)])

    # wall_xm (x=0)
    write_ply(os.path.join(meshes_dir, "wall_xm.ply"),
              [(0,0,0), (0,W,0), (0,W,H), (0,0,H)],
              [(0,1,2), (0,2,3)])

    # wall_yp (y=W)
    write_ply(os.path.join(meshes_dir, "wall_yp.ply"),
              [(0,W,0), (L,W,0), (L,W,H), (0,W,H)],
              [(0,2,1), (0,3,2)])

    # wall_ym (y=0)
    write_ply(os.path.join(meshes_dir, "wall_ym.ply"),
              [(0,0,0), (L,0,0), (L,0,H), (0,0,H)],
              [(0,1,2), (0,2,3)])

    # XML 场景定义
    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<scene version="2.1.0">
  <integrator type="path">
    <integer name="max_depth" value="12"/>
  </integrator>
  <bsdf type="twosided" id="mat-itu_concrete">
    <bsdf type="diffuse">
      <rgb value="0.5 0.5 0.5" name="reflectance"/>
    </bsdf>
  </bsdf>
  <emitter type="constant" id="World">
    <rgb value="1.0 1.0 1.0" name="radiance"/>
  </emitter>
  <shape type="ply" id="mesh-floor">
    <string name="filename" value="meshes/floor.ply"/>
    <boolean name="face_normals" value="true"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>
  <shape type="ply" id="mesh-ceiling">
    <string name="filename" value="meshes/ceiling.ply"/>
    <boolean name="face_normals" value="true"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>
  <shape type="ply" id="mesh-wall_xp">
    <string name="filename" value="meshes/wall_xp.ply"/>
    <boolean name="face_normals" value="true"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>
  <shape type="ply" id="mesh-wall_xm">
    <string name="filename" value="meshes/wall_xm.ply"/>
    <boolean name="face_normals" value="true"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>
  <shape type="ply" id="mesh-wall_yp">
    <string name="filename" value="meshes/wall_yp.ply"/>
    <boolean name="face_normals" value="true"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>
  <shape type="ply" id="mesh-wall_ym">
    <string name="filename" value="meshes/wall_ym.ply"/>
    <boolean name="face_normals" value="true"/>
    <ref id="mat-itu_concrete" name="bsdf"/>
  </shape>
</scene>'''
    with open(os.path.join(out_dir, "room_50m.xml"), 'w', encoding='utf-8') as f:
        f.write(xml)
    print(f"saved room_50m.xml + meshes to {out_dir}")

if __name__ == "__main__":
    make_room(os.path.join(os.path.dirname(__file__), "..", "data", "room_50m"))
