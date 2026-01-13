import struct
import os

class SceneInterface:
    """Class for reading and writing interface mvs binary files"""

    MVSI_MAGIC = b"MVSI"
    MVS_MAGIC = b"MVS\x00"
    VERSION_MAX = 7
    DEFAULT_WRITE_VERSION = 7

    class _BW:
        """Binary writer for MVS files"""

        def __init__(self, fh):
            self.fh = fh

        def write(self, data):
            self.fh.write(data)

        def u32(self, val):
            self.write(struct.pack("<I", val))

        def u64(self, val):
            self.write(struct.pack("<Q", val))

        def f32(self, val):
            self.write(struct.pack("<f", val))

        def f64(self, val):
            self.write(struct.pack("<d", val))

        def p3f(self, vals):
            self.write(struct.pack("<fff", *vals))

        def p3d(self, vals):
            self.write(struct.pack("<ddd", *vals))

        def mat33d(self, m):
            flat = m[0] + m[1] + m[2]
            self.write(struct.pack("<" + "d" * 9, *flat))

        def mat44d(self, m):
            flat = m[0] + m[1] + m[2] + m[3]
            self.write(struct.pack("<" + "d" * 16, *flat))

        def col3(self, vals):
            self.write(struct.pack("<BBB", *vals))

        def string(self, s):
            encoded = s.encode("utf-8")
            self.u64(len(encoded))
            if len(encoded) > 0:
                self.write(encoded)

        def vec(self, items, write_fn):
            self.u64(len(items))
            for item in items:
                write_fn(item)

    class _BR:
        def __init__(self, fh):
            self.fh = fh
            cur = fh.tell()
            try:
                fh.seek(0, os.SEEK_END)
                self.size = fh.tell()
            finally:
                fh.seek(cur, os.SEEK_SET)

        def read(self, n):
            b = self.fh.read(n)
            if b is None or len(b) != n:
                raise EOFError("Unexpected EOF")
            return b

        def seek(self, off, whence=os.SEEK_SET):
            self.fh.seek(off, whence)

        def tell(self):
            return self.fh.tell()

        def u32(self):
            return struct.unpack("<I", self.read(4))[0]

        def u64(self):
            return struct.unpack("<Q", self.read(8))[0]

        def f32(self):
            return struct.unpack("<f", self.read(4))[0]

        def f64(self):
            return struct.unpack("<d", self.read(8))[0]

        def p3f(self):
            return list(struct.unpack("<fff", self.read(12)))

        def p3d(self):
            return list(struct.unpack("<ddd", self.read(24)))

        def mat33d(self):
            m = list(struct.unpack("<" + "d" * 9, self.read(8 * 9)))
            return [m[0:3], m[3:6], m[6:9]]

        def mat44d(self):
            m = list(struct.unpack("<" + "d" * 16, self.read(8 * 16)))
            return [m[0:4], m[4:8], m[8:12], m[12:16]]

        def col3(self):
            return list(struct.unpack("<BBB", self.read(3)))

        def string(self):
            n = self.u64()
            if n == 0:
                return ""
            if n > 128 * 1024 * 1024 or n > (self.size - self.tell()):
                raise ValueError("Invalid string length")
            return self.read(n).decode("utf-8", errors="strict")

        def vec(self, fn):
            n = self.u64()
            return [fn() for _ in range(n)]

    @staticmethod
    def _detect_mvsi_version(br: "SceneInterface._BR") -> int:
        head = br.read(4)
        if head == SceneInterface.MVSI_MAGIC:
            ver = br.u32()
            _ = br.u32()
            if ver > SceneInterface.VERSION_MAX:
                raise ValueError("Unsupported MVSI version")
            return ver
        if head == SceneInterface.MVS_MAGIC:
            raise ValueError(
                "MVS project file detected; use SaveInterface to export MVSI"
            )
        br.seek(0)
        return 0

    @staticmethod
    def _read_optional_segments(br: "SceneInterface._BR", ver: int):
        lines = []
        linesNormal = []
        linesColor = []
        transform = []
        obb = {}
        if ver > 0:

            def read_line():
                pt1 = br.p3f()
                pt2 = br.p3f()

                def read_view():
                    return {"imageID": br.u32(), "confidence": br.f32()}

                return {"pt1": pt1, "pt2": pt2, "views": br.vec(read_view)}

            lines = br.vec(read_line)
            linesNormal = br.vec(lambda: {"n": br.p3f()})
            linesColor = br.vec(lambda: {"c": br.col3()})
            if ver > 1:
                transform = br.mat44d()
                if ver > 5:
                    obb = {"rot": br.mat33d(), "ptMin": br.p3d(), "ptMax": br.p3d()}
        return lines, linesNormal, linesColor, transform, obb

    @staticmethod
    def read(path):
        with open(path, "rb") as fh:
            br = SceneInterface._BR(fh)
            ver = SceneInterface._detect_mvsi_version(br)

            def read_platform():
                name = br.string()

                def read_camera():
                    cam = {
                        "name": br.string(),
                        "bandName": br.string() if ver > 3 else "",
                        "width": br.u32() if ver > 0 else 0,
                        "height": br.u32() if ver > 0 else 0,
                    }
                    cam["K"] = br.mat33d()
                    cam["R"] = br.mat33d()
                    cam["C"] = br.p3d()
                    return cam

                def read_pose():
                    return {"R": br.mat33d(), "C": br.p3d()}

                return {
                    "name": name,
                    "cameras": br.vec(read_camera),
                    "poses": br.vec(read_pose),
                }

            def read_image():
                d = {
                    "name": br.string(),
                    "maskName": br.string() if ver > 4 else "",
                    "platformID": br.u32(),
                    "cameraID": br.u32(),
                    "poseID": br.u32(),
                    "ID": br.u32() if ver > 2 else 0xFFFFFFFF,
                }
                if ver > 6:
                    d["minDepth"] = br.f32()
                    d["avgDepth"] = br.f32()
                    d["maxDepth"] = br.f32()

                    def read_vs():
                        return {
                            "ID": br.u32(),
                            "points": br.u32(),
                            "scale": br.f32(),
                            "angle": br.f32(),
                            "area": br.f32(),
                            "score": br.f32(),
                        }

                    d["viewScores"] = br.vec(read_vs)
                else:
                    d["viewScores"] = []
                return d

            def read_vertex():
                X = br.p3f()

                def read_view():
                    return {"imageID": br.u32(), "confidence": br.f32()}

                return {"X": X, "views": br.vec(read_view)}

            platforms = br.vec(read_platform)
            images = br.vec(read_image)
            vertices = br.vec(read_vertex)
            verticesNormal = br.vec(lambda: {"n": br.p3f()})
            verticesColor = br.vec(lambda: {"c": br.col3()})
            lines, linesNormal, linesColor, transform, obb = (
                SceneInterface._read_optional_segments(br, ver)
            )

            iface = {
                "platforms": platforms,
                "images": images,
                "vertices": vertices,
                "verticesNormal": verticesNormal,
                "verticesColor": verticesColor,
                "lines": lines,
                "linesNormal": linesNormal,
                "linesColor": linesColor,
                "transform": transform,
                "obb": obb,
            }
            return ver, iface

    @staticmethod
    def _write_header(bw: "_BW", version: int):
        """Write MVS file header."""
        bw.write(SceneInterface.MVSI_MAGIC)
        bw.u32(version)
        bw.u32(0)  # reserved

    @staticmethod
    def _write_platforms(bw: "_BW", platforms: list, version: int):
        """Write platforms section (cameras and poses)."""

        def write_camera(cam):
            bw.string(cam.get("name", ""))
            if version > 3:
                bw.string(cam.get("bandName", ""))
            if version > 0:
                bw.u32(cam.get("width", 0))
                bw.u32(cam.get("height", 0))
            bw.mat33d(cam["K"])
            bw.mat33d(cam.get("R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            bw.p3d(cam.get("C", [0.0, 0.0, 0.0]))

        def write_pose(pose):
            bw.mat33d(pose["R"])
            bw.p3d(pose["C"])

        def write_platform(plat):
            bw.string(plat.get("name", ""))
            bw.vec(plat.get("cameras", []), write_camera)
            bw.vec(plat.get("poses", []), write_pose)

        bw.vec(platforms, write_platform)

    @staticmethod
    def _write_images(bw: "_BW", images: list, version: int):
        """Write images section."""

        def write_view_score(vs):
            bw.u32(vs.get("ID", 0))
            bw.u32(vs.get("points", 0))
            bw.f32(vs.get("scale", 0.0))
            bw.f32(vs.get("angle", 0.0))
            bw.f32(vs.get("area", 0.0))
            bw.f32(vs.get("score", 0.0))

        def write_image(img):
            bw.string(img.get("name", ""))
            if version > 4:
                bw.string(img.get("maskName", ""))
            bw.u32(img.get("platformID", 0))
            bw.u32(img.get("cameraID", 0))
            bw.u32(img.get("poseID", 0))
            if version > 2:
                bw.u32(img.get("ID", 0xFFFFFFFF))
            if version > 6:
                bw.f32(img.get("minDepth", 0.0))
                bw.f32(img.get("avgDepth", 0.0))
                bw.f32(img.get("maxDepth", 0.0))
                bw.vec(img.get("viewScores", []), write_view_score)

        bw.vec(images, write_image)

    @staticmethod
    def _write_vertices(bw: "_BW", vertices: list):
        """Write vertices section with views."""

        def write_view(view):
            bw.u32(view["imageID"])
            bw.f32(view.get("confidence", 1.0))

        def write_vertex(vert):
            bw.p3f(vert["X"])
            bw.vec(vert.get("views", []), write_view)

        bw.vec(vertices, write_vertex)

    @staticmethod
    def _write_vertex_attributes(bw: "_BW", normals: list, colors: list):
        """Write vertex normals and colors."""

        def write_normal(vn):
            bw.p3f(vn.get("n", [0.0, 0.0, 1.0]))

        def write_color(vc):
            bw.col3(vc.get("c", [128, 128, 128]))

        bw.vec(normals, write_normal)
        bw.vec(colors, write_color)

    @staticmethod
    def _write_lines(bw: "_BW", lines: list, normals: list, colors: list):
        """Write lines section with normals and colors."""

        def write_line_view(view):
            bw.u32(view["imageID"])
            bw.f32(view.get("confidence", 1.0))

        def write_line(line):
            bw.p3f(line["pt1"])
            bw.p3f(line["pt2"])
            bw.vec(line.get("views", []), write_line_view)

        def write_normal(ln):
            bw.p3f(ln.get("n", [0.0, 0.0, 1.0]))

        def write_color(lc):
            bw.col3(lc.get("c", [128, 128, 128]))

        bw.vec(lines, write_line)
        bw.vec(normals, write_normal)
        bw.vec(colors, write_color)

    @staticmethod
    def _write_transform(bw: "_BW", transform: list):
        """Write transform matrix."""
        if transform:
            bw.mat44d(transform)
        else:
            # Identity transform
            bw.mat44d(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

    @staticmethod
    def _write_obb(bw: "_BW", obb: dict):
        """Write oriented bounding box."""
        if obb:
            bw.mat33d(obb["rot"])
            bw.p3d(obb["ptMin"])
            bw.p3d(obb["ptMax"])
        else:
            # Default OBB
            bw.mat33d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            bw.p3d([0.0, 0.0, 0.0])
            bw.p3d([0.0, 0.0, 0.0])

    @staticmethod
    def save(path: str, scene_data: dict, version: int = None):
        """
        Save scene data to an MVS interface file.

        Args:
            path: Output file path
            scene_data: Dictionary containing scene data with the following structure:
                - platforms: list of {name, cameras[], poses[]}
                    - cameras: {name, bandName, width, height, K, R, C}
                    - poses: {R, C}
                - images: list of {name, maskName, platformID, cameraID, poseID, ID,
                         minDepth, avgDepth, maxDepth, viewScores[]}
                - vertices: list of {X, views[]}
                    - views: {imageID, confidence}
                - verticesNormal: list of {n}
                - verticesColor: list of {c} (BGR format)
                - lines: list of {pt1, pt2, views[]}
                - linesNormal: list of {n}
                - linesColor: list of {c}
                - transform: 4x4 matrix or empty list
                - obb: {rot, ptMin, ptMax} or empty dict
            version: MVS version to write (default: DEFAULT_WRITE_VERSION)
        """
        if version is None:
            version = SceneInterface.DEFAULT_WRITE_VERSION

        with open(path, "wb") as fh:
            bw = SceneInterface._BW(fh)

            # Write header
            SceneInterface._write_header(bw, version)

            # Write main sections
            SceneInterface._write_platforms(
                bw, scene_data.get("platforms", []), version
            )
            SceneInterface._write_images(bw, scene_data.get("images", []), version)
            SceneInterface._write_vertices(bw, scene_data.get("vertices", []))
            SceneInterface._write_vertex_attributes(
                bw,
                scene_data.get("verticesNormal", []),
                scene_data.get("verticesColor", []),
            )

            # Write optional segments (version > 0)
            if version > 0:
                SceneInterface._write_lines(
                    bw,
                    scene_data.get("lines", []),
                    scene_data.get("linesNormal", []),
                    scene_data.get("linesColor", []),
                )

                # Write transform (version > 1)
                if version > 1:
                    SceneInterface._write_transform(bw, scene_data.get("transform", []))

                    # Write OBB (version > 5)
                    if version > 5:
                        SceneInterface._write_obb(bw, scene_data.get("obb", {}))
