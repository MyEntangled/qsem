import numpy as np

class RectMesh:
    """
    Axis-aligned tensor-product rectangular mesh in nD.

    Representation:
      - nodes[d]: 1D array of nodal coordinates along dimension d, length N_d+1
      - N[d]: number of elements along dimension d
      - strides: row-major strides to unravel a flat element index into (i1,...,in)

    Endpoints of element e in dimension d:
      [ nodes[d][i_d], nodes[d][i_d + 1] ]
    """

    def __init__(self, nodes):
        """
        nodes: sequence of 1D numpy arrays, one per dimension, each strictly increasing.
        """
        self.nodes = [np.asarray(a, dtype=float) for a in nodes]
        self.n = len(self.nodes)

        # number of elements per dimension
        self.N = np.array([a.size - 1 for a in self.nodes], dtype=np.int64)
        if np.any(self.N <= 0):
            raise ValueError("Each nodes[d] must have length >= 2.")

        # row-major strides: last dimension varies fastest
        self.strides = np.empty(self.n, dtype=np.int64)
        prod = 1
        for d in range(self.n - 1, -1, -1):
            self.strides[d] = prod
            prod *= int(self.N[d])
        self.num_elems = int(prod)

        # Optional sanity checks
        for d, a in enumerate(self.nodes):
            if not np.all(np.isfinite(a)):
                raise ValueError(f"nodes[{d}] contains non-finite values.")
            if np.any(np.diff(a) <= 0):
                raise ValueError(f"nodes[{d}] must be strictly increasing.")

    def unravel(self, e):
        """
        Convert flat element index/indices to multi-index array of shape (..., n).
        """
        e = np.asarray(e, dtype=np.int64)
        if np.any(e < 0) or np.any(e >= self.num_elems):
            raise IndexError("Element index out of range.")

        # Broadcast-friendly unravel
        idx = np.empty(e.shape + (self.n,), dtype=np.int64)
        for d in range(self.n):
            idx[..., d] = (e // self.strides[d]) % self.N[d]
        return idx

    def endpoints(self, e):
        """
        Get per-dimension endpoints for element index/indices.

        Returns:
          lo, hi: arrays of shape (..., n)
            lo[..., d] = left/lower endpoint in dimension d
            hi[..., d] = right/upper endpoint in dimension d
        """
        idx = self.unravel(e)  # (..., n)

        lo = np.empty_like(idx, dtype=float)
        hi = np.empty_like(idx, dtype=float)

        # Gather per dimension (vectorized)
        for d in range(self.n):
            i = idx[..., d]
            a = self.nodes[d]
            lo[..., d] = a[i]
            hi[..., d] = a[i + 1]
        return lo, hi

    def bbox(self, e):
        """
        Convenience: return bounding boxes as (..., 2, n) with [lo; hi].
        """
        lo, hi = self.endpoints(e)
        return np.stack([lo, hi], axis=-2)


# ---- Example ----
if __name__ == "__main__":
    # 3D nonuniform grid
    x = np.array([0.0, 0.2, 0.7, 1.0])      # N1 = 3 elems
    y = np.array([-1.0, 0.0, 2.0])          # N2 = 2 elems
    z = np.array([10.0, 11.0, 11.5, 13.0])  # N3 = 3 elems

    mesh = RectMesh([x, y, z])
    print("num_elems =", mesh.num_elems)  # 3*2*3 = 18

    e = 7
    idx = mesh.unravel(e)
    lo, hi = mesh.endpoints(e)
    print("e =", e, "idx =", idx)         # (i1,i2,i3)
    print("lo =", lo, "hi =", hi)

    lo, hi = mesh.endpoints(np.arange(mesh.num_elems))
    print("lo shape:", lo.shape)           # (num_elems, n)
    print("hi shape:", hi.shape)           # (num_elems, n)

    # Batch query
    E = np.array([0, 1, 7, 17])
    boxes = mesh.bbox(E)                  # (len(E), 2, 3)
    print("boxes shape:", boxes.shape)
    print(boxes)
