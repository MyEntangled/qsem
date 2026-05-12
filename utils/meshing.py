import numpy as np
from itertools import product

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
        nodes:
          - nD case: sequence of 1D arrays, one per dimension
          - 1D case: a single 1D array/list of nodal coordinates
        """
        if isinstance(nodes, np.ndarray) and nodes.ndim == 1:
            node_arrays = [np.asarray(nodes, dtype=float)]
        else:
            nodes_seq = list(nodes)
            if len(nodes_seq) == 0:
                raise ValueError("nodes must not be empty.")

            if np.isscalar(nodes_seq[0]):
                # 1D convenience form: RectMesh([x0, x1, ...]) or RectMesh((...))
                node_arrays = [np.asarray(nodes_seq, dtype=float)]
            else:
                node_arrays = [np.asarray(a, dtype=float) for a in nodes_seq]

        self.nodes = node_arrays
        self.n = len(self.nodes)
        self.is_1d = (self.n == 1)

        # Optional sanity checks
        for d, a in enumerate(self.nodes):
            if not np.all(np.isfinite(a)):
                raise ValueError(f"nodes[{d}] contains non-finite values.")
            if np.any(np.diff(a) <= 0):
                raise ValueError(f"nodes[{d}] must be strictly increasing.")

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

        self.endpoints = self.bbox(np.arange(self.num_elems))  # precompute all endpoints

        # Check uniformity: all elements have the same size in each dimension
        self.is_uniform = all(
            np.allclose(np.diff(a), np.diff(a)[0]) for a in self.nodes
        )




    def unravel(self, e):
        """
        Convert flat element index/indices to multi-index array.

        Returns
        -------
        nD: array of shape (..., n)
        1D: array/scalar of shape (...) (the single-axis element index)
        """
        e = np.asarray(e, dtype=np.int64)
        if np.any(e < 0) or np.any(e >= self.num_elems):
            raise IndexError("Element index out of range.")

        # Broadcast-friendly unravel
        idx = np.empty(e.shape + (self.n,), dtype=np.int64)
        for d in range(self.n):
            idx[..., d] = (e // self.strides[d]) % self.N[d]
        if self.is_1d:
            return idx[..., 0]
        return idx

    def get_endpoints(self, e):
        """
        Get per-dimension endpoints for element index/indices.

        Returns:
          nD:
            lo, hi: arrays of shape (..., n)
              lo[..., d] = left/lower endpoint in dimension d
              hi[..., d] = right/upper endpoint in dimension d
          1D:
            lo, hi: arrays/scalars of shape (...)
        """
        if self.is_1d:
            i = np.asarray(self.unravel(e), dtype=np.int64)
            a = self.nodes[0]
            lo = a[i]
            hi = a[i + 1]
            return lo, hi

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
        Convenience bounding boxes.

        Returns
        -------
        nD: array of shape (..., 2, n) with [lo; hi]
        1D: array of shape (..., 2) with [lo, hi]
        """
        lo, hi = self.get_endpoints(e)
        if self.is_1d:
            return np.stack([lo, hi], axis=-1)
        return np.stack([lo, hi], axis=-2)

    def find_elements(self, coords):
        """
        Return element index/indices whose bounding boxes contain `coords`.

        Parameters
        ----------
        coords :
            nD: sequence length n with one coordinate per axis
            1D: scalar (or length-1 sequence)
            Use `None` to mean "any value" on that axis.
            Example (3D): `(x0, y0, z0)`, `(x0, None, None)`.

        Returns
        -------
        int | list[int]
            Single element index if exactly one match exists, otherwise a list of
            matching indices. Returns `[]` if no element contains the query.

        Notes
        -----
        Intervals are treated as half-open in each axis: `[nodes[d][i], nodes[d][i+1])`,
        except the global upper endpoint `nodes[d][-1]`, which is assigned to the
        final element in that axis.
        A point on an internal grid node belongs to the element that starts at that
        node (lower-endpoint owner), not the element ending there.
        """
        if self.is_1d and (coords is None or np.isscalar(coords)):
            coords = (coords,)

        if len(coords) != self.n:
            raise ValueError(f"Expected {self.n} coordinates, got {len(coords)}.")

        per_dim_candidates = []

        for d, x in enumerate(coords):
            if x is None:
                per_dim_candidates.append(np.arange(self.N[d], dtype=np.int64))
                continue

            x = float(x)
            a = self.nodes[d]
            n_elem_d = int(self.N[d])

            if x < a[0] or x > a[-1]:
                return []

            if x == a[-1]:
                # Global upper endpoint belongs to the final element.
                i = n_elem_d - 1
            else:
                # Unique half-open interval index satisfying a[i] <= x < a[i+1]
                i = int(np.searchsorted(a, x, side="right") - 1)
                if i < 0 or i >= n_elem_d:
                    return []
            per_dim_candidates.append(np.asarray([i], dtype=np.int64))

        matches = []
        for multi_idx in product(*per_dim_candidates):
            e = int(sum(int(i_d) * int(self.strides[d]) for d, i_d in enumerate(multi_idx)))
            matches.append(e)

        if not matches:
            return []
        if len(matches) == 1:
            return matches[0]
        return matches


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
    lo, hi = mesh.get_endpoints(e)
    print("e =", e, "idx =", idx)         # (i1,i2,i3)
    print("lo =", lo, "hi =", hi)

    lo, hi = mesh.get_endpoints(np.arange(mesh.num_elems))
    print("lo shape:", lo.shape)           # (num_elems, n)
    print("hi shape:", hi.shape)           # (num_elems, n)

    # Batch query
    E = np.array([0, 1, 7, 17])
    boxes = mesh.bbox(E)                  # (len(E), 2, 3)
    print("boxes shape:", boxes.shape)
    print(boxes)

    print("Endpoints:")
    print(mesh.endpoints)

    a = mesh.find_elements((0.1, 0.5, 11.2))
    b = mesh.find_elements((0.2, 0.0, 10.0))  # lower-endpoint owner => unique element
    c = mesh.find_elements((0.5, None, 12.0))
    d = mesh.find_elements((1.5, 0.0, 10.0))
    e = mesh.find_elements((1.0, 0.0, 10.0))  # global upper endpoint in x -> final x element
    print(a, b, c, d, e)

    mesh1d = RectMesh(x)
    print("1D endpoints shape:", mesh1d.endpoints.shape)
    print(mesh1d.endpoints)
    print("1D find:", mesh1d.find_elements(0.2), mesh1d.find_elements(1.0), mesh1d.find_elements(100))
