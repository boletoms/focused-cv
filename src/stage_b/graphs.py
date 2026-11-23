"""N7: CUDA graphs for the fast focus budget (batch 1, fixed frame size). On small backbones the budget's saving is
lost to kernel launches and Python (N1: about 2.3 ms per frame on ViT-S). A graph replays a whole forward pass with one
launch, but needs fixed shapes; the protected count varies per frame, so it is rounded up to a size class (multiple of
`step`, `promote` in methods.py) and one graph is captured per class, on first use. Per frame outside the graph: the
dilation, one host sync for the count, the promotion and two copies into static buffers. Dense gets one graph too, so
the comparison is graph against graph. Usage: GraphRunner(model, name, (H, W)).run(x, interest)"""
import torch

from .methods import dilate_interest, promote
from .model import State
from .timing import from_name


class GraphRunner:
    def __init__(self, model, name, hw, dtype=torch.bfloat16, warmup=3):
        self.model, self.name, self.hw, self.dtype, self.warmup = model, name, hw, dtype, warmup
        self.blocks = len(model.bb.model.layer)
        self.grid = (hw[0] // 16, hw[1] // 16)
        self.graphs = {}                           # class size (or "dense") -> (graph, x, interest, out)
        self.pool = torch.cuda.graph_pool_handle()
        probe = from_name(name, self.grid[0], self.blocks) if name != "dense" else None
        self.static = probe is not None and not getattr(probe, "per_image", False)   # R8: fixed-shape baselines
        self.step = getattr(probe, "size_class", 0) if probe is not None else 0
        self.theta = getattr(probe, "theta", 0.01) if probe is not None else 0.01
        self.dilate = getattr(probe, "dilate", 0) if probe is not None else 0
        assert name == "dense" or self.static or self.step, "graphs need a size class (name ending _cls<step>)"

    def _capture(self, key, x, interest):
        if key == "dense":
            m = None
        elif key == "static":
            m = from_name(self.name, self.grid[0], self.blocks)
        else:
            m = from_name(self.name, self.grid[0], self.blocks)
            m.dilate, m.n_high_fixed = 0, key      # dilation and promotion happen outside the graph
        sx = x.clone()
        si = None if interest is None else interest.clone()
        fwd = lambda: self.model(sx, structure=m, dtype=self.dtype, interest=si).final
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(self.warmup):
                fwd()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool), torch.no_grad():
            out = fwd()
        self.graphs[key] = (g, sx, si, out)

    def prepare(self, interest):
        """Dilation and promotion (outside the graph): returns (class size, promoted map)."""
        if self.name == "dense":
            return "dense", None
        if self.static:
            return "static", None
        st = State(size=None, coords=None, src=None, carried=None, pairs=[], tokens=[], tiles=[],
                   interest=interest.float())
        st.grid = self.grid
        dilate_interest(st, self.dilate)
        return tuple(reversed(promote(st.interest, self.theta, self.step, self.grid)))

    @torch.no_grad()
    def run(self, x, interest=None):
        key, imap = self.prepare(interest)
        if key not in self.graphs:
            self._capture(key, x, imap)
        g, sx, si, out = self.graphs[key]
        sx.copy_(x)
        if si is not None:
            si.copy_(imap)
        g.replay()
        return out
