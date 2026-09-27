"""Batch-local distance cache shared by observation and shielding."""


class MapServices:
    def __init__(self, batch, *, cache_mode="bounded", **cost_to_go_options):
        if cache_mode not in {"bounded", "vendor-window"}:
            raise ValueError("cache mode must be bounded or vendor-window")
        self.batch = batch
        self.cache_mode = cache_mode
        self.options = cost_to_go_options
        self._cost_to_go = None

    def cost_to_go(self):
        """One shared cache; generation/state invalidation remains cache-owned."""
        if self._cost_to_go is None:
            if self.cache_mode == "vendor-window":
                from .vendor_windowed import CUDAVendorWindowCostToGo
                self._cost_to_go = CUDAVendorWindowCostToGo(self.batch, **self.options)
                return self._cost_to_go
            if max(self.batch.width, self.batch.height) > 74:
                from .windowed import CUDAWindowCostToGo
                factory = CUDAWindowCostToGo
            else:
                from .cuda import CUDACostToGo
                factory = CUDACostToGo
            self._cost_to_go = factory(self.batch, **self.options)
        return self._cost_to_go

    def metadata(self):
        return {"requested": self.cache_mode,
                "selected": None if self._cost_to_go is None else type(self._cost_to_go).__name__}
