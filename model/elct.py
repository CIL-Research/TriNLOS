import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as ssp
from typing import Sequence, Tuple, Union


class Elct(nn.Module):
    """
    Learnable-parameter LCT volume reconstructor.

    This variant *only* harmonizes the coordinate conventions and FFT centering
    so that (a) PSF / LoG filters are centered correctly and (b) padding/crop
    rules are symmetric. The goal: increasing filter_sizes will not induce
    deterministic pixel shifts.

    Key changes vs original:
      - enforce ifftshift(psf) before fftn and keep fpsf as complex64
      - build LoG kernels at pad_shape center and ifftshift them before fftn
      - use symmetric padding (centered) and center-crop after ifftn
      - explicit fftshift/ifftshift helpers for clarity
      - robust broadcasting for fusion parameters

    This file focuses on coordinate fixes only (no structural algorithmic
    changes). If you want additional stability tricks (win/windowing,
    gradient clipping, lower LR for fusion params) we can add them separately.
    """

    def __init__(
        self,
        *,
        fixed_shape: Tuple[int, int, int],          # (T, H, W)
        wall_size: float = 2.0,
        bin_len: float = 32e-12 * 3e8,
        material: Union[float, torch.Tensor, nn.Parameter] = 2.0,
        filter_sizes: Sequence[int] = (3,),
        fk_eps: float = 0.05,
        use_log_filter: bool = True,
        use_fk_mask: bool = True,
        use_weighted_fusion: bool = True,
        fusion_scale: Union[float, torch.Tensor, nn.Parameter] = 0.25,
        fusion_power: Union[float, torch.Tensor, nn.Parameter] = 2.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.wall_size          = wall_size
        self.crop               = fixed_shape[0]
        self.bin_len            = bin_len

        # allow external Parameter or constant
        self.material     = material if isinstance(material, (torch.Tensor, nn.Parameter)) \
                           else nn.Parameter(torch.tensor(material, dtype=torch.float32))
        self.fusion_scale = fusion_scale if isinstance(fusion_scale, (torch.Tensor, nn.Parameter)) \
                           else nn.Parameter(torch.tensor(fusion_scale, dtype=torch.float32))
        self.fusion_power = fusion_power if isinstance(fusion_power, (torch.Tensor, nn.Parameter)) \
                           else nn.Parameter(torch.tensor(fusion_power, dtype=torch.float32))

        self.filter_sizes       = list(filter_sizes)
        self.use_log_filter     = use_log_filter
        self.use_fk_mask        = use_fk_mask
        self.use_weighted_fusion = use_weighted_fusion
        self.fk_eps             = fk_eps
        self.device             = device

        self._prepare_static_buffers(fixed_shape)
        self.relu = nn.ReLU()

    # -------------------- helper shifts --------------------
    @staticmethod
    def _fftshift(x, dim=None):
        return torch.fft.fftshift(x, dim=dim)

    @staticmethod
    def _ifftshift(x, dim=None):
        return torch.fft.ifftshift(x, dim=dim)

    # ------------------------------------------------------------------
    def _prepare_static_buffers(self, fixed_shape: Tuple[int, int, int]):
        M, H, W = fixed_shape
        assert H == W, "Only square spatial grids are supported."
        N = H

        c = 3e8
        width = self.wall_size / 2.0
        bin_resolution = self.bin_len / c
        trange = M * c * bin_resolution
        slope = width / trange

        # pad shape used for frequency-domain ops: use even sizes (2*M, 2*N, 2*N)
        pad_shape = (2 * M, 2 * N, 2 * N)
        self.pad_shape = pad_shape

        # ---------------- PSF & its FFT (centered convention) ----------------
        # definePsf returns a kernel whose peak is in the array center. To get
        # the correct frequency-domain representation for convolution we place
        # the kernel with origin at array[0,0,0] via ifftshift, then fftn.
        psf = torch.tensor(self.definePsf(N, M, slope), dtype=torch.float32, device=self.device)
        psf = self._ifftshift(psf)                 # place origin at index 0 for FFT conv
        fpsf = torch.fft.fftn(psf, s=pad_shape).to(torch.complex64)
        self.register_buffer("fpsf", fpsf, persistent=False)

        # ---------------- Resampling matrices ----------------
        mtx, mtxi = self.resamplingOperator(M)
        self.register_buffer("mtx",  torch.tensor(mtx,  dtype=torch.float32, device=self.device), persistent=False)
        self.register_buffer("mtxi", torch.tensor(mtxi, dtype=torch.float32, device=self.device), persistent=False)

        # ---------------- z-grid base (no exponent yet) ----------------
        gridz_base = torch.linspace(0, 1, M, device=self.device).view(-1, 1, 1).expand(M, N, N)
        self.register_buffer("gridz_base", gridz_base, persistent=False)

        # ---------------- fk-mask ----------------
        if self.use_fk_mask:
            fk = self.soft_light_cone_mask(pad_shape, eps=self.fk_eps, sigma=0.02, device=self.device)
            # fk is a real mask in frequency domain, keep float32
            self.register_buffer("fk_mask", fk.to(torch.float32), persistent=False)

        # ---------------- LoG filters (FFT) - build shift-correct kernels ----------------
        H_logs = []
        for sz in self.filter_sizes:
            # ensure odd size for symmetric kernel
            if isinstance(sz, int):
                if sz % 2 == 0:
                    sz = sz + 1
                kern = self.log_filter(sz)
            else:
                kern = self.log_filter(sz)

            # create zero-padded kernel placed in pad array center
            k = torch.zeros(pad_shape, dtype=torch.float32, device=self.device)
            cz, cy, cx = pad_shape[0] // 2, pad_shape[1] // 2, pad_shape[2] // 2
            r0 = kern.shape[0] // 2
            k[cz - r0: cz - r0 + kern.shape[0],
              cy - r0: cy - r0 + kern.shape[1],
              cx - r0: cx - r0 + kern.shape[2]] = torch.tensor(kern, dtype=torch.float32, device=self.device)

            # important: if kernel was built centered, shift it so FFT uses origin-at-zero convention
            k = self._ifftshift(k)
            H_logs.append(torch.fft.fftn(k, s=pad_shape).to(torch.complex64))

        if len(H_logs) == 0:
            H_logs = [torch.ones(pad_shape, dtype=torch.complex64, device=self.device)]

        # stack as (F, 2M, 2N, 2N)
        H_logs_stack = torch.stack(H_logs, dim=0)
        self.register_buffer("H_logs_stack", H_logs_stack, persistent=False)

        self.num_filters = len(self.filter_sizes)
        self.M, self.N = M, N

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B,C,T,H,W)
        x = self.relu(x).float()

        B, C, _, _, _ = x.shape

        # grid exponent per-channel then expanded to (B*C, M, N, N)
        gridz_pow_c = self.gridz_base.unsqueeze(0) ** self.material.view(C, 1, 1, 1)      # (C,M,N,N)
        gridz_pow_bc = gridz_pow_c.unsqueeze(0).repeat(B, 1, 1, 1, 1).view(-1, self.M, self.N, self.N)

        # reorder to (B·C, N, N, M)
        x_batched = (
            x.permute(0, 1, 3, 4, 2)      # (B,C,H,W,T)
             .contiguous()
             .view(B * C, self.N, self.N, self.M)
        )

        if self.training:
            vol_batched = self._lct_precomputed_batch(x_batched, gridz_pow_bc)
        else:
            with torch.no_grad():
                vol_batched = self._lct_precomputed_batch(x_batched, gridz_pow_bc)
        vol = vol_batched.view(B, C, *vol_batched.shape[1:])                 # (B,C,M,N,N)

        # optional: kill extreme near/far slices
        vol[..., :5, :, :] = 0
        vol[..., -5:, :, :] = 0
        return vol.abs()

    def _lct_precomputed_batch(
        self,
        data_hw_t: torch.Tensor,   # (B·C, N, N, M)
        gridz_pow:  torch.Tensor,  # (B·C, M, N, N)
    ) -> torch.Tensor:
        Bc, H, W, T = data_hw_t.shape
        M, N = self.M, self.N
        pad_shape = self.pad_shape

        # 1) z-correction: permute and multiply
        data = data_hw_t.permute(0, 3, 1, 2) * gridz_pow   # (B·C, M, N, N)
        data_rs = data.view(Bc, M, N * N)

        # 2) resampling (to z-grid)
        vol_z = torch.einsum('mn,bnq->bmq', self.mtx, data_rs)
        vol_z = vol_z.view(Bc, M, N, N)   # (B·C, M, N, N)

        # 3) symmetric center padding -> pad tuple expects (W_left,W_right,H_left,H_right,T_left,T_right)
        # we use symmetric pad around center so cropping after ifft is consistent
        pad = (N // 2, N // 2, N // 2, N // 2, M // 2, M // 2)
        datapad = F.pad(vol_z, pad)

        # 4) FFT
        data_fft = torch.fft.fftn(datapad, dim=(-3, -2, -1))  # (Bc, 2M, 2N, 2N)

        # multiply by PSF (freq-domain). fpsf is complex64 and already in pad_shape
        base_fft = data_fft * self.fpsf.conj()

        # 5) apply LoG filters in frequency domain
        # H_logs_stack: (F, 2M, 2N, 2N) complex64
        H_logs = self.H_logs_stack.unsqueeze(1)   # (F, 1, 2M, 2N, 2N)
        fft_common = base_fft.unsqueeze(0) * H_logs  # (F, Bc, 2M, 2N, 2N)

        # 6) inverse FFT -> bring to spatial domain
        # we expect ifftn result with origin-at-zero; we shift it so center is at array center
        r = torch.fft.ifftn(fft_common, dim=(-3, -2, -1))  # complex
        r = r.real  # use real part for responses

        # center the volume (equivalent to bringing origin to center) then crop central M,N,N
        r = self._fftshift(r, dim=(-3, -2, -1))  # (F,Bc,2M,2N,2N)

        # crop central region
        cz, cy, cx = r.shape[-3] // 2, r.shape[-2] // 2, r.shape[-1] // 2
        r_cropped = r[..., cz - M // 2: cz - M // 2 + M,
                        cy - N // 2: cy - N // 2 + N,
                        cx - N // 2: cx - N // 2 + N]   # (F,Bc,M,N,N)

        # 7) compute energies per filter
        w = (r_cropped.abs() ** 2).sum(dim=(-3, -2, -1))  # (F, Bc)

        # if fk_mask is enabled, compute masked version as well
        if self.use_fk_mask:
            fk = self.fk_mask.view(1, 1, *self.fk_mask.shape)  # (1,1,2M,2N,2N)
            r_m = torch.fft.ifftn(fft_common * fk, dim=(-3, -2, -1)).real
            r_m = self._fftshift(r_m, dim=(-3, -2, -1))
            r_m_cropped = r_m[..., cz - M // 2: cz - M // 2 + M,
                                cy - N // 2: cy - N // 2 + N,
                                cx - N // 2: cx - N // 2 + N]
            w_m = (r_m_cropped.abs() ** 2).sum(dim=(-3, -2, -1))  # (F, Bc)

        # helper aggregator: softmax across filters then mtxi
        def _aggregate(resp_F_Bc_MNN, weight_F_Bc):
            attn = torch.softmax(weight_F_Bc, dim=0)            # (F, Bc)
            vol = (attn[..., None, None, None] * resp_F_Bc_MNN).sum(dim=0)  # (Bc, M, N, N)
            vol = torch.einsum('mn,bnq->bmq', self.mtxi,
                                vol.view(Bc, M, -1)).view(Bc, M, N, N)
            return vol

        vol_unm = _aggregate(r_cropped, w)
        if not self.use_fk_mask:
            return vol_unm

        vol_msk = _aggregate(r_m_cropped, w_m)

        # 8) weighted fusion: robust broadcasting across B,C
        if self.use_weighted_fusion:
            # fusion_scale / fusion_power can be scalar or channel-wise
            fs = self.fusion_scale
            fp = self.fusion_power

            # make fs, fp arrays of length Bc robustly
            if fs.numel() == 1:
                fs_rep = fs.item() * torch.ones(Bc, device=vol_unm.device)
            elif fs.numel() == vol_unm.shape[0] // 1:  # unlikely; fallback to repeat per batch
                fs_rep = fs.view(1, -1).repeat(Bc // fs.numel(), 1).view(-1).to(vol_unm.device)
            else:
                # treat as per-channel: len(fs) == C
                C = fs.numel()
                rep = Bc // C
                fs_rep = fs.repeat(rep).to(vol_unm.device)

            if fp.numel() == 1:
                fp_rep = fp.item() * torch.ones(Bc, device=vol_unm.device)
            elif fp.numel() == vol_unm.shape[0] // 1:
                fp_rep = fp.view(1, -1).repeat(Bc // fp.numel(), 1).view(-1).to(vol_unm.device)
            else:
                C = fp.numel()
                rep = Bc // C
                fp_rep = fp.repeat(rep).to(vol_unm.device)

            # create depth mask (Bc, M, 1, 1)
            z = torch.linspace(0.0, 1.0, M, device=vol_unm.device)
            # guard numerical stability
            eps = 1e-6
            fs_rep = fs_rep.clamp(min=eps)
            fp_rep = fp_rep.clamp(min=eps)
            depth_mask = 1.0 / (1.0 + (z[None, :] / fs_rep[:, None]) ** fp_rep[:, None])
            depth_mask = depth_mask.view(Bc, M, 1, 1)

            return depth_mask * vol_msk + (1.0 - depth_mask) * vol_unm

        return vol_msk

    # ------------------------------------------------------------------ static helpers
    @staticmethod
    def log_filter(size: Union[int, Tuple[int, int, int]]):
        if isinstance(size, int):
            size = (size, size, size)
        std = np.array(size) / (4 * np.sqrt(2 * np.log(2)))
        mid = [(s - 1) // 2 for s in size]
        z, y, x = np.meshgrid(
            np.arange(-mid[0], mid[0] + 1),
            np.arange(-mid[1], mid[1] + 1),
            np.arange(-mid[2], mid[2] + 1),
            indexing="ij",
        )
        r2 = x**2 + y**2 + z**2
        sigma = np.mean(std)
        gaussian = np.exp(-r2 / (2 * sigma**2))
        kernel = ((r2 - 3 * sigma**2) * gaussian) / (sigma**5)
        kernel -= kernel.mean()
        return kernel.astype(np.float32)

    def soft_light_cone_mask(self,shape, eps=0.05, sigma=0.02, device="cuda"):
        kz, ky, kx = torch.meshgrid(
            torch.fft.fftfreq(shape[0], device=device),
            torch.fft.fftfreq(shape[1], device=device),
            torch.fft.fftfreq(shape[2], device=device),
            indexing="ij",
        )
        kx = torch.fft.fftshift(kx); ky = torch.fft.fftshift(ky); kz = torch.fft.fftshift(kz)
        k_mag = torch.sqrt(kx**2 + ky**2 + kz**2) + 1e-12
        cone_diff = torch.abs(kz - k_mag) / k_mag
        mask = torch.sigmoid(-(cone_diff - eps)/sigma)
        return torch.fft.ifftshift(mask)

    @staticmethod
    def light_cone_mask(shape: Tuple[int, int, int], eps: float = 0.05):
        kz, ky, kx = np.meshgrid(
            np.fft.fftfreq(shape[0]),
            np.fft.fftfreq(shape[1]),
            np.fft.fftfreq(shape[2]),
            indexing="ij",
        )
        kx, ky, kz = np.fft.fftshift(kx), np.fft.fftshift(ky), np.fft.fftshift(kz)
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2) + 1e-12
        cone_diff = np.abs(kz - k_mag) / k_mag
        mask = (cone_diff < eps).astype(np.float32)
        return np.fft.ifftshift(mask)

    @staticmethod
    def definePsf(N: int, M: int, slope: float):
        x = np.linspace(-1, 1, 2 * N)
        z = np.linspace(0, 2, 2 * M)
        gridz, gridy, gridx = np.meshgrid(z, x, x, indexing="ij")
        a = (4 * slope) ** 2 * (gridx**2 + gridy**2) - gridz
        b = np.abs(a)
        c = b.min(axis=0, keepdims=True)
        d = (np.abs(b - c) < 1e-8).astype(np.float32)
        e = d / np.sqrt(d.sum())
        e = np.roll(e, N, axis=1)
        e = np.roll(e, N, axis=2)
        return e

    @staticmethod
    def spatio_temporal_downsample(
        data: torch.Tensor,
        crop: int,
        bin_len: float,
        *,
        down_time: int = 2,
        down_space: int = 1,
    ):
        if down_space > 1:
            for _ in range(int(np.log2(down_space))):
                data = data[::2] + data[1::2]
                data = data[:, ::2] + data[:, 1::2]
        if down_time > 1:
            H, W, T = data.shape
            data = data.view(H, W, T // down_time, down_time).sum(dim=3)
        crop //= down_time
        bin_len *= down_time
        return data, crop, bin_len

    @staticmethod
    def resamplingOperator(M: int):
        row = M**2
        col = M
        x = np.arange(row, dtype=np.float32) + 1
        rowidx = np.arange(row)
        colidx = np.ceil(np.sqrt(x)) - 1
        data = np.ones_like(rowidx, dtype=np.float32)
        mtx1 = ssp.csr_matrix((data, (rowidx, colidx)), shape=(row, col))
        mtx2 = ssp.spdiags([1.0 / np.sqrt(x)], [0], row, row)
        mtx = (mtx2 @ mtx1).tocoo()
        for _ in range(int(np.log2(M))):
            mtx = 0.5 * (mtx.tocsr()[0::2] + mtx.tocsr()[1::2])
        return mtx.toarray(), mtx.T.toarray()
