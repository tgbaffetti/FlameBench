from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.extmath import randomized_svd

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# TransformerSequence — nucleo condiviso: processa A_w (coefficienti latenti)
# e phi (forzante) esattamente come nello script originale. NON TOCCARE.
# =============================================================================
class TransformerSequence(nn.Module):
    """
    Sequenza di token: [phi_1, a_{1,1..r}, phi_2, a_{2,1..r}, ..., phi_n, a_{n,1..r},
                         phi_{n+1}, PRED_1, ..., PRED_r]
    Output: delta_a (B, r) — residuo predetto per ogni componente latente.
    "rank_pod" qui indica genericamente la dimensione dello spazio latente
    (che sia POD o AE encoder non fa differenza per il Transformer).
    """

    TOKEN_PHI, TOKEN_POD, TOKEN_PRED = 0, 1, 2
    SEG_PAST, SEG_FUTURE = 0, 1

    def __init__(self, rank_pod, n_past, n_layers=4, embed_dim=128, num_heads=8, hidden_dim=256):
        super().__init__()
        self.rank_pod = rank_pod
        self.n_past = n_past
        self.embed_dim = embed_dim
        self.L = n_past * (rank_pod + 1) + 1 + rank_pod

        self.in_proj_phi = nn.Linear(1, embed_dim)
        self.in_proj_pod = nn.Linear(1, embed_dim)
        self.in_proj_pred = nn.Linear(1, embed_dim)

        self.type_embed = nn.Embedding(3, embed_dim)
        self.pos_embed = nn.Embedding(self.L, embed_dim)
        self.seg_embed = nn.Embedding(2, embed_dim)

        self.pred_token = nn.Parameter(torch.zeros(rank_pod, 1))

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha": nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim),
                ),
                "norm1": nn.LayerNorm(embed_dim),
                "norm2": nn.LayerNorm(embed_dim),
            })
            for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(embed_dim, 1)

    def _build_type_ids(self, device):
        ids = []
        for _ in range(self.n_past):
            ids.append(self.TOKEN_PHI)
            ids.extend([self.TOKEN_POD] * self.rank_pod)
        ids.append(self.TOKEN_PHI)
        ids.extend([self.TOKEN_PRED] * self.rank_pod)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _build_seg_ids(self, device):
        n_past_tokens = self.n_past * (self.rank_pod + 1)
        ids = [self.SEG_PAST] * n_past_tokens + [self.SEG_FUTURE] * (1 + self.rank_pod)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, a_history, phi_context, return_attn=False):
        """
        a_history   : list di n_past tensori (B, r)
        phi_context : (B, n_past+1)
        """
        B = a_history[0].shape[0]
        device = a_history[0].device

        embedded = []
        for i in range(self.n_past):
            embedded.append(self.in_proj_phi(phi_context[:, i:i + 1]))
            for k in range(self.rank_pod):
                embedded.append(self.in_proj_pod(a_history[i][:, k:k + 1]))
        embedded.append(self.in_proj_phi(phi_context[:, self.n_past:self.n_past + 1]))

        x_past_emb = torch.stack(embedded, dim=1)
        pred_emb = self.in_proj_pred(self.pred_token.unsqueeze(0).expand(B, -1, -1))
        x = torch.cat([x_past_emb, pred_emb], dim=1)

        pos_ids = torch.arange(self.L, device=device)
        type_ids = self._build_type_ids(device)
        seg_ids = self._build_seg_ids(device)

        x = (x + self.type_embed(type_ids).unsqueeze(0)
               + self.pos_embed(pos_ids).unsqueeze(0)
               + self.seg_embed(seg_ids).unsqueeze(0))

        attn_weights_all = []
        for layer in self.layers:
            if return_attn:
                attn_out, attn_w = layer["mha"](x, x, x, need_weights=True, average_attn_weights=True)
                attn_weights_all.append(attn_w)
            else:
                attn_out, _ = layer["mha"](x, x, x, need_weights=False)
            x = layer["norm1"](x + attn_out)
            x_ff = layer["ff"](x)
            x = layer["norm2"](x + x_ff)

        pred_out = x[:, -self.rank_pod:, :]
        delta_a = self.out_proj(pred_out).squeeze(-1)

        if return_attn:
            return delta_a, attn_weights_all
        return delta_a


# =============================================================================
# POD Reducer — lineare, fittato una volta sola (non allenabile)
# =============================================================================
class PODReducer:
    """
    snapshots : (Nt, Nf, Ncells)  — flatten interno a D = Nf*Ncells.
    Funziona indipendentemente dal fatto che la griglia sia strutturata o meno,
    perché opera solo sulla dimensione flattenata.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.U_r: np.ndarray | None = None
        self.mean: np.ndarray | None = None

    def fit(self, snapshots: np.ndarray) -> np.ndarray:
        Nt, Nf, Ncells = snapshots.shape
        D = Nf * Ncells
        X = snapshots.reshape(Nt, D).T
        self.mean = X.mean(axis=1, keepdims=True)
        X_c = X - self.mean
        r = min(self.rank, min(D, Nt))
        U_r, S_r, Vt_r = randomized_svd(X_c, n_components=r, n_oversamples=20, n_iter=7, random_state=42)
        self.U_r = U_r
        return S_r[:, None] * Vt_r

    def to_torch(self, device):
        if not hasattr(self, "_cached_device") or self._cached_device != str(device):
            self._U_r_t = torch.tensor(self.U_r, dtype=torch.float32, device=device)
            self._mean_t = torch.tensor(self.mean.flatten(), dtype=torch.float32, device=device)
            self._cached_device = str(device)

    def encode_torch(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, Nf, Ncells) -> (B, r)"""
        B = x.shape[0]
        D = int(np.prod(x.shape[1:]))
        self.to_torch(x.device)
        x_c = x.reshape(B, D) - self._mean_t.unsqueeze(0)
        return x_c @ self._U_r_t

    def decode_torch(self, a: torch.Tensor, shape_out) -> torch.Tensor:
        """a : (B, r) -> (B, *shape_out)  — utile solo per inferenza/test."""
        B = a.shape[0]
        self.to_torch(a.device)
        X = a @ self._U_r_t.T + self._mean_t.unsqueeze(0)
        return X.reshape(B, *shape_out)


# =============================================================================
# MLP generico — encoder/decoder per il modello AE-Transformer
# Usa Linear layers perché la griglia non è più strutturata (CFD non uniforme).
# =============================================================================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.LeakyReLU):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Interfaccia comune per tutti i modelli
# =============================================================================
class BaseForecastModel(nn.Module):
    """
    x_window   : (B, n_past+1, Nf, Ncells)  — ultimi n_past = storia, ultimo = target
    phi_window : (B, n_past+1)              — phi(t-n_past+1..t) + phi(t+1)
    """

    def preprocess(self, train_snapshots: np.ndarray) -> dict:
        """Step opzionale non-gradiente (es. fit POD). Ritorna dict da salvare nel checkpoint."""
        return {}

    def requires_training_loop(self) -> bool:
        """False per modelli fittati in forma chiusa (es. POD-ARX)."""
        return True

    def fit_closed_form(self, train_dataset) -> dict:
        """Implementato solo da modelli con requires_training_loop()=False."""
        raise NotImplementedError

    def compute_loss(self, x_window: torch.Tensor, phi_window: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_checkpoint_data(self) -> dict:
        """Stato extra non incluso in state_dict() (es. basi POD, non sono nn.Parameter)."""
        return {}

    @classmethod
    def from_config(cls, cfg: dict, nf: int, n_cells: int, n_past: int, device):
        raise NotImplementedError


# =============================================================================
# POD-Transformer — IDENTICO a quello inviato in precedenza
# =============================================================================
class PODTransformerModel(BaseForecastModel):
    def __init__(self, rank_pod, n_past, n_layers, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.rank_pod = rank_pod
        self.n_past = n_past
        self.transformer = TransformerSequence(rank_pod, n_past, n_layers, embed_dim, num_heads, hidden_dim)
        self.pod = PODReducer(rank=rank_pod)

    def preprocess(self, train_snapshots: np.ndarray) -> dict:
        self.pod.fit(train_snapshots)
        return {
            "pod_U_r": self.pod.U_r,
            "pod_mean": self.pod.mean,
            "pod_rank": self.pod.U_r.shape[1],
        }

    def compute_loss(self, x_window, phi_window):
        a_history = [self.pod.encode_torch(x_window[:, i]) for i in range(self.n_past)]
        delta = self.transformer(a_history, phi_window)
        z_pred = a_history[-1] + delta
        z_true = self.pod.encode_torch(x_window[:, self.n_past])
        return F.mse_loss(z_pred, z_true)

    def extra_checkpoint_data(self) -> dict:
        return {"pod_U_r": self.pod.U_r, "pod_mean": self.pod.mean, "pod_rank": self.pod.U_r.shape[1]}

    @classmethod
    def from_config(cls, cfg, nf, n_cells, n_past, device):
        model = cls(
            rank_pod=cfg["rank_POD"],
            n_past=n_past,
            n_layers=cfg["transformer_layers"],
            embed_dim=cfg["transformer_embed_dim"],
            num_heads=cfg["transformer_heads"],
            hidden_dim=cfg["transformer_hidden_dim"],
        )
        return model.to(device)


# =============================================================================
# AE-Transformer — stesso TransformerSequence, ma la riduzione latente è
# un MLP encoder/decoder allenabile insieme al Transformer (joint training)
# =============================================================================
class AETransformerModel(BaseForecastModel):
    def __init__(self, data_dim, latent_dim, encoder_hidden, decoder_hidden,
                 n_past, n_layers, embed_dim, num_heads, hidden_dim,
                 weight_reconstruction=1.0, weight_latent_pred=1.0):
        super().__init__()
        self.n_past = n_past
        self.weight_reconstruction = weight_reconstruction
        self.weight_latent_pred = weight_latent_pred

        self.encoder = MLP(data_dim, encoder_hidden, latent_dim)
        self.decoder = MLP(latent_dim, decoder_hidden, data_dim)
        self.transformer = TransformerSequence(latent_dim, n_past, n_layers, embed_dim, num_heads, hidden_dim)

    def encode(self, x):
        B = x.shape[0]
        return self.encoder(x.reshape(B, -1))

    def decode(self, z, shape_out):
        B = z.shape[0]
        return self.decoder(z).reshape(B, *shape_out)

    def compute_loss(self, x_window, phi_window):
        shape_out = x_window.shape[2:]  # (Nf, Ncells)
        a_history = [self.encode(x_window[:, i]) for i in range(self.n_past)]
        delta = self.transformer(a_history, phi_window)
        z_pred = a_history[-1] + delta

        x_true = x_window[:, self.n_past]
        z_true = self.encode(x_true)
        latent_loss = F.mse_loss(z_pred, z_true)

        x_rec = self.decode(z_true, shape_out)
        rec_loss = F.mse_loss(x_rec, x_true)

        return self.weight_reconstruction * rec_loss + self.weight_latent_pred * latent_loss

    @classmethod
    def from_config(cls, cfg, nf, n_cells, n_past, device):
        data_dim = nf * n_cells
        model = cls(
            data_dim=data_dim,
            latent_dim=cfg["latent_dim"],
            encoder_hidden=cfg["encoder_hidden"],
            decoder_hidden=cfg["decoder_hidden"],
            n_past=n_past,
            n_layers=cfg["transformer_layers"],
            embed_dim=cfg["transformer_embed_dim"],
            num_heads=cfg["transformer_heads"],
            hidden_dim=cfg["transformer_hidden_dim"],
            weight_reconstruction=cfg.get("weight_reconstruction", 1.0),
            weight_latent_pred=cfg.get("weight_latent_pred", 1.0),
        )
        return model.to(device)



class LSTMModel(BaseForecastModel):
    """
    Storia POD-ridotta + phi passata -> LSTM -> concat phi_future -> delta_a.
    Stesso schema residuale degli altri modelli.
    """

    def __init__(self, rank_pod, n_past, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.rank_pod = rank_pod
        self.n_past = n_past
        self.pod = PODReducer(rank=rank_pod)
        self.lstm = nn.LSTM(
            input_size=rank_pod + 1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size), nn.GELU(), nn.Linear(hidden_size, rank_pod),
        )

    def preprocess(self, train_snapshots):
        self.pod.fit(train_snapshots)
        return {"pod_U_r": self.pod.U_r, "pod_mean": self.pod.mean, "pod_rank": self.pod.U_r.shape[1]}

    def compute_loss(self, x_window, phi_window):
        a_hist = torch.stack([self.pod.encode_torch(x_window[:, i]) for i in range(self.n_past)], dim=1)
        phi_hist = phi_window[:, :self.n_past].unsqueeze(-1)
        lstm_in = torch.cat([a_hist, phi_hist], dim=-1)          # (B, n_past, r+1)
        out, _ = self.lstm(lstm_in)
        last_hidden = out[:, -1, :]
        phi_future = phi_window[:, self.n_past:self.n_past + 1]  # (B, 1)
        delta = self.head(torch.cat([last_hidden, phi_future], dim=-1))
        z_pred = a_hist[:, -1, :] + delta
        z_true = self.pod.encode_torch(x_window[:, self.n_past])
        return F.mse_loss(z_pred, z_true)

    def extra_checkpoint_data(self):
        return {"pod_U_r": self.pod.U_r, "pod_mean": self.pod.mean, "pod_rank": self.pod.U_r.shape[1]}

    @classmethod
    def from_config(cls, cfg, nf, n_cells, n_past, device):
        model = cls(
            rank_pod=cfg["rank_POD"], n_past=n_past,
            hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
            dropout=cfg.get("dropout", 0.0),
        )
        return model.to(device)

class SpectralConv3d(nn.Module):
    """Fourier layer 3D — implementazione standard (Li et al., 2020)."""

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2, modes3)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    def _compl_mul3d(self, x, w):
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x):
        B = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        out_ft = torch.zeros(
            B, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft[:, :, :m1, :m2, :m3] = self._compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.weights1)
        out_ft[:, :, -m1:, :m2, :m3] = self._compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], self.weights2)
        out_ft[:, :, :m1, -m2:, :m3] = self._compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = self._compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.weights4)
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNO3d(nn.Module):
    """Architettura FNO3D esattamente come nel paper: lifting -> 4 Fourier layers -> projection."""

    def __init__(self, modes1, modes2, modes3, width, in_channels, out_channels):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)

        self.convs = nn.ModuleList([SpectralConv3d(width, width, modes1, modes2, modes3) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(4)])

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x : (B, D1, D2, D3, in_channels)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, width, D1, D2, D3)
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            if i < len(self.convs) - 1:
                x = F.gelu(x)
        x = x.permute(0, 2, 3, 4, 1)  # (B, D1, D2, D3, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)               # (B, D1, D2, D3, out_channels)
        return x


class FNO3DModel(BaseForecastModel):
    """
    Richiede una griglia STRUTTURATA (grid_shape=[Nz,Nx]) — non applicabile
    a mesh CFD non strutturate senza prima interpolare su griglia regolare.

    D3 = n_past (asse "storia", analogo all'asse T_in del paper originale).
    Il next-step è ottenuto prendendo l'ultima fetta lungo D3 in output.
    """

    def __init__(self, nf, grid_shape, n_past, modes1, modes2, modes3, width):
        super().__init__()
        self.nf = nf
        self.grid_shape = tuple(grid_shape)
        self.n_past = n_past
        in_channels = nf + 2  # Nf feature + 1 canale phi_hist + 1 canale phi_future (broadcast)
        self.fno = FNO3d(modes1, modes2, modes3, width, in_channels=in_channels, out_channels=nf)

    def _reshape_to_grid(self, x_flat):
        B = x_flat.shape[0]
        Nz, Nx = self.grid_shape
        return x_flat.reshape(B, self.nf, Nz, Nx)

    def compute_loss(self, x_window, phi_window):
        B = x_window.shape[0]
        Nz, Nx = self.grid_shape

        # (B, n_past, Nf, Nz, Nx) -> (B, Nz, Nx, n_past, Nf)
        hist = torch.stack([self._reshape_to_grid(x_window[:, i]) for i in range(self.n_past)], dim=1)
        hist = hist.permute(0, 3, 4, 1, 2)

        phi_hist = phi_window[:, :self.n_past].view(B, 1, 1, self.n_past, 1).expand(B, Nz, Nx, self.n_past, 1)
        phi_future = phi_window[:, self.n_past:self.n_past + 1].view(B, 1, 1, 1, 1)
        phi_future = phi_future.expand(B, Nz, Nx, self.n_past, 1)

        inp = torch.cat([hist, phi_hist, phi_future], dim=-1)  # (B, Nz, Nx, n_past, Nf+2)
        out = self.fno(inp)                                    # (B, Nz, Nx, n_past, Nf)
        out_next = out[:, :, :, -1, :]                          # ultima fetta -> next-step
        x_pred = out_next.permute(0, 3, 1, 2).reshape(B, self.nf, -1)

        x_true = x_window[:, self.n_past]
        return F.mse_loss(x_pred, x_true)

    @classmethod
    def from_config(cls, cfg, nf, n_cells, n_past, device):
        grid_shape = cfg["grid_shape"]
        if grid_shape[0] * grid_shape[1] != n_cells:
            raise ValueError(
                f"grid_shape {grid_shape} incompatibile con Ncells={n_cells}. "
                "FNO3D richiede una griglia strutturata: se la mesh è non uniforme, "
                "serve prima interpolare su una griglia regolare."
            )
        model = cls(
            nf=nf, grid_shape=grid_shape, n_past=n_past,
            modes1=cfg["modes1"], modes2=cfg["modes2"], modes3=cfg["modes3"], width=cfg["width"],
        )
        return model.to(device)

class UNetBlock(nn.Module):
    """Doppia convoluzione + BN + attivazione, blocco base di ogni livello U-Net."""

    def __init__(self, in_ch, out_ch, padding_mode="circular"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net vera: downsampling path con skip connections salvate,
    upsampling path che le concatena ai livelli corrispondenti.
    channels, es. [32, 64, 128] definisce la profondità.
    """

    def __init__(self, in_channels, out_channels, channels, padding_mode="circular"):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.downs.append(UNetBlock(prev_ch, ch, padding_mode))
            self.pools.append(nn.AvgPool2d(2))
            prev_ch = ch

        self.bottleneck = UNetBlock(prev_ch, prev_ch * 2, padding_mode)

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        prev_ch = prev_ch * 2
        for ch in rev_channels:
            self.ups.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=2, stride=2))
            self.up_blocks.append(UNetBlock(ch * 2, ch, padding_mode))  # *2 per la concat skip
            prev_ch = ch

        self.out_conv = nn.Conv2d(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, up_block, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)   # ← skip connection
            x = up_block(x)

        return self.out_conv(x)


class UNetModel(BaseForecastModel):
    """Richiede griglia strutturata (grid_shape), stesso vincolo di FNO3D."""

    def __init__(self, nf, grid_shape, n_past, channels, padding_mode="circular"):
        super().__init__()
        self.nf = nf
        self.grid_shape = tuple(grid_shape)
        self.n_past = n_past
        in_channels = nf * n_past + (n_past + 1)  # storia + phi_hist + phi_future broadcast
        self.unet = UNet(in_channels, nf, channels, padding_mode=padding_mode)

    def _reshape_to_grid(self, x_flat):
        B = x_flat.shape[0]
        Nz, Nx = self.grid_shape
        return x_flat.reshape(B, self.nf, Nz, Nx)

    def compute_loss(self, x_window, phi_window):
        B = x_window.shape[0]
        Nz, Nx = self.grid_shape

        hist = torch.cat([self._reshape_to_grid(x_window[:, i]) for i in range(self.n_past)], dim=1)
        phi_channels = phi_window.view(B, self.n_past + 1, 1, 1).expand(B, self.n_past + 1, Nz, Nx)
        inp = torch.cat([hist, phi_channels], dim=1)

        x_pred_grid = self.unet(inp)
        x_pred = x_pred_grid.reshape(B, self.nf, -1)

        x_true = x_window[:, self.n_past]
        return F.mse_loss(x_pred, x_true)

    @classmethod
    def from_config(cls, cfg, nf, n_cells, n_past, device):
        grid_shape = cfg["grid_shape"]
        if grid_shape[0] * grid_shape[1] != n_cells:
            raise ValueError(
                f"grid_shape {grid_shape} incompatibile con Ncells={n_cells}. "
                "UNet richiede una griglia strutturata."
            )
        model = cls(
            nf=nf, grid_shape=grid_shape, n_past=n_past,
            channels=cfg["channels"],
            padding_mode=cfg.get("padding_mode", "circular"),
        )
        return model.to(device)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures


class PODARXModel(BaseForecastModel):
    """
    Non a gradiente: POD + regressione lineare/polinomiale per modo, in
    forma chiusa. poly_degree=1 -> ARX, poly_degree>1 -> NARX (stessa classe).

    Regressore per il modo k: [A_hist(flatten, n_past*r), phi_future] -> A_target[k]
    """

    def __init__(self, rank_pod, n_past, poly_degree=1, use_ridge=False, alpha=0.0):
        super().__init__()
        self.rank_pod = rank_pod
        self.n_past = n_past
        self.poly_degree = poly_degree
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.pod = PODReducer(rank=rank_pod)
        self.arx_models = []          # lista di LinearRegression/Ridge, uno per modo
        self.poly_transformers = []   # lista di PolynomialFeatures o None

    def requires_training_loop(self):
        return False

    def preprocess(self, train_snapshots):
        self.pod.fit(train_snapshots)
        return {"pod_U_r": self.pod.U_r, "pod_mean": self.pod.mean, "pod_rank": self.pod.U_r.shape[1]}

    def fit_closed_form(self, train_dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(train_dataset, batch_size=4096, shuffle=False)

        hist_list, target_list, phi_future_list = [], [], []
        with torch.no_grad():
            for x_window, phi_window in loader:
                a_hist = torch.stack(
                    [self.pod.encode_torch(x_window[:, i]) for i in range(self.n_past)], dim=1
                )  # (B, n_past, r)
                a_target = self.pod.encode_torch(x_window[:, self.n_past])  # (B, r)
                hist_list.append(a_hist.cpu().numpy())
                target_list.append(a_target.cpu().numpy())
                phi_future_list.append(phi_window[:, self.n_past].cpu().numpy())

        A_hist = np.concatenate(hist_list, axis=0)        # (N, n_past, r)
        A_target = np.concatenate(target_list, axis=0)    # (N, r)
        phi_future = np.concatenate(phi_future_list, axis=0)  # (N,)

        N, n_past, r = A_hist.shape
        X = np.concatenate([A_hist.reshape(N, n_past * r), phi_future[:, None]], axis=1)

        self.arx_models = []
        self.poly_transformers = []
        for mode in range(r):
            y = A_target[:, mode]
            Xm = X
            poly = None
            if self.poly_degree > 1:
                poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
                Xm = poly.fit_transform(Xm)
            self.poly_transformers.append(poly)

            reg = Ridge(alpha=self.alpha) if self.use_ridge else LinearRegression()
            reg.fit(Xm, y)
            self.arx_models.append(reg)
            print(f"[POD-ARX] modo {mode + 1}/{r} fittato (poly_degree={self.poly_degree})")

        return {}

    def extra_checkpoint_data(self):
        return {
            "pod_U_r": self.pod.U_r,
            "pod_mean": self.pod.mean,
            "pod_rank": self.pod.U_r.shape[1],
            "arx_models": self.arx_models,
            "poly_transformers": self.poly_transformers,
            "poly_degree": self.poly_degree,
            "n_past_arx": self.n_past,
        }

    @classmethod
    def from_config(cls, cfg, nf, n_cells, n_past, device):
        model = cls(
            rank_pod=cfg["rank_POD"], n_past=n_past,
            poly_degree=cfg.get("poly_degree", 1),
            use_ridge=cfg.get("use_ridge", False),
            alpha=cfg.get("alpha", 0.0),
        )
        return model  # resta su CPU: sklearn non usa GPU



# =============================================================================
# Registry — per aggiungere un nuovo modello: scrivere la classe (eredita
# BaseForecastModel, implementa compute_loss + from_config) e registrarla qui.
# =============================================================================
MODEL_REGISTRY = {
    "pod-transformer": PODTransformerModel,
    "ae-transformer": AETransformerModel,
    "lstm": LSTMModel,
    "fno3d": FNO3DModel,
    "unet": UNetModel,
    "pod-arx": PODARXModel,
    "pod-narx": PODARXModel,
}

def build_model(name: str, cfg: dict, nf: int, n_cells: int, n_past: int, device):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modello '{name}' non registrato. Disponibili: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name].from_config(cfg, nf=nf, n_cells=n_cells, n_past=n_past, device=device)