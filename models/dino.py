"""DINO self-supervised learning framework with temporal contrastive loss."""

from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ViTSmall


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class DINOHead(nn.Module):
    """MLP projection head with bottleneck used in DINO.

    Architecture: ``in_dim → 2048 → 2048 → 256 → out_dim (65536)``

    The last linear layer uses weight normalisation (no bias) and the output is
    L2-normalised before being returned.
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        # Last layer with weight normalisation and no bias.
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # Initialise last layer weight norm magnitude to 1.
        self.last_layer.weight_g.data.fill_(1.0)
        # Fix last layer weight norm so it is not trained by default
        # (following the official DINO implementation).
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) feature vector.

        Returns:
            (B, out_dim) L2-normalised projection.
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# ---------------------------------------------------------------------------
# DINO loss
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """Cross-entropy loss between teacher and student softmax distributions.

    The teacher logits are centred (subtract running mean of teacher outputs)
    and sharpened with a separate temperature before the softmax.  The student
    uses its own temperature.
    """

    def __init__(
        self,
        out_dim: int = 65536,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        # Running centre – not a learnable parameter.
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor | list[torch.Tensor]) -> None:
        """EMA update of the centre vector.

        Args:
            teacher_output: (B, out_dim) or list of (B, out_dim) raw teacher logits.
        """
        if isinstance(teacher_output, list):
            teacher_output = torch.cat(teacher_output, dim=0)
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1.0 - self.center_momentum
        )

    def forward(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the DINO loss.

        The loss is the mean cross-entropy over all (teacher, student) pairs
        where teacher and student views are *different*.

        Args:
            student_outputs: list of (B, out_dim) tensors, one per student crop.
            teacher_outputs: list of (B, out_dim) tensors, one per teacher
                (global) crop.

        Returns:
            Scalar loss.
        """
        # Student softmax (no centering).
        student_probs = [
            F.log_softmax(s / self.student_temp, dim=-1) for s in student_outputs
        ]
        # Teacher softmax with centering and sharpening.
        teacher_probs = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1)
            for t in teacher_outputs
        ]

        total_loss = torch.tensor(0.0, device=student_outputs[0].device)
        n_loss_terms = 0

        for t_idx, tp in enumerate(teacher_probs):
            for s_idx, sp in enumerate(student_probs):
                # Skip when the teacher and student view are the same.
                if t_idx == s_idx:
                    continue
                # Cross-entropy: -sum(p * log(q))
                loss = -torch.sum(tp * sp, dim=-1).mean()
                total_loss = total_loss + loss
                n_loss_terms += 1

        total_loss = total_loss / max(n_loss_terms, 1)
        return total_loss


# ---------------------------------------------------------------------------
# Temporal contrastive loss (NT-Xent)
# ---------------------------------------------------------------------------

class TemporalContrastiveLoss(nn.Module):
    """NT-Xent (normalised temperature-scaled cross-entropy) loss for
    temporal pairs: (z_t, z_{t+tau}) of the same bacterium at different times.
    """

    def __init__(self, temperature: float = 0.3) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_t: torch.Tensor,
        z_t_tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t:     (B, D) L2-normalised embeddings at time *t*.
            z_t_tau: (B, D) L2-normalised embeddings at time *t + tau*.

        Returns:
            Scalar NT-Xent loss.
        """
        z_t = F.normalize(z_t, dim=-1, p=2)
        z_t_tau = F.normalize(z_t_tau, dim=-1, p=2)

        B = z_t.shape[0]
        # Concatenate both views: [z_t; z_t_tau] → (2B, D)
        z = torch.cat([z_t, z_t_tau], dim=0)

        # Full similarity matrix (2B, 2B).
        sim = z @ z.T / self.temperature  # (2B, 2B)

        # Mask out self-similarity on the diagonal.
        mask_self = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i).
        labels = torch.cat(
            [torch.arange(B, 2 * B, device=z.device),
             torch.arange(0, B, device=z.device)],
            dim=0,
        )
        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# DINO wrapper (student + teacher)
# ---------------------------------------------------------------------------

class DINOWrapper(nn.Module):
    """Wraps student and teacher networks (backbone + projection head).

    The teacher is an exponential-moving-average copy of the student and is
    never trained directly through gradients.
    """

    def __init__(
        self,
        backbone_kwargs: Optional[dict] = None,
        head_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        head_kwargs = head_kwargs or {}

        # Student
        self.student_backbone = ViTSmall(**backbone_kwargs)
        self.student_head = DINOHead(**head_kwargs)

        # Teacher – deep-copy student, then freeze.
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)
        self._freeze_teacher()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _freeze_teacher(self) -> None:
        """Disable gradient computation for teacher parameters."""
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------
    def forward_student(self, crops: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run the student on every crop.

        Args:
            crops: list of (B, 1, H, W) image tensors (global + local crops).

        Returns:
            List of (B, out_dim) projection tensors, one per crop.
        """
        outputs: List[torch.Tensor] = []
        for crop in crops:
            feat = self.student_backbone(crop)   # (B, 384)
            proj = self.student_head(feat)       # (B, out_dim)
            outputs.append(proj)
        return outputs

    @torch.no_grad()
    def forward_teacher(self, crops: List[torch.Tensor]) -> List[torch.Tensor]:
        """Run the teacher on every crop (no gradient).

        Args:
            crops: list of (B, 1, H, W) image tensors (typically only global
                crops are passed to the teacher).

        Returns:
            List of (B, out_dim) projection tensors, one per crop.
        """
        outputs: List[torch.Tensor] = []
        for crop in crops:
            feat = self.teacher_backbone(crop)
            proj = self.teacher_head(feat)
            outputs.append(proj)
        return outputs

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
        """EMA update: teacher ← momentum * teacher + (1 - momentum) * student.

        Args:
            momentum: EMA decay rate (typically 0.996 → 1.0 via cosine schedule).
        """
        for t_param, s_param in zip(
            self.teacher_backbone.parameters(),
            self.student_backbone.parameters(),
        ):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

        for t_param, s_param in zip(
            self.teacher_head.parameters(),
            self.student_head.parameters(),
        ):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
