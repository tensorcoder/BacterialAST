"""DINO self-supervised learning framework."""

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
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        nn.utils.parametrizations.weight_norm(self.last_layer)
        self.last_layer.parametrizations.weight.original0.data.fill_(1.0)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    and sharpened with a separate temperature before the softmax.
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
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor | list[torch.Tensor]) -> None:
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
        student_probs = [
            F.log_softmax(s / self.student_temp, dim=-1) for s in student_outputs
        ]
        teacher_probs = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1)
            for t in teacher_outputs
        ]

        total_loss = torch.tensor(0.0, device=student_outputs[0].device)
        n_loss_terms = 0

        for t_idx, tp in enumerate(teacher_probs):
            for s_idx, sp in enumerate(student_probs):
                if t_idx == s_idx:
                    continue
                loss = -torch.sum(tp * sp, dim=-1).mean()
                total_loss = total_loss + loss
                n_loss_terms += 1

        total_loss = total_loss / max(n_loss_terms, 1)
        return total_loss


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

        self.student_backbone = ViTSmall(**backbone_kwargs)
        self.student_head = DINOHead(**head_kwargs)

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    def forward_student(self, crops: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for crop in crops:
            feat = self.student_backbone(crop)
            proj = self.student_head(feat)
            outputs.append(proj)
        return outputs

    @torch.no_grad()
    def forward_teacher(self, crops: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for crop in crops:
            feat = self.teacher_backbone(crop)
            proj = self.teacher_head(feat)
            outputs.append(proj)
        return outputs

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996) -> None:
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
