"""M2 policy-value network for CardEnv's variable candidate sets."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import hashlib
import json

import torch
from torch import Tensor, nn

from dgisim_ext.env_api import Choice, Obs

__all__ = ["PolicyValueNet"]


_KIND_ORDER = ("tuple", "card", "dice", "cost", "commit",
               "card_count", "dice_count", "cost_count")
_KIND_TO_ID = {name: i for i, name in enumerate(_KIND_ORDER)}
_ACTION_TYPES = ("tuple", "skill", "card", "swap", "end_round", "reroll",
                 "tuning", "dice", "cost", "commit")
_ACTION_TYPE_STRIDE = len(_KIND_ORDER)
_ACTION_TYPE_TO_ID = {name: i for i, name in enumerate(_ACTION_TYPES)}


class PolicyValueNet(nn.Module):
    """Shared MLP encoder with content-based candidate scoring.

    Candidate logits are computed independently from candidate order.  The
    caller may therefore sort or permute candidates without changing the
    score assigned to a particular candidate.
    """

    def __init__(self, obs_dim: int = 2778, feature_dim: int = 4,
                 hidden_dim: int = 256, kind_count: int = len(_KIND_ORDER) * len(_ACTION_TYPES)):
        super().__init__()
        if obs_dim <= 0 or feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("network dimensions must be positive")
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.kind_count = kind_count
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.kind_embedding = nn.Embedding(kind_count, 16)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim + 16, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _state(self, enc: Tensor, seat: Tensor, stage: Tensor) -> Tensor:
        if enc.ndim == 1:
            enc = enc.unsqueeze(0)
        if enc.ndim != 2 or enc.shape[-1] != self.obs_dim:
            raise ValueError(f"enc must have shape [batch, {self.obs_dim}]")
        seat = seat.reshape(-1, 1).to(dtype=enc.dtype, device=enc.device)
        stage = stage.reshape(-1, 1).to(dtype=enc.dtype, device=enc.device)
        if seat.shape[0] != enc.shape[0] or stage.shape[0] != enc.shape[0]:
            raise ValueError("seat and stage batch sizes must match enc")
        # The engine encoding is categorical. Scaling keeps the small dynamic
        # metadata from being overwhelmed by code values in the first layer.
        return self.encoder(torch.cat((enc / 256.0, seat, stage / 100.0), dim=-1))

    def forward(self, enc: Tensor, seat: Tensor, stage: Tensor,
                candidate_features: Tensor, candidate_kinds: Tensor
                ) -> tuple[Tensor, Tensor]:
        """Return ``(candidate_logits, value)`` for one or more states.

        The common rollout path passes a single state and ``N`` candidates,
        producing logits ``[N]`` and value ``[1]``.
        """
        state = self._state(enc, seat, stage)
        if candidate_features.ndim == 2:
            candidate_features = candidate_features.unsqueeze(0)
        if candidate_kinds.ndim == 1:
            candidate_kinds = candidate_kinds.unsqueeze(0)
        if candidate_features.ndim != 3 or candidate_features.shape[-1] != self.feature_dim:
            raise ValueError("candidate_features must have shape [batch, choices, feature_dim]")
        if candidate_kinds.shape[:2] != candidate_features.shape[:2]:
            raise ValueError("candidate kind and feature shapes must match")
        if state.shape[0] != candidate_features.shape[0]:
            raise ValueError("state and candidate batch sizes must match")
        # Choice payload codes are engine dictionary IDs (typically 1..3000),
        # while counts/semantic tags are small.  Keep the code channel on the
        # same scale as the state encoder so raw IDs cannot dominate logits.
        candidate_features = candidate_features.clone()
        if candidate_features.shape[-1] >= 1:
            candidate_features[..., 0] = candidate_features[..., 0] / 4096.0
        if candidate_features.shape[-1] >= 2:
            candidate_features[..., 1] = candidate_features[..., 1] / 64.0
        kind = self.kind_embedding(candidate_kinds.long())
        expanded = state.unsqueeze(1).expand(-1, candidate_features.shape[1], -1)
        logits = self.policy_head(torch.cat((expanded, candidate_features, kind), dim=-1)).squeeze(-1)
        value = self.value_head(state).squeeze(-1)
        return (logits.squeeze(0), value.squeeze(0)) if logits.shape[0] == 1 else (logits, value)

    @staticmethod
    def _kind_id(kind: str) -> int:
        try:
            return _KIND_TO_ID[kind]
        except KeyError as exc:
            raise ValueError(f"unknown choice kind: {kind}") from exc

    @staticmethod
    def _prefix_payload(obs: Obs) -> tuple:
        """Canonical, structured prefix data used by every policy call.

        Keeping the action labels and payment counts as fields (rather than a
        character sum) makes payment allocations distinguishable and gives
        rollout/PPO/bootstrap one stable encoding contract.
        """
        return (tuple(str(x) for x in obs.meta.get("choice_prefix", ())),
                tuple(int(x) for x in obs.meta.get("selection_prefix", ())))

    @classmethod
    def observation_stage(cls, obs: Obs) -> float:
        prefix, selection = cls._prefix_payload(obs)
        payload = json.dumps({"actions": prefix, "payment": selection},
                             ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        # Two independent digest bytes are retained in the legacy stage scalar.
        # This preserves checkpoint dimensions while avoiding order/character
        # sums and keeping the value numerically close to stage_index.
        code = int.from_bytes(digest, "big")
        return (float(obs.stage_index) + len(prefix) / 100.0
                + (code % 65521) / 6552100.0
                + ((code // 65521) % 65521) / 655210000.0)

    def tensors_for(self, obs: Obs, device: torch.device | str | None = None
                    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Convert one public ``Obs`` to tensors consumed by :meth:`forward`."""
        if not obs.choices:
            raise ValueError("observation has no choices")
        device = torch.device(device) if device is not None else None
        enc = torch.tensor(obs.enc, dtype=torch.float32, device=device)
        # Prefix context is part of the public observation but is not included
        # in the fixed engine encoding. Fold stable prefix summaries into the
        # final two dynamic channels so payment/target choices are path-aware
        # without changing checkpoint tensor dimensions.
        seat = torch.tensor(float(obs.seat.value), dtype=torch.float32, device=device)
        stage = torch.tensor(self.observation_stage(obs),
                             dtype=torch.float32, device=device)
        features = torch.zeros((len(obs.choices), self.feature_dim), dtype=torch.float32, device=device)
        kinds = torch.empty(len(obs.choices), dtype=torch.long, device=device)
        for i, choice in enumerate(obs.choices):
            if len(choice.feature) > self.feature_dim:
                raise ValueError("choice feature exceeds configured feature_dim")
            features[i, :len(choice.feature)] = torch.tensor(choice.feature, dtype=torch.float32, device=device)
            # Composite embedding: semantic choice kind and coarse action family
            # are learned separately through disjoint IDs.
            kinds[i] = self._kind_id(choice.kind) + _ACTION_TYPE_STRIDE * _ACTION_TYPE_TO_ID.get(
                choice.action_type, 0)
        return enc, seat, stage, features, kinds

    def distribution(self, obs: Obs, device: torch.device | str | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return logits, categorical probabilities and value for one Obs."""
        tensors = self.tensors_for(obs, device=device)
        logits, value = self(*tensors)
        return logits, torch.softmax(logits, dim=-1), value

    def value_from_obs(self, obs: Obs, device: torch.device | str | None = None) -> Tensor:
        """Evaluate the state value without requiring action candidates."""
        device = torch.device(device) if device is not None else None
        enc = torch.tensor(obs.enc, dtype=torch.float32, device=device)
        seat = torch.tensor(float(obs.seat.value), dtype=torch.float32, device=device)
        stage = torch.tensor(self.observation_stage(obs), dtype=torch.float32, device=device)
        return self.value_head(self._state(enc, seat, stage)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: Obs, deterministic: bool = False,
            generator: torch.Generator | None = None
            , device: torch.device | str | None = None) -> tuple[int, Tensor, Tensor, Tensor]:
        """Choose a legal candidate and return index, log-prob, value, entropy."""
        logits, probs, value = self.distribution(obs, device=device)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            index = probs.argmax()
        else:
            # Categorical.sample() has no generator argument on all supported
            # torch versions; multinomial does and keeps policy RNG explicit.
            index = torch.multinomial(probs, 1, generator=generator).squeeze(0)
        log_prob = dist.log_prob(index)
        entropy = dist.entropy()
        return int(index.item()), log_prob, value, entropy
