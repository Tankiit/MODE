import torch.nn.functional as F
from torch import Tensor


def sieve_masked_loss(model, inputs: Tensor, targets: Tensor,
                      mask: Tensor, sliding_window_num_blocks: Tensor,
                      scale: bool = True) -> Tensor:
    captured = {}

    def hook_fn(module, input, output):
        captured["logits"] = output

    handle = model.lm_head.register_forward_hook(hook_fn)

    try:
        full_loss = model(inputs, targets, sliding_window_num_blocks)
    finally:
        handle.remove()

    logits = captured.get("logits")
    if logits is None:
        return full_loss

    logits = model._logits_softcap_scale * torch.sigmoid(
        (logits + model._logits_softcap_shift) / model._logits_softcap_divisor
    )

    logits_flat = logits.float().view(-1, logits.size(-1)) if not model.training else logits.view(-1, logits.size(-1))

    per_token = F.cross_entropy(logits_flat, targets, reduction="none")

    masked = per_token * mask.float()
    loss = masked.sum()

    if scale:
        num_selected = mask.sum().clamp(min=1).float()
        total = float(targets.size(0))
        loss = loss * (total / num_selected)

    return loss
