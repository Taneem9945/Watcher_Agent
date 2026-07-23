from __future__ import annotations


def rearrange(tensor, pattern, **axes_lengths):
    pattern = pattern.strip()
    if pattern == "n -> d n":
        d = int(axes_lengths["d"])
        return tensor.unsqueeze(0).expand(d, -1)
    if pattern == "b l d -> d (b l)":
        return tensor.permute(2, 0, 1).reshape(tensor.shape[2], -1)
    if pattern == "d -> d 1":
        return tensor.unsqueeze(-1)
    if pattern == "b d l -> (b l) d":
        return tensor.permute(0, 2, 1).reshape(-1, tensor.shape[1])
    if pattern == "d (b l) -> b d l":
        l = int(axes_lengths["l"])
        b = tensor.shape[1] // l
        return tensor.reshape(tensor.shape[0], b, l).permute(1, 0, 2)
    if pattern == "(b l) dstate -> b dstate l":
        l = int(axes_lengths["l"])
        b = tensor.shape[0] // l
        return tensor.reshape(b, l, tensor.shape[1]).permute(0, 2, 1)
    if pattern == "(b l) dstate -> b 1 dstate l":
        l = int(axes_lengths["l"])
        b = tensor.shape[0] // l
        return tensor.reshape(b, l, tensor.shape[1]).permute(0, 2, 1).unsqueeze(1)
    if pattern == "b d l -> b l d":
        return tensor.permute(0, 2, 1)
    if pattern == "d 1 w -> d w":
        return tensor.squeeze(1)
    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


def repeat(tensor, pattern, **axes_lengths):
    pattern = pattern.strip()
    if pattern == "n -> d n":
        d = int(axes_lengths["d"])
        return tensor.unsqueeze(0).expand(d, -1)
    raise NotImplementedError(f"Unsupported repeat pattern: {pattern}")
