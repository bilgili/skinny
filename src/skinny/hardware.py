"""GPU hardware detection and selection.

Abstracts Vulkan physical device enumeration behind vendor-aware logic so
the renderer and video encoder can adapt to Intel, NVIDIA, or AMD hardware
via a single ``--gpu`` CLI flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import vulkan as vk


class GpuVendor(Enum):
    INTEL = auto()
    NVIDIA = auto()
    AMD = auto()
    UNKNOWN = auto()


_VENDOR_IDS: dict[int, GpuVendor] = {
    0x8086: GpuVendor.INTEL,
    0x10DE: GpuVendor.NVIDIA,
    0x1002: GpuVendor.AMD,
}

_H264_ENCODERS: dict[GpuVendor, str] = {
    GpuVendor.NVIDIA: "h264_nvenc",
    GpuVendor.INTEL: "h264_qsv",
    GpuVendor.AMD: "h264_amf",
}


@dataclass
class GpuInfo:
    """Describes a Vulkan-capable GPU with vendor metadata."""

    vendor: GpuVendor
    name: str
    device_id: int
    vendor_id: int
    device_type: int
    vk_physical_device: object

    @property
    def is_discrete(self) -> bool:
        return self.device_type == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU

    @property
    def preferred_h264_encoder(self) -> str:
        return _H264_ENCODERS.get(self.vendor, "libx264")

    def __repr__(self) -> str:
        kind = "discrete" if self.is_discrete else "integrated"
        return f"GpuInfo({self.vendor.name}, {self.name!r}, {kind})"


def enumerate_gpus(vk_instance) -> list[GpuInfo]:
    """List all Vulkan-capable GPUs with vendor detection."""
    devices = vk.vkEnumeratePhysicalDevices(vk_instance)
    if not devices:
        return []
    out: list[GpuInfo] = []
    for dev in devices:
        props = vk.vkGetPhysicalDeviceProperties(dev)
        vendor_id = props.vendorID
        vendor = _VENDOR_IDS.get(vendor_id, GpuVendor.UNKNOWN)
        out.append(GpuInfo(
            vendor=vendor,
            name=props.deviceName,
            device_id=props.deviceID,
            vendor_id=vendor_id,
            device_type=props.deviceType,
            vk_physical_device=dev,
        ))
    return out


def select_gpu(
    vk_instance,
    preference: str | None = None,
) -> GpuInfo:
    """Select a GPU by preference string.

    Parameters
    ----------
    vk_instance : VkInstance
    preference : str or None
        ``"intel"``, ``"nvidia"``, ``"amd"`` — pick first matching vendor.
        ``"discrete"`` — pick first discrete GPU.
        ``None`` or ``"auto"`` — prefer discrete, fall back to first.

    Raises
    ------
    RuntimeError
        No GPU found, or no GPU matching the requested preference.
    """
    gpus = enumerate_gpus(vk_instance)
    if not gpus:
        raise RuntimeError("No Vulkan-capable GPU found")

    if preference is None or preference == "auto":
        for g in gpus:
            if g.is_discrete:
                return g
        return gpus[0]

    pref = preference.lower()

    if pref == "discrete":
        for g in gpus:
            if g.is_discrete:
                return g
        raise RuntimeError("No discrete GPU found")

    vendor_map = {
        "intel": GpuVendor.INTEL,
        "nvidia": GpuVendor.NVIDIA,
        "amd": GpuVendor.AMD,
    }
    target = vendor_map.get(pref)
    if target is None:
        raise ValueError(
            f"Unknown GPU preference {preference!r}. "
            f"Use: intel, nvidia, amd, discrete, or auto."
        )

    for g in gpus:
        if g.vendor == target:
            return g

    available = ", ".join(repr(g) for g in gpus)
    raise RuntimeError(
        f"No {preference} GPU found. Available: {available}"
    )
