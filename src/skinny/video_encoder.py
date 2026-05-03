"""H264 video encoder with hardware-accelerated fallback chain.

Detects GPU vendor via ``GpuInfo`` and tries the matching hardware encoder
first (QSV for Intel, NVENC for NVIDIA, AMF for AMD), falling back to
libx264 software, then JPEG-only mode.

Packets are output in AVCC format (4-byte length prefix) for WebCodecs
compatibility.  The ``avcc_description`` property returns the
AVCDecoderConfigurationRecord needed by the browser's VideoDecoder.
"""

from __future__ import annotations

import io
import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skinny.hardware import GpuInfo

log = logging.getLogger(__name__)


# ── Annex B / AVCC helpers ──────────────────────────────────────────

def _split_annex_b(data: bytes) -> list[bytes]:
    """Split Annex B bitstream into individual NAL units (without start codes)."""
    nals: list[bytes] = []
    i = 0
    n = len(data)
    while i < n:
        if i + 3 < n and data[i:i+4] == b'\x00\x00\x00\x01':
            start = i + 4
        elif i + 2 < n and data[i:i+3] == b'\x00\x00\x01':
            start = i + 3
        else:
            i += 1
            continue
        end = n
        j = start
        while j < n:
            if j + 3 < n and data[j:j+4] == b'\x00\x00\x00\x01':
                end = j
                break
            if j + 2 < n and data[j:j+3] == b'\x00\x00\x01':
                end = j
                break
            j += 1
        nal = data[start:end]
        if nal:
            nals.append(nal)
        i = end
    return nals


def _annex_b_to_avcc(data: bytes) -> bytes:
    """Convert Annex B bitstream to AVCC format (4-byte length prefixed NAL units)."""
    nals = _split_annex_b(data)
    parts: list[bytes] = []
    for nal in nals:
        parts.append(struct.pack(">I", len(nal)))
        parts.append(nal)
    return b"".join(parts)


def _build_avcc_description(sps: bytes, pps: bytes) -> bytes:
    """Build AVCDecoderConfigurationRecord from raw SPS and PPS NAL units.

    sps includes NAL header byte (0x67); profile/compat/level are at [1:4].
    """
    buf = bytearray()
    buf.append(1)            # configurationVersion
    buf.append(sps[1])       # AVCProfileIndication
    buf.append(sps[2])       # profile_compatibility
    buf.append(sps[3])       # AVCLevelIndication
    buf.append(0xFF)         # lengthSizeMinusOne = 3 (4 bytes) | reserved 6 bits
    buf.append(0xE1)         # numOfSequenceParameterSets = 1 | reserved 3 bits
    buf.extend(struct.pack(">H", len(sps)))
    buf.extend(sps)
    buf.append(1)            # numOfPictureParameterSets
    buf.extend(struct.pack(">H", len(pps)))
    buf.extend(pps)
    return bytes(buf)


class VideoEncoder:
    """Encode raw RGBA frames to H264 AVCC packets (or JPEG as fallback)."""

    def __init__(
        self,
        width: int,
        height: int,
        gpu_info: GpuInfo | None = None,
        fps: int = 20,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._force_next_keyframe = False
        self._codec_ctx = None
        self._codec_name: str | None = None
        self._frame_count = 0
        self._avcc_description: bytes | None = None

        self._codec_name = self._try_open_encoder(gpu_info)

    def _try_open_encoder(self, gpu_info: GpuInfo | None) -> str | None:
        """Try encoders in priority order, return name of first that opens successfully."""
        try:
            import av  # noqa: F401
        except ImportError:
            log.warning("PyAV not installed — falling back to JPEG encoding")
            return None

        if gpu_info is not None:
            candidates = [gpu_info.preferred_h264_encoder, "libx264"]
        else:
            candidates = ["h264_nvenc", "h264_qsv", "h264_amf", "libx264"]

        for name in candidates:
            try:
                self._init_codec(name)
                log.info("H264 encoder opened: %s", name)
                return name
            except Exception as exc:
                log.debug("Encoder %s failed: %s", name, exc)
                self._codec_ctx = None
                continue

        log.warning("No H264 encoder found — falling back to JPEG")
        return None

    def _init_codec(self, codec_name: str) -> None:
        import av
        from fractions import Fraction

        self._codec_ctx = av.CodecContext.create(codec_name, "w")
        self._codec_ctx.width = self.width
        self._codec_ctx.height = self.height
        self._codec_ctx.framerate = self.fps
        self._codec_ctx.time_base = Fraction(1, self.fps)
        self._codec_ctx.pix_fmt = "yuv420p"
        self._codec_ctx.gop_size = 30
        self._codec_ctx.flags |= av.codec.context.Flags.global_header

        if codec_name == "h264_nvenc":
            self._codec_ctx.options = {
                "preset": "p1",
                "tune": "ull",
                "rc": "cbr",
                "profile": "baseline",
            }
        elif codec_name == "h264_qsv":
            self._codec_ctx.options = {
                "preset": "veryfast",
                "profile": "baseline",
            }
        elif codec_name == "h264_amf":
            self._codec_ctx.options = {
                "usage": "ultralowlatency",
                "profile": "baseline",
            }
        elif codec_name == "libx264":
            self._codec_ctx.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "profile": "baseline",
            }

        self._codec_ctx.open()
        self._extract_description()

    def _extract_description(self) -> None:
        """Extract SPS/PPS from codec extradata and build AVCDecoderConfigurationRecord."""
        extradata = self._codec_ctx.extradata
        if not extradata:
            log.warning("No extradata from codec — H264 WebCodecs may not work")
            return
        nals = _split_annex_b(bytes(extradata))
        sps = None
        pps = None
        for nal in nals:
            nal_type = nal[0] & 0x1F
            if nal_type == 7:
                sps = nal
            elif nal_type == 8:
                pps = nal
        if sps and pps:
            self._avcc_description = _build_avcc_description(sps, pps)
            log.info("AVCC description built: %d bytes (SPS=%d, PPS=%d)",
                     len(self._avcc_description), len(sps), len(pps))
        else:
            log.warning("Could not extract SPS/PPS from extradata")

    @property
    def avcc_description(self) -> bytes | None:
        """AVCDecoderConfigurationRecord for WebCodecs VideoDecoder.configure()."""
        return self._avcc_description

    @property
    def is_hardware(self) -> bool:
        return self._codec_name is not None and self._codec_name != "libx264"

    @property
    def is_h264(self) -> bool:
        return self._codec_ctx is not None

    @property
    def encoder_name(self) -> str:
        return self._codec_name or "jpeg"

    def force_keyframe(self) -> None:
        """Force next encoded frame to be an I-frame."""
        self._force_next_keyframe = True

    def encode_h264(self, raw_rgba: bytes) -> list[tuple[bool, bytes]]:
        """Encode one RGBA frame. Returns list of (is_keyframe, avcc_data) tuples."""
        if self._codec_ctx is None:
            return []

        import av

        frame = av.VideoFrame(self.width, self.height, "rgba")
        frame.planes[0].update(raw_rgba)
        frame = frame.reformat(format="yuv420p")
        frame.pts = self._frame_count
        from fractions import Fraction
        frame.time_base = Fraction(1, self.fps)

        if self._force_next_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
            self._force_next_keyframe = False

        packets = self._codec_ctx.encode(frame)
        self._frame_count += 1
        result = []
        for pkt in packets:
            is_key = pkt.is_keyframe
            avcc = _annex_b_to_avcc(bytes(pkt))
            result.append((is_key, avcc))
        return result

    def encode_jpeg(self, raw_rgba: bytes, quality: int = 80) -> bytes:
        """Encode one RGBA frame as JPEG. Always available (uses Pillow)."""
        from PIL import Image

        img = Image.frombuffer("RGBA", (self.width, self.height), raw_rgba, "raw", "RGBA", 0, 1)
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    def flush(self) -> list[tuple[bool, bytes]]:
        """Flush remaining H264 packets from encoder."""
        if self._codec_ctx is None:
            return []
        packets = self._codec_ctx.encode(None)
        return [(pkt.is_keyframe, _annex_b_to_avcc(bytes(pkt))) for pkt in packets]

    def close(self) -> None:
        if self._codec_ctx is not None:
            try:
                self._codec_ctx.close()
            except AttributeError:
                pass
            self._codec_ctx = None
