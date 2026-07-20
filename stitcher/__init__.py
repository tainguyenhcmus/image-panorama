"""Gói ghép ảnh panorama bằng pipeline computer vision cổ điển."""

from .pipeline import StitchConfig, StitchError, stitch_panorama

__all__ = ["StitchConfig", "StitchError", "stitch_panorama"]
