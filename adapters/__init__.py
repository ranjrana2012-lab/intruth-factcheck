"""Capture adapters — Zone 1 (native perception).

Each adapter feeds Int16 16kHz mono PCM (or OCR text) into the cognitive core via the same
protocol. Add screen OCR, Screenpipe integration, etc. here.
"""
from .wasapi_capture import WasapiCapture, make_default_capture

__all__ = ["WasapiCapture", "make_default_capture"]
