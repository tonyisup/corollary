# M7 Phase 4: Mobile Device-Profile Layer

> **Date:** 2026-05-14
> **Status:** Complete

## Concept

On a mobile device, the acoustic path between the speaker and microphone changes with device orientation (portrait, landscape, pocket, face-down, on-table). Each orientation has a different "device profile" — a different IR or device transfer function.

The device-profile layer maintains per-orientation IR profiles and switches between them based on orientation sensor input.

## Implementation

New module: `packages/cerebellum/device_profile.py`

### DeviceOrientation

Enum of canonical orientations: UNKNOWN, PORTRAIT, LANDSCAPE, POCKET, FACE_DOWN, ON_TABLE.

### OrientationSensor

Abstract interface for orientation sensing:
- `current_orientation()` — returns current orientation
- `orientation_stable_frames` — consecutive frames with same orientation
- Mock implementation for testing/desktop use

### DeviceProfile

Per-orientation acoustic profile:
- Own IR with orientation-specific prior (seeded via `guess_ir()`)
- Independent NLMS adaptation
- Error tracking and serialization

### DeviceProfileBank

Manages multiple profiles:
- `update_orientation()` — switches active profile based on sensor
- `adapt()` — adapts the active profile
- Auto-creates profiles for new orientations
- Prunes least-used profiles when at max
- Falls back to UNKNOWN profile when no sensor data

### Integration into BabblingCerebellum

- `set_device_profile_bank(bank)` attaches a `DeviceProfileBank`
- Works alongside the IR bank (phase 3) — each profile can have its own bank
- `predict_pcm()` and `predict_mel()` use the active profile's IR via `_active_ir`

## Tests

17 new unit tests in `tests/unit/test_cerebellum_device_profile.py`:
- DeviceOrientation enum and string parsing
- DeviceProfile initialization, guess_ir, serialization
- OrientationSensor mock behavior
- DeviceProfileBank switching, adaptation, pruning, serialization
- Cerebellum integration: bank attachment and orientation switching
