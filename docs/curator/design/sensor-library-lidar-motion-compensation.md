# LiDAR Motion Compensation

This derives the transform used to motion-compensate LiDAR points: given
points captured at time `t`, express them in the ego frame at a different
reference time `t0`. It covers the math only — it does not define the
`LidarData`/`EgoTrajectory` fields that would supply the inputs (see
[Sensor Library LiDAR Data Design](sensor-library-lidar-data.md)), the
interpolation policy for ego pose between samples, or an implementation.

## Notation

`(dest_frame)_T_(source_frame)` names a transform from the source frame to
the destination frame.

| Symbol | Meaning |
| --- | --- |
| `t` | An arbitrary capture time. |
| `t0` | The time to motion-compensate to — the timestamp all points are aligned to. |
| `w_T_vt` | Vehicle-to-world transform at time `t` (ego pose at `t`). |
| `w_T_v0` | Vehicle-to-world transform at time `t0` (ego pose at `t0`). |
| `v_T_l` | LiDAR-to-vehicle transform (extrinsics / calibration). |
| `P_l_t` | LiDAR points in the LiDAR frame at time `t`. |
| `P_w_t` | LiDAR points in the world frame at time `t`. |

## Problem

Projecting points into the world frame for visualization is direct:

```
P_w_t = w_T_vt * v_T_l * P_l_t
```

But for labelling we want points relative to ego at `t0`, not `t`:

```
P_v0_t = v0_T_l(t) * P_l_t
```

## Deriving the transform

What we know:

- `w_T_v0` — ego pose at `t0` (invertible to `v0_T_w`).
- `w_T_vt` — ego pose at `t`. If `t` doesn't land exactly on an
  `EgoTrajectory` sample, this needs an interpolation policy; picking one
  (and any staleness bound, e.g. `NearestTimestampPolicy(max_delta_ns=...)`)
  is out of scope for this note.
- `v_T_l` — LiDAR extrinsics.

What we want: take a point straight from the LiDAR frame at time `t` to
the ego frame at `t0`, by first moving it into the vehicle frame at `t`
(`v_T_l`), then from the vehicle frame at `t` to the vehicle frame at `t0`
(`v0_T_vt`):

```
P_v0_t = v0_T_vt * v_T_l * P_l_t
```

`v0_T_vt` is the transform from ego pose at `t` to ego pose at `t0`:

```
v0_T_vt = v0_T_w * w_T_vt
```

We don't have `v0_T_w` directly, but we have its inverse, `w_T_v0`:

```
v0_T_w = (w_T_v0)^-1
```

## Result

```
v0_T_l(t) = (w_T_v0)^-1 * w_T_vt * v_T_l
P_v0_t = v0_T_l(t) * P_l_t
```

`v0_T_l(t)` is written as a function of `t` because `w_T_vt` varies per
capture time, while `w_T_v0` and `v_T_l` are fixed for a given batch.
