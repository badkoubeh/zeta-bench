"""Post-hoc visualization of rocket-landing trajectories.

Two entry points, both consuming a :class:`Trajectory` populated by an
evaluation script. No env or dynamics imports — this module sits at the
bottom of the dependency chain so any caller can use it.

- :func:`plot_timeseries` writes a multi-panel PNG: altitude, velocity,
  attitude, action commands, fuel + reward. The view a controls engineer
  reaches for first when diagnosing why an episode failed.
- :func:`animate_side_view` writes an MP4: 2D X-Z side view with the
  rocket body rotated by pitch, a thrust vector arrow scaled by throttle,
  a faded trajectory trail, and a corner HUD.

Frame coordinates throughout are *plot* coordinates: horizontal = NED X,
vertical = altitude = -NED Z. Pitch rotates the rocket body about the
body Y-axis; in plot coordinates this is a CW rotation by ``pitch`` (i.e.
multiplication by ``R(-pitch)``) so positive pitch tilts the nose toward
+X.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import FancyArrow, Polygon
from numpy.typing import NDArray

# Use the ffmpeg binary bundled with imageio-ffmpeg so we don't depend on
# a system-wide install.
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# Visualisation scale for the rocket body. The dynamics treat the rocket
# as a point mass, so the body footprint is purely cosmetic.
_ROCKET_BODY_HEIGHT_M: float = 20.0
_ROCKET_BODY_WIDTH_M: float = 3.7
_ROCKET_NOSE_HEIGHT_M: float = 5.0
# Max thrust arrow length in metres, at full throttle.
_THRUST_ARROW_MAX_M: float = 40.0


@dataclass
class Trajectory:
    """One episode's per-step trajectory in physical units.

    All time-series arrays share the same leading dimension ``T``.

    Parameters
    ----------
    t : (T,) array
        Time stamps in seconds (start at 0, step ``1/control_hz``).
    pos_NED : (T, 3) array
        Position in NED frame (m).
    vel_NED : (T, 3) array
        Velocity in NED frame (m/s).
    euler : (T, 3) array
        Euler angles ``[roll, pitch, yaw]`` (rad).
    omega_body : (T, 3) array
        Angular rate in body frame (rad/s).
    action : (T, 3) array
        Commanded action ``[throttle, gimbal_pitch, gimbal_yaw]``.
    reward : (T,) array
        Per-step total reward.
    fuel_kg : (T,) array
        Fuel mass remaining (kg).
    meta : dict
        Scene context and metadata. Expected keys: ``outcome``,
        ``episode_idx``, ``seed``, ``return_total``,
        ``pad_radius_m``, ``oob_cylinder_radius_m``, ``oob_ceiling_m``,
        ``target_descent_mps`` (for the velocity-target line).
    """

    t: NDArray[np.float64]
    pos_NED: NDArray[np.float64]
    vel_NED: NDArray[np.float64]
    euler: NDArray[np.float64]
    omega_body: NDArray[np.float64]
    action: NDArray[np.float64]
    reward: NDArray[np.float64]
    fuel_kg: NDArray[np.float64]
    meta: dict[str, Any] = field(default_factory=dict)


def plot_timeseries(traj: Trajectory, out_path: Path) -> None:
    """Write a 4×2 multi-panel time-series figure to ``out_path`` as PNG.

    Panels: altitude, descent velocity (with target), lateral position,
    lateral velocity, Euler angles, angular rates, action commands, and
    fuel + reward.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t = traj.t
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    title = (
        f"Episode {traj.meta.get('episode_idx', '?')}  "
        f"outcome={traj.meta.get('outcome', '?')}  "
        f"return={traj.meta.get('return_total', float('nan')):.2f}"
    )
    fig.suptitle(title, fontsize=12)

    altitude = -traj.pos_NED[:, 2]
    axes[0, 0].plot(t, altitude, color="C0")
    axes[0, 0].axhline(0.0, color="k", lw=0.8, ls=":")
    axes[0, 0].set_ylabel("altitude (m)")
    axes[0, 0].set_title("Altitude")

    vz = traj.vel_NED[:, 2]
    axes[0, 1].plot(t, vz, color="C1", label="vz (NED +down)")
    target = traj.meta.get("target_descent_mps")
    if target is not None:
        axes[0, 1].axhline(float(target), color="k", lw=0.8, ls="--", label="target")
        axes[0, 1].legend(loc="best", fontsize=8)
    axes[0, 1].set_ylabel("vz (m/s)")
    axes[0, 1].set_title("Descent velocity")

    axes[1, 0].plot(t, traj.pos_NED[:, 0], label="x", color="C2")
    axes[1, 0].plot(t, traj.pos_NED[:, 1], label="y", color="C3")
    axes[1, 0].set_ylabel("lateral pos (m)")
    axes[1, 0].set_title("Lateral position (NED)")
    axes[1, 0].legend(loc="best", fontsize=8)

    axes[1, 1].plot(t, traj.vel_NED[:, 0], label="vx", color="C2")
    axes[1, 1].plot(t, traj.vel_NED[:, 1], label="vy", color="C3")
    axes[1, 1].set_ylabel("lateral vel (m/s)")
    axes[1, 1].set_title("Lateral velocity (NED)")
    axes[1, 1].legend(loc="best", fontsize=8)

    axes[2, 0].plot(t, np.rad2deg(traj.euler[:, 0]), label="roll", color="C4")
    axes[2, 0].plot(t, np.rad2deg(traj.euler[:, 1]), label="pitch", color="C5")
    axes[2, 0].plot(t, np.rad2deg(traj.euler[:, 2]), label="yaw", color="C6")
    axes[2, 0].set_ylabel("Euler (deg)")
    axes[2, 0].set_title("Attitude")
    axes[2, 0].legend(loc="best", fontsize=8)

    axes[2, 1].plot(t, np.rad2deg(traj.omega_body[:, 0]), label="ωx", color="C4")
    axes[2, 1].plot(t, np.rad2deg(traj.omega_body[:, 1]), label="ωy", color="C5")
    axes[2, 1].plot(t, np.rad2deg(traj.omega_body[:, 2]), label="ωz", color="C6")
    axes[2, 1].set_ylabel("body rates (deg/s)")
    axes[2, 1].set_title("Angular rates")
    axes[2, 1].legend(loc="best", fontsize=8)

    ax_action = axes[3, 0]
    ax_action.plot(t, traj.action[:, 0], label="throttle", color="C0")
    ax_action.set_ylabel("throttle [0, 1]")
    ax_action.set_ylim(-0.05, 1.05)
    ax_gimbal = ax_action.twinx()
    ax_gimbal.plot(t, traj.action[:, 1], label="gimbal pitch", color="C1", ls="--")
    ax_gimbal.plot(t, traj.action[:, 2], label="gimbal yaw", color="C2", ls="--")
    ax_gimbal.set_ylabel("gimbal cmd [-1, 1]")
    ax_gimbal.set_ylim(-1.05, 1.05)
    ax_action.set_xlabel("t (s)")
    ax_action.set_title("Action commands")
    lines1, labels1 = ax_action.get_legend_handles_labels()
    lines2, labels2 = ax_gimbal.get_legend_handles_labels()
    ax_action.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    ax_fuel = axes[3, 1]
    ax_fuel.plot(t, traj.fuel_kg, color="C7", label="fuel")
    ax_fuel.set_ylabel("fuel (kg)")
    ax_fuel.set_xlabel("t (s)")
    ax_reward = ax_fuel.twinx()
    ax_reward.plot(t, traj.reward, color="C8", ls="--", label="reward/step")
    ax_reward.set_ylabel("reward")
    ax_fuel.set_title("Fuel & reward")
    lines1, labels1 = ax_fuel.get_legend_handles_labels()
    lines2, labels2 = ax_reward.get_legend_handles_labels()
    ax_fuel.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _rocket_polygon(x: float, altitude: float, pitch: float) -> NDArray[np.float64]:
    """Vertices of a 2D rocket silhouette centred at ``(x, altitude)``, rotated
    by ``pitch`` rad. Local frame: nose at +up, base at -up.
    """
    w = _ROCKET_BODY_WIDTH_M
    h = _ROCKET_BODY_HEIGHT_M
    nh = _ROCKET_NOSE_HEIGHT_M
    # Local vertices (origin = body geometric centre)
    local = np.array(
        [
            [-w / 2, -h / 2],  # base left
            [+w / 2, -h / 2],  # base right
            [+w / 2, +h / 2],  # shoulder right
            [0.0, +h / 2 + nh],  # nose tip
            [-w / 2, +h / 2],  # shoulder left
        ],
        dtype=np.float64,
    )
    # The body frame is FRD (X = nose/forward) on an NED inertial frame, so the
    # "nose straight up" attitude is a +90° pitch: quat_to_euler returns
    # pitch = +π/2 when the rocket is vertical. The local polygon is drawn with
    # its nose at +up, so the plot rotation must offset that convention by π/2.
    # Rotate by (pitch − π/2) (CCW positive in matplotlib): at pitch = π/2 the
    # rocket is drawn vertical; at pitch = 0 (nose pointing +X) it lies along +X.
    theta = pitch - np.pi / 2.0
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    world = local @ rot.T + np.array([x, altitude], dtype=np.float64)
    return world


def _rocket_base_and_exhaust_dir(
    x: float, altitude: float, pitch: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Base point of the rocket and the unit exhaust direction in plot coords.

    Exhaust points along the body -Y_local axis (i.e. out of the engine
    nozzle, away from the nose), rotated by the same ``(pitch − π/2)`` offset
    as :func:`_rocket_polygon` so it points downward when the rocket is upright.
    """
    theta = pitch - np.pi / 2.0
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    base_local = np.array([0.0, -_ROCKET_BODY_HEIGHT_M / 2], dtype=np.float64)
    base_world = rot @ base_local + np.array([x, altitude], dtype=np.float64)
    exhaust_dir = rot @ np.array([0.0, -1.0], dtype=np.float64)
    return base_world, exhaust_dir


def animate_side_view(traj: Trajectory, out_path: Path, fps: int) -> None:
    """Write a 2D X-Z side-view MP4 animation of the trajectory.

    The rocket is drawn as a 5-vertex polygon (body + nose) rotated by
    pitch. A thrust arrow extends from the rocket base along the engine
    exhaust direction, scaled by commanded throttle. A faded polyline
    shows the past trajectory. A HUD in the corner reports t, throttle,
    fuel, pitch, vz.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T = int(traj.t.shape[0])
    if T == 0:
        return

    x = traj.pos_NED[:, 0]
    altitude = -traj.pos_NED[:, 2]
    pitch = traj.euler[:, 1]
    throttle = np.clip(traj.action[:, 0], 0.0, 1.0)

    pad_radius = float(traj.meta.get("pad_radius_m", 30.0))
    oob_radius = float(traj.meta.get("oob_cylinder_radius_m", 200.0))
    ceiling = float(traj.meta.get("oob_ceiling_m", 600.0))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-oob_radius, oob_radius)
    ax.set_ylim(0.0, ceiling)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m, NED)")
    ax.set_ylabel("altitude (m, -NED Z)")
    ax.set_title(
        f"Episode {traj.meta.get('episode_idx', '?')}  "
        f"outcome={traj.meta.get('outcome', '?')}  "
        f"return={traj.meta.get('return_total', float('nan')):.2f}"
    )

    # Ground line + pad highlight.
    ax.axhline(0.0, color="0.3", lw=1.0)
    ax.plot(
        [-pad_radius, pad_radius],
        [0.0, 0.0],
        color="C2",
        lw=4.0,
        solid_capstyle="butt",
        label="pad",
    )

    # Artists updated each frame.
    rocket_patch = Polygon(_rocket_polygon(x[0], altitude[0], pitch[0]), closed=True, color="C0")
    ax.add_patch(rocket_patch)

    (trail_line,) = ax.plot([], [], color="C0", alpha=0.35, lw=1.0)

    thrust_arrow_artist: list[FancyArrow | None] = [None]  # in list for closure mutability

    hud = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontfamily="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="0.6"),
    )

    def _update(k: int) -> tuple[Any, ...]:
        rocket_patch.set_xy(_rocket_polygon(x[k], altitude[k], pitch[k]))

        trail_line.set_data(x[: k + 1], altitude[: k + 1])

        if thrust_arrow_artist[0] is not None:
            thrust_arrow_artist[0].remove()
            thrust_arrow_artist[0] = None

        if throttle[k] > 0.0:
            base, direction = _rocket_base_and_exhaust_dir(x[k], altitude[k], pitch[k])
            arrow_len = throttle[k] * _THRUST_ARROW_MAX_M
            dx, dy = direction * arrow_len
            thrust_arrow_artist[0] = ax.arrow(
                base[0],
                base[1],
                dx,
                dy,
                head_width=4.0,
                head_length=4.0,
                fc="C3",
                ec="C3",
                length_includes_head=True,
                alpha=0.85,
            )

        hud.set_text(
            "\n".join(
                [
                    f"t      = {traj.t[k]:6.2f} s",
                    f"alt    = {altitude[k]:7.1f} m",
                    f"vz     = {traj.vel_NED[k, 2]:+7.2f} m/s",
                    f"pitch  = {np.rad2deg(pitch[k]):+7.2f} deg",
                    f"thrtl  = {throttle[k]:6.2%}",
                    f"fuel   = {traj.fuel_kg[k]:7.1f} kg",
                ]
            )
        )
        return rocket_patch, trail_line, hud

    anim = FuncAnimation(fig, _update, frames=T, blit=False, interval=1000 / max(fps, 1))
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3200)
    anim.save(str(out_path), writer=writer, dpi=100)
    plt.close(fig)
