# Parameter sources for ModerateFidelityDynamics

The vehicle parameters in `configs/env.yaml` (consumed by
`dynamics.moderate_fidelity.ModerateFidelityParams`) are grounded in
publicly documented Falcon-9 / Merlin-1D values. This document gives
the source for each.

> **Note.** Earlier versions of this project claimed parameters were
> "grounded in RocketPy". That was inaccurate — RocketPy's default
> rocket ("Calisto") is a small hobbyist model, not a Falcon-class
> vehicle. The values below are instead sourced from public engine /
> vehicle specifications, so the project's claims align with what the
> code actually models.

## Vehicle mass & thrust

| Parameter | Value | Source / rationale |
|---|---|---|
| `dry_mass_kg` | 25 000 | Approximate Falcon-9 first-stage dry mass with empty propellant tanks (≈ 22 t structure + landing legs + Merlin engine + residuals). Public SpaceX press materials and FAA filings. |
| `initial_fuel_kg` | 5 000 | Landing-burn-only propellant reserve. The simulation models the *terminal descent* (last few hundred metres), not ascent — so the full ~400 t propellant load is irrelevant. |
| `max_thrust_N` | 845 000 | Merlin-1D vacuum-derated sea-level thrust per [SpaceX Falcon-9 user guide](https://www.spacex.com/media/Capabilities&Services.pdf) (typical published value 845 kN; throttled range 392–845 kN). |
| `isp_s` | 282 | Merlin-1D sea-level specific impulse. SpaceX engine specs. |
| `throttle_min` | 0.4 | Merlin-1D minimum sustainable throttle (~40–46%); below this the engine flames out. Conservative end of the published range. |
| `throttle_max` | 1.0 | Full thrust. |

## Aerodynamics

| Parameter | Value | Source / rationale |
|---|---|---|
| `drag_coefficient` | 0.75 | Order-of-magnitude scalar Cd for a slender cylindrical body in subsonic flight. A real Falcon-9 has Cd varying with Mach number and angle of attack; this is a single-value approximation valid for the terminal-descent regime. See Anderson, *Fundamentals of Aerodynamics* §1.5. |
| `reference_area_m2` | 10.5 | Cross-section of Falcon-9 first stage (radius ≈ 1.83 m → π·r² ≈ 10.5). Public vehicle dimensions. |

## Moments of inertia

| Parameter | Value | Source / rationale |
|---|---|---|
| `inertia_xx` | 2.5e6 | Pitch/yaw inertia: I_xx ≈ I_yy for a slender cylinder. Order-of-magnitude estimate from `I ≈ (1/12) m L²` with m = 30 t, L = 47 m → ≈ 5.5e6; scaled down to 2.5e6 to reflect that mass is distributed (not point-mass at extremities). |
| `inertia_yy` | 2.5e6 | Equal to I_xx by cylindrical symmetry (modulo CoM offset from geometric centre). |
| `inertia_zz` | 1.2e5 | Roll inertia: `I ≈ (1/2) m r²` with m = 30 t, r = 1.83 m → ≈ 5e4. Scaled up to account for non-uniform mass distribution. |
| `engine_lever_arm_m` | 15 | Distance from centre of mass to engine gimbal pivot. Order-of-magnitude estimate for a Falcon-9 with most fuel at the bottom (CoM ~1/3 up from the bottom, engine at the very bottom → ~15 m below CoM for a 47 m stage). |

## Actuator limits

| Parameter | Value | Source / rationale |
|---|---|---|
| `gimbal_max_rad` | 0.0873 | ±5° gimbal range. Conservative for Merlin-1D, which has been publicly reported as ~10° peak. |

## Environment

| Parameter | Value | Source / rationale |
|---|---|---|
| `G_EARTH` (constant) | 9.81 m/s² | Local gravitational acceleration at Earth's surface. Constant across the project's drop altitudes (< 600 m → < 0.02% variation). |
| `G0_TSIOLKOVSKY` (constant) | 9.80665 m/s² | Standard gravity in the Tsiolkovsky / specific-impulse relation. *Definitional*, not local-gravity. |
| `RHO_AIR_SEA_LEVEL` (constant) | 1.225 kg/m³ | ISA sea-level air density. Constant — no altitude variation in moderate fidelity. |

## Known limitations of these values

1. **Inertia tensor is constant.** As fuel burns, real CoM and inertia change.
   Moderate fidelity ignores this — error grows as fuel fraction drops.
2. **Drag is single-coefficient.** Real Cd depends on Mach, AoA, surface
   roughness. Acceptable for terminal descent (< 0.5 Mach, near-vertical
   attitude), not for high-speed phases.
3. **Atmosphere is constant.** No ρ(h) variation — sea-level value used
   throughout. Drop altitudes < 600 m → density variation < 8% (still
   simplified).
4. **Single gimballed engine, no RCS.** Roll torque is identically zero
   in this model (see `dynamics/equations_of_motion.py::compute_gimbal_torque_body`).
   Real Falcon-9 uses cold-gas RCS thrusters for roll. Documented limitation.
5. **No engine spool-up dynamics.** Throttle commands take effect within
   one control tick (instantaneous response). Real engines have ~100 ms
   throttle bandwidth.

These are the explicit *moderate*-fidelity simplifications. The architecture
admits a `HighFidelityDynamics` subclass that can address them without
touching anything outside `dynamics/` (see `docs/ARCHITECTURE.md` §2).
