"""Plant parameters for the planned budget hardware build.

Rig (direct drive, no gearboxes):
  * shoulder: iPower GM5208-12 gimbal motor + AS5048A encoder, fixed to the frame
  * elbow:    iPower GM3506 gimbal motor + AS5048A encoder; its STATOR is
              bolted to the tip of link 1, its rotor drives link 2
              -> the whole elbow motor mass is carried by link 1
  * drivers:  2x SimpleFOC Mini (DRV8313), one ESP32/STM32 board
  * links:    3D-printed (PETG for link 1), 12 cm / 10 cm, 15 g tip mass

All values are datasheet numbers or estimates; replace them with measured /
identified values once the rig exists.

    python hardware.py            # print the plant parameters
"""
from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass
class Motor:
    name: str
    mass: float            # kg (motor + encoder)
    rated_torque: float    # N m, continuous
    peak_torque: float     # N m
    rotor_inertia: float   # kg m^2 of the rotating bell (direct drive)
    price_usd: float
    notes: str = ""


# Datasheet "load torque" is given in g.cm: rated = low end of the range,
# peak = high end.  Rotor inertia is a rough m_bell * r^2 estimate.
SHOULDER = Motor("iPower GM5208-12 + AS5048A", 0.209, 0.18, 0.25,
                 4.0e-5, 62, "15.2 ohm, 12N14P, 3-5S")
ELBOW = Motor("iPower GM3506 + AS5048A", 0.072, 0.06, 0.10,
              6.0e-6, 42, "5.6 ohm, 24N22P, 12 V")


@dataclass
class PrintedBeam:
    """Rectangular 3D-printed beam, joint-to-joint length `length`."""
    length: float                  # m, joint axis to joint axis / tip
    width: float = 0.020           # m
    height: float = 0.010          # m
    density: float = 1270.0        # kg/m^3 (PETG 1270, PLA 1240)
    fill_fraction: float = 0.5     # walls + infill, effective solid fraction
    hub_mass: float = 0.010        # kg of screws/inserts/clamp at the driven end
    extra: list = field(default_factory=list)  # [(mass, distance_from_joint)]

    @property
    def beam_mass(self):
        return self.density * self.fill_fraction * self.length * self.width * self.height


def link_lumped(beam, lumps, extra_inertia=0.0):
    """Mass, COM distance and inertia ABOUT THE JOINT AXIS (DFKI convention)
    of a uniform beam plus point masses [(m, r), ...]."""
    mb, L = beam.beam_mass, beam.length
    parts = [(mb, L / 2, mb * L ** 2 / 3.0), (beam.hub_mass, 0.0, 0.0)]
    for m, r in list(lumps) + list(beam.extra):
        parts.append((m, r, m * r ** 2))
    m = sum(p[0] for p in parts)
    r = sum(p[0] * p[1] for p in parts) / m
    I = sum(p[2] for p in parts) + extra_inertia
    return m, r, I


def build(l1=0.12, l2=0.10, tip_mass=0.015, elbow_mount_mass=0.015,
          torque_mode="rated"):
    """Return a dict with the plant keys config.py expects."""
    beam1 = PrintedBeam(length=l1, width=0.018, height=0.008, hub_mass=0.010)
    beam2 = PrintedBeam(length=l2, width=0.015, height=0.006, hub_mass=0.008)
    # Direct drive: each motor's bell turns with the link it drives, so its
    # inertia is added to that link (DFKI's geared Ir model is not used).
    m1, r1, I1 = link_lumped(beam1, [(ELBOW.mass + elbow_mount_mass, l1)],
                             SHOULDER.rotor_inertia)
    m2, r2, I2 = link_lumped(beam2, [(tip_mass, l2)] if tip_mass else [],
                             ELBOW.rotor_inertia)
    if torque_mode == "rated":
        tl = [SHOULDER.rated_torque, ELBOW.rated_torque]
    elif torque_mode == "peak":
        tl = [SHOULDER.peak_torque, ELBOW.peak_torque]
    else:
        raise ValueError(torque_mode)
    return dict(
        MASS=[m1, m2], LENGTH=[l1, l2], COM=[r1, r2], INERTIA=[I1, I2],
        MOTOR_INERTIA=0.0, GEAR_RATIO=1, TORQUE_LIMIT=tl,
    )


def static_torque_demand(p):
    """Worst-case gravity torque (arm horizontal and straight)."""
    m1, m2 = p["MASS"]; l1, _ = p["LENGTH"]; r1, r2 = p["COM"]
    return np.array([(m1 * r1 + m2 * (l1 + r2)) * G, m2 * r2 * G])


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    p = build()
    need = static_torque_demand(p)
    tl = np.array(p["TORQUE_LIMIT"])
    print(f"{SHOULDER.name} (shoulder) + {ELBOW.name} (elbow), "
          f"motors ${SHOULDER.price_usd + ELBOW.price_usd}")
    for k in ("MASS", "LENGTH", "COM", "INERTIA", "TORQUE_LIMIT"):
        print(f"  {k:13s} {np.round(p[k], 5)}")
    print(f"  static demand {need.round(3)} N m  ->  rated margin {(tl / need).round(2)}x")