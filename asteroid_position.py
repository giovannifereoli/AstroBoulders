import numpy as np
import spiceypy as spice

# Load SPICE kernels (replace with appropriate paths)
spice.furnsh("Kernels/de432s.bsp")  # SPICE planetary ephemeris
spice.furnsh("Kernels/HERA_sc_LPC_EMA_2024c.bsp")  # SPICE HERA trajectory
spice.furnsh("Kernels/didymos_hor_200101_300101_v01.bsp")  # SPICE kernel for Didymos
spice.furnsh("Kernels/naif0012.tls")  # Leap seconds kernel

# We now need to define an epoch (date) to query for positions of the various bodies
# Keep in mind that the kernels do not contain the information for all epochs, only for some
# ranges. You will get an error if you query outside of these bounds
et = spice.str2et("2028-Jan-04 12:00:00")
print(et)

# Get the state of the Earth with respect to the Sun
earth_state, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "SUN")

# Get the state of Didymos with respect to the Sun
didymos_state, _ = spice.spkezr("DIDYMOS_BARYCENTER", et, "J2000", "NONE", "SUN")

# Calculate the position of Didymos with respect to the Earth
rel_state = earth_state - didymos_state
r = rel_state[:3]
v = rel_state[3:]

# Calculate the velocity of Didymos with respect to the Earth
print("Position vector at epoch: ", r)
print("Velocity vector at epoch: ", v)
