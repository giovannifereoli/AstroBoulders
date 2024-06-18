import os

# Reference: NASA Planetary Factsheets


class Planet:
    def __init__(self, body_name, radius, min_radius, density, volume, shape_model):
        self.body_name = body_name
        self.G = 6.67430e-11  # Gravitational constant
        self.radius = radius  # Equatorial radius in meters
        self.min_radius = min_radius  # Polar radius in meters (or minimum radius)
        self.density = density  # Density in kg/m^3
        self.volume = volume  # Volume in cubic meters
        self.mass = self.volume * self.density  # Mass in kilograms
        self.mu = self.G * self.mass  # Gravitational parameter

        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = os.path.join(current_dir, shape_model)  # Path to shape model file


class Earth(Planet):
    def __init__(self):
        super().__init__(
            body_name="Earth",
            radius=6378137.0,
            min_radius=6356752.3,
            density=5515,
            volume=1.08321e21,
            shape_model="../Files/ShapeModels/Earth/Earth_shape_model.obj",
        )


class Moon(Planet):
    def __init__(self):
        super().__init__(
            body_name="Moon",
            radius=1737100.0,
            min_radius=1737100.0,
            density=3340,
            volume=2.199e10,
            shape_model="../Files/ShapeModels/Moon/Moon_shape_model.obj",
        )
