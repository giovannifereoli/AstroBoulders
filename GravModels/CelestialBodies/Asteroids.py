import os

# Reference: https://3d-asteroids.space/asteroids/


class Asteroid:
    def __init__(self, body_name, radius, min_radius, density, volume, shape_model):
        self.body_name = body_name
        self.G = 6.67408e-11  # Gravitational constant
        self.radius = radius  # Mean Brillouin Sphere radius [m]
        self.min_radius = min_radius  # Minimum Brillouin Sphere radius [m]
        self.density = density  # Density [kg/m^3]
        self.volume = volume  # Volume [m^3]
        self.mass = self.volume * self.density  # Mass in kilograms
        self.mu = self.G * self.mass  # Gravitational parameter

        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = os.path.join(current_dir, shape_model)  # Path to shape model file


class Didymos(Asteroid):
    def __init__(self):
        super().__init__(
            "DIDYMOS_BARYCENTER",
            radius=422.72084951400757,
            min_radius=390.4433846473694,
            density=2170,
            volume=0.2485481753057092 * 10**9,
            shape_model="../Files/ShapeModels/Didymos/Didymos_shape_model_1.obj",
        )


class Dimorphos(Asteroid):
    def __init__(self):
        super().__init__(
            "DIMORPHOS_BARYCENTER",
            radius=104.00000214576721,
            min_radius=66.50000065565109,
            density=2170,
            volume=0.002308071031887155 * 10**9,
            shape_model="../Files/ShapeModels/Dimorphos/Dimorphos_shape_model.obj",
        )


class Bennu(Asteroid):
    def __init__(self):
        super().__init__(
            "BENNU_BARYCENTER",
            radius=289.39155844633757,
            min_radius=None,  # Update with actual value if available
            density=1250,
            volume=63255149.177708544,
            shape_model="../Files/ShapeModels/Bennu/Bennu_shape_model.obj",
        )
