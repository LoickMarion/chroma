import unittest
import numpy as np

from chroma.geometry import Solid, Geometry
from chroma.make import box
from chroma.sim import Simulation
from chroma.demo.optics import water
from chroma.event import Photons
from chroma.rootimport import ROOT
ROOT.gROOT.SetBatch(1)

from chroma.histogram import Histogram, rootify

class TestRayleigh(unittest.TestCase):
    def setUp(self):
        self.cube = Geometry(water)
        self.cube.add_solid(Solid(box(100,100,100), water, water))
        self.cube.build(use_cache=False)
        self.sim = Simulation(self.cube, geant4_processes=0)

        nphotons = 100000
        pos = np.tile([0,0,0], (nphotons,1)).astype(np.float32)
        dir = np.tile([0,0,1], (nphotons,1)).astype(np.float32)
        pol = np.zeros_like(pos)
        phi = np.random.uniform(0, 2*np.pi, nphotons).astype(np.float32)
        pol[:,0] = np.cos(phi)
        pol[:,1] = np.sin(phi)
        t = np.zeros(nphotons, dtype=np.float32)
        wavelengths = np.empty(nphotons, np.float32)
        wavelengths.fill(400.0)

        self.photons = Photons(pos=pos, dir=dir, pol=pol, t=t, wavelengths=wavelengths)

    def testAngularDistributionPolarized(self):
        # Fully polarized photons
        self.photons.pol[:] = [1.0, 0.0, 0.0]

        photons_end = self.sim.simulate([self.photons], keep_photons_end=True, max_steps=1).next().photons_end
        aborted = (photons_end.flags & (1 << 31)) > 0
        self.assertFalse(aborted.any())

        # Compute the dot product between initial and final dir
        rayleigh_scatters = (photons_end.flags & (1 << 4)) > 0
        cos_scatter = (self.photons.dir[rayleigh_scatters] * photons_end.dir[rayleigh_scatters]).sum(axis=1)
        theta_scatter = np.arccos(cos_scatter)
        h = Histogram(bins=100, range=(0, np.pi))
        h.fill(theta_scatter)
        h = rootify(h)

        # The functional form for polarized light should be
        # (1 + \cos^2 \theta)\sin \theta according to GEANT4 physics
        # reference manual.
        f = ROOT.TF1("pol_func", "[0]*(1+cos(x)**2)*sin(x)", 0, np.pi)
        h.Fit(f)
        self.assertGreater(f.GetProb(), 1e-3)

