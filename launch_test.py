import numpy as np
from earth2studio.run_orbit import run

from earth2studio.data import NCAR_ERA5
#from earth2studio.data import WB2ERA5

from earth2studio.io import ZarrBackend
from earth2studio.lexicon.ncar import NCAR_ERA5Lexicon
from earth2studio.models.dx import OrbitGlobalPrecip9_5M

package = OrbitGlobalPrecip9_5M.load_default_package()
orbit = OrbitGlobalPrecip9_5M.load_model(package, "global", "9.5m", "precipitation")

data = NCAR_ERA5()
#data = WB2ERA5()
io = ZarrBackend("outputs/aifs_forecast.zarr")

#NCAR_ERA5
time = np.datetime64('2025-01-01T12:00:00')
#run(["2025-01-02T00:00:00"], orbit, data, io)
run([time], orbit, data, io)
