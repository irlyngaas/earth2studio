
from datetime import datetime
import numpy as np
import torch
from loguru import logger

from earth2studio.utils.time import to_time_array

from earth2studio.data import DataSource, prep_data_array
from earth2studio.io import IOBackend
from earth2studio.models.dx import OrbitGlobalPrecip9_5M

def run(
    time: list[str] | list[datetime] | list[np.datetime64],
    orbit: OrbitGlobalPrecip9_5M,
    data: DataSource,
    io: IOBackend,
) -> IOBackend:

    logger.info("Running ORBIT-2 inference!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")

    orbit = orbit.to(device)

    time_in = time

    time = to_time_array(time)
    x, coords = prep_data_array(
        data(time, orbit.input_coords()["variable"]), device=device
    )
    logger.success(f"Fetched data from {data.__class__.__name__}")

    initial_hour = int(time_in[0].astype(str)[11:13]) 

    #time_list = []
    #for i in range(24):
    #    time_list.append((time_in + np.timedelta64(1, 'h') * (-1 * i)).astype(str))
    #print(time_list)
    #time = to_time_array(time_list)
    #p, coords = prep_data_array(
    #    data(time, ["cp", "lsp"]), device=device
    #)
    #print("P", p.shape)
    
    for i in range(24):
        time_i = time_in + np.timedelta64(1, 'h') * (-1 * i)
        time = to_time_array(time_i)
        p, coords = prep_data_array(
            data(time, ["cp", "lsp"]), device=device
        )
        t, coords = prep_data_array(
            data(time, ["t2m"]), device=device
        )
        if i == 0:
            p_total = p
            t_max = t
            t_min = t
        else:
            p_total = p_total + p
            t_max = torch.max(t_max, t)
            t_min = torch.min(t_min, t)
    p1 = p_total[:,:,0]
    p2 = p_total[:,:,0]
    p = p1+p2
    t_min.squeeze(0)
    t_max.squeeze(0)
    x = torch.cat((x,p,t_max,t_min), dim=1)
        

    output_coords = orbit.output_coords(orbit.input_coords())
    io.add_array(output_coords, output_coords["variable"], overwrite=True)

    logger.info("Inference Starting")
    x, coords = orbit(x, coords)




    check_data(time, data, orbit, device)


    return io
