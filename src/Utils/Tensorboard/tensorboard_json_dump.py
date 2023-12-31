# Reads Tensorboard log file, extracting scalars (e.g. loss and accuracy) and writes them to a json file.
# see https://stackoverflow.com/questions/41074688/how-do-you-read-tensorboard-files-programmatically/41083104#41083104

import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import json

def extract_scalars_from_tfevent_file(path):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    
#    print(event_acc.Tags())
    
    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        scalars[tag] = [{'wall_time': e.wall_time, 'step': e.step, 'value': e.value} for e in events]

    return scalars

def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0},)
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

scalars = extract_scalars_from_tfevent_file(sys.argv[1])

# Write the scalars to a json file
with open(sys.argv[2], 'w') as f:
    json.dump(scalars, f)
