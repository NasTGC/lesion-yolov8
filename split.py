import splitfolders

splitfolders.ratio("new_data", output="data_split",
    seed=1337, ratio=(.7, .3), group_prefix=None, move=False) # default values