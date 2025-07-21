#!/usr/bin/env python

import os

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
os.system(f"rm -rf {_CUR_DIR}/*.toml")

DOMAINS = [
    "blocks",
    "transport",
    "woodworking",
]


for domain in DOMAINS:
    file = f"{_CUR_DIR}/{domain}.toml"
    with open(file, "w") as f:
        f.write(f"domain_pddl = 'cost_partition/icaps26/{domain}/domain.pddl'\n")
        f.write(f"tasks_dir = 'cost_partition/icaps26/{domain}/training'\n")
        f.write(f"validation_dir = 'cost_partition/icaps26/{domain}/validation'\n")
