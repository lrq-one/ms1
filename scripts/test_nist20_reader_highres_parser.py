#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from rassp.data.nist20_reader import _parse_peak_line

cases = [
    (
        '269.1863 21.28 "C14H25N2O3=p-C6H13NO/1.2ppm;C17H23N3=p-C3H15O4/-8.7ppm 14/14"',
        [(269.1863, 21.28)],
    ),
    (
        '375.3005 80.12 "C23H39N2O2=p-H2O/-0.3ppm 12/12"; 393.3111 999.00 "p/-0.2ppm 12/12";',
        [(375.3005, 80.12), (393.3111, 999.00)],
    ),
    (
        '375   80; 393  999;',
        [(375.0, 80.0), (393.0, 999.0)],
    ),
]

for raw, expected in cases:
    got = _parse_peak_line(raw)
    print("RAW:", raw)
    print("GOT:", got)
    print("EXPECTED:", expected)
    assert len(got) == len(expected), (got, expected)
    for (gmz, gi), (emz, ei) in zip(got, expected):
        assert abs(gmz - emz) < 1e-6, (got, expected)
        assert abs(gi - ei) < 1e-6, (got, expected)

print("OK: high-res MSP peak parser is safe.")