#!/usr/bin/env python3
"""mask_census_s31.py (S31, D-33 integrity; CPU, read-only): the set of pixel values of the training masks of every domain, Prostate and Fundus.
The oracle gate applies the labeled batch's label mapping to the unlabeled domains' masks; this checks that every domain of a benchmark is encoded alike."""
import os
import numpy as np
from PIL import Image
for root in ('../../data/ProstateSlice', '../../data/Fundus'):
    for d, dirs, fs in os.walk(root):
        dirs.sort()
        parts = d.replace('\\', '/').split('/'); png = sorted(f for f in fs if f.lower().endswith('.png'))
        if 'mask' not in parts or 'train' not in parts or 'test' in parts or not png:
            continue
        vals = set()
        for f in png:
            m = Image.open(os.path.join(d, f))
            if m.mode == 'RGB':
                m = m.convert('L')
            vals |= set(np.unique(np.array(m)).tolist())
        v = sorted(vals); s = str(v) if len(v) <= 8 else f'{v[:4]} ... {v[-2:]}'
        print(f'{d}: {len(png)} files, {len(v)} values {s}')
print('census done')
