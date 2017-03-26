#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 17:59:02 2017

@author: gabriel
"""
import matplotlib.pyplot as plt
d = {'val_loss': [1.8045902869935098, 1.4066245407019711, 1.2997468097078735, 1.2745861446160505, 1.2329056663147548, 1.2066122427854167, 1.2119543196559841, 1.2145212251186717, 1.1980790525257152, 1.1948087133301326, 1.1833698348663615, 1.1939644900230297, 1.1967664097922044, 1.185976298001868, 1.1942095388919849, 1.1698082122796543, 1.1986598408981965, 1.1847895283449028, 1.192664647577609, 1.1756875767769552, 1.1701122989094435, 1.1814948260872917], 'loss': [2.1652434962545328, 1.5534147051158707, 1.3435548133916524, 1.262847356438493, 1.2188634450587101, 1.1926006059163556, 1.171446492138186, 1.1609552109744239, 1.1502167024923, 1.1448024740279861, 1.1370630037979961, 1.1331649956908658, 1.1301752989770715, 1.1284438054037595, 1.1276413518358728, 1.1216932110936699, 1.1280921410441405, 1.1276944167712275, 1.1288055684128133, 1.127366394699981, 1.1290622345550887, 1.1284181825353552], 'val_categorical_accuracy': [0.52664299081881105, 0.63342880056901052, 0.66071714980452856, 0.67075266360908814, 0.67832894191511783, 0.68556917561704311, 0.68270515912030805, 0.68124642004993863, 0.68976209571363634, 0.6916561653438601, 0.69195402305952058, 0.69060220727306165, 0.69285523358337131, 0.69317600342963992, 0.69310726703371828, 0.69498606191111367, 0.68905181967525198, 0.69196166043684515, 0.69285523358291612, 0.69522282050165496, 0.69455073129617728, 0.69227479285161764], 'categorical_accuracy': [0.44946881635113273, 0.59637107244780418, 0.649525715255864, 0.66908136925494188, 0.68003536130705089, 0.68652718164988291, 0.69145332764072964, 0.69367582140347328, 0.69597087080948794, 0.6980253410151438, 0.69966166159503829, 0.70078627399244742, 0.70123306397095819, 0.70171995052580516, 0.70169894755772499, 0.70358539417715871, 0.70209991292959173, 0.70233476409873219, 0.70206172577043824, 0.70186888049648766, 0.70194716420193237, 0.70197389525312659]}

for key in d.keys():
    if 'loss' in key:
        plt.plot(d[key])
plt.legend([key for key in d.keys() if 'loss' in key])
plt.show()
for key in d.keys():
    if 'accuracy' in key:
        plt.plot(d[key])
        print(key)
plt.legend([key for key in d.keys() if 'accuracy' in key])
plt.show()