import numpy as np
from numpy import pi 
from math import floor, sqrt
import math
import csv

from scipy.fftpack import  rfftfreq, rfft, irfft
from scipy import signal
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def get_probe_type_from_xml(filename):
    # probe type is only saved in "setup" xml, not with csv
    probe_map = [.1,.2,.5,1,2,5,10,20,50,100,200,500,1000,2000,5000]
    tree = ET.parse(filename)
    root = tree.getroot()
    p = root.find("ch_setup").find("CH1").find("probe").text
    return (probe_map[int(p)])

def load_csv(filename, setup_xml = "", probe_factor = 1):
    if setup_xml:
        probe_factor = get_probe_type_from_xml(setup_xml)
        
    with open(filename, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]
    # get sample interval
    si = (float(data[1][1][4:]))
    # skip metadata
    dx = [float(x[0]) for x in data[12:]] 
    dy = [float(x[1]) for x in data[12:]] 
    dx = np.array(dx)
    dy = np.array(dy)
    dy *= probe_factor
    
    return [dx, dy, si]

def plot_wave(dx, dy):
    plt.figure(figsize = (20, 10))
    plt.subplot(211)
    plt.plot(dx,dy )
    plt.title("Generated Signal")
    plt.show()

def get_power(a, load):
    a = a ** 2
    s = sum(a)/a.size
    s = sqrt(s) # rms volts
    s = (s ** 2 ) / load # watts
    print(f"Power: {s}W")

def get_hamming_fft(a, sample_interval):
    a = a * signal.windows.hamming(a.size)
    yf = rfft(a)
    xf = rfftfreq(a.size, sample_interval)
    return [xf, yf]

def get_THD(xf, yf):
    new_x = []
    new_y = []
    i = 0
    m = 0
    freq = 0
    freq_index = 0
    while xf[i] < 25000:
        new_x.append(xf[i])
        new_val = sqrt(yf[i]**2 + yf[i+1]**2)
        new_y.append(new_val)
        if new_val > m:
            m = new_val
            freq = xf[i]
            freq_index = int(i/2)
        i += 2
    freq_amp = 0
    harmonics_amp = 0
    for i, v in enumerate(new_y):
        # handle some spillage
        if freq_index - 5 < i < freq_index + 5:
            freq_amp += v
        else:
            if v > 1000:
                harmonics_amp += v ** 2
                
    # THD for voltage
    print(f"THD: {100 * sqrt(harmonics_amp)/freq_amp}%")

def plot_fft(xf, yf, log_y = True):
    new_x = []
    new_y = []
    i = 0
    m = 0
    freq = 0
    freq_index = 0
    fig,ax = plt.subplots()
    ax.set_xscale('log')
    yf = convert_to_dBV(yf)
    plt.plot(xf, yf)
    plt.show()

def convert_to_dBV(a):
    a = np.abs(a)
    a = np.log10(a)
    a = 20* a
    return a

filename = "/home/evan/amp_measurements/17W.csv"
l = load_csv(filename, probe_factor = 10)
get_power(l[1], 8)
b = get_hamming_fft(l[1], l[2])
get_THD(b[0], b[1])
plot_wave(l[0],l[1])
plot_fft(b[0], b[1])
