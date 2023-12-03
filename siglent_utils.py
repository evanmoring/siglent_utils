import numpy as np
from math import floor, sqrt
import csv 
from scipy.fftpack import  rfftfreq, rfft
from scipy import signal
import xml.etree.ElementTree as ET 
import argparse

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
    o = {
        "t": dx,
        "v": dy,
        "sample_interval": si
    }
    
    return o

def plot_wave(l):
    dx = l["t"]
    dy = l["v"]
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    plt.figure(figsize = (20, 10))
    plt.subplot(211)
    plt.plot(dx,dy )
    plt.title("Signal")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1fV'))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f seconds'))
    plt.show()

def get_rms(a):
    a = a ** 2
    s = sum(a)/a.size
    s = sqrt(s) # rms volts
    return s

def get_power(a, load):
    s = get_rms(a)
    return (s**2) / load # watts


def get_hamming_fft(l):
    a = l["v"] * signal.windows.hamming(l["v"].size)
    yf = rfft(a)
    xf = rfftfreq(a.size, l["sample_interval"])
    return {"freq": xf, "v":yf}

def get_THD(h):
    xf = h["freq"]
    yf = h["v"]
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
            #if v > 1000:
                #harmonics_amp += v ** 2
            harmonics_amp += v ** 2
                
    # THD for voltage
    return 100 * sqrt(harmonics_amp)/freq_amp

def plot_fft(h, log_y = True):
    xf = h["freq"]
    yf = h["v"]
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    new_x = []
    new_y = []
    i = 0
    m = 0
    freq = 0
    freq_index = 0
    fig,ax = plt.subplots()
    ax.set_xscale('log')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1fdB'))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1fhz'))
    yf = convert_to_dBV(yf)
    plt.plot(xf, yf)
    plt.title("FFT")
    plt.show()

def convert_to_dBV(a):
    a = np.abs(a)
    a = np.log10(a)
    a = 20* a
    return a

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Siglent Utilities',
                    description='Helper functions for the Siglent SDS 1202X-E oscilloscope',
                    epilog='Example command: \'python3 siglent_utils.py -f SDS00001.csv -s SDS00001.xml -t\'')

    parser.add_argument('--filename', '-f', help = "filename of csv")
    parser.add_argument('--setup', '-s', required = False, help = "filename of xml setup file. Useful for getting probe attenuation factor")
    parser.add_argument('--display_fft', action = "store_true")
    parser.add_argument('--display_wave', action = "store_true")
    parser.add_argument('--power', '-p', type = float, help = "RMW watts. Takes the load as an argument")
    parser.add_argument('--voltage', '-v', action = "store_true", help = "RMS voltage")
    parser.add_argument('--thd', '-t', action = "store_true", help = "actually total harmonic distortion + noise")
    parser.add_argument('--probe_factor', '-pf', type = float, help = "probe attenuation factor")

    args = parser.parse_args()
    if args.probe_factor:
        l = load_csv(args.filename, probe_factor = args.probe_factor)
    elif args.setup:
        l = load_csv(args.filename, setup_xml = args.setup)
    else:
        l = load_csv(args.filename)

    hamming = None

    if args.thd:
        hamming = get_hamming_fft(l)
        print(f"THD: {get_THD(hamming): .2f}%")

    if args.voltage:
        v = get_rms(l["v"])
        print(f"RMS volts: {v: .4f}")

    if args.power:
        w = get_power(l["v"], args.power)
        print(f"RMS watts: {w: .4f}")

    if args.display_fft:
        hamming = hamming or get_hamming_fft(l)
        plot_fft(hamming)

    if args.display_wave:
        plot_wave(l)
