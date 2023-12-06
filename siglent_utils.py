import numpy as np
from math import floor, sqrt
import csv 
import copy
from scipy.fftpack import  rfftfreq, rfft
from scipy import signal
import xml.etree.ElementTree as ET 
import argparse

def get_probe_type_from_xml(filename):
    # probe type is only saved in "setup" xml, not with csv
    tree = ET.parse(filename)
    root = tree.getroot()
    ch1 = root.find("ch_setup").find("CH1").find("probe").text
    ch2 = root.find("ch_setup").find("CH2").find("probe").text
    probe_map = [.1,.2,.5,1,2,5,10,20,50,100,200,500,1000,2000,5000]
    p = {
        "CH1": probe_map[int(ch1)],
        "CH2": probe_map[int(ch2)]
    }
    return p

def load_csv(filename, setup_xml = "", probe_factor = 1):
    probe_map = {"CH1": probe_factor, "CH2": probe_factor}
    if setup_xml:
        probe_map = get_probe_type_from_xml(setup_xml)
        
    with open(filename, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    s = data[1][1]
    # this line looks like `CH1:0.0000010000000 CH2:0.0000010000000`
    s = " ".join(s.split(":"))
    s = s.split()
    channels = int(len(s)/2)
    channel_names = []
    for i in range(channels):
        channel_names.append(s[i*2])

    o = {}
    # skip metadata
    data = data[12:]
    dx = np.array([float(x[0]) for x in data]) 
    for i, name in enumerate(channel_names):
        si = float(s[1 + i*2])
        pf = probe_map[name]
        dy = np.array([float(x[i + 1]) * pf for x in data]) 
        o[name] = Channel(name, si, dx, dy)
    return o

class Channel:
    def __init__(self, channel_index: int, si: float, time: list, volts: list):
        self.channel_index = channel_index
        self.si = si
        self.time = time
        self.volts = volts
        self.hamming = None

    def plot_wave(self):
        dx = self.time
        dy = self.volts
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        #plt.figure(figsize = (20, 10))
        #plt.subplot(211)
        fig,ax = plt.subplots()
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1fV'))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f seconds'))
        plt.plot(dx,dy )
        plt.title("Signal")
        plt.show()

    def get_rms(self):
        a = copy.deepcopy(self.volts)
        a = a ** 2
        s = sum(a)/a.size
        s = sqrt(s) # rms volts
        return s

    def get_power(self, load: float):
        s = self.get_rms()
        return (s**2) / load # watts


    def get_hamming_fft(self):
        v = np.array(self.volts)
        a = v * signal.windows.hamming(v.size)
        yf = rfft(a)
        xf = rfftfreq(a.size, self.si)
        self.hamming = {"freq": xf, "v":yf}

    def get_THD(self):
        if not self.hamming:
            self.get_hamming_fft()
        h = self.hamming
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

    def plot_fft(self, log_y = True):
        if not self.hamming:
            self.get_hamming_fft()
        h = self.hamming
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
    parser.add_argument('--channel', '-c', choices = ["CH1", "CH2"], default = "CH1", type = str, help = "which scope channel to use in the form CH1. Defaults to CH1")
    parser.add_argument('--voltage', '-v', action = "store_true", help = "RMS voltage")
    parser.add_argument('--thd', '-t', action = "store_true", help = "actually total harmonic distortion + noise")
    parser.add_argument('--probe_factor', '-pf', type = float, help = "probe attenuation factor")

    args = parser.parse_args()
    if args.probe_factor:
        channel_dict = load_csv(args.filename, probe_factor = args.probe_factor)
    elif args.setup:
        channel_dict = load_csv(args.filename, setup_xml = args.setup)
    else:
        channel_dict = load_csv(args.filename)

    channel = args.channel

    c = channel_dict[channel]

    if args.thd:
        thd = c.get_THD()
        print(f"THD: {thd: .2f}%")

    if args.voltage:
        v = c.get_rms()
        print(f"RMS volts: {v: .4f}")

    if args.power:
        w = c.get_power(args.power)
        print(f"RMS watts: {w: .4f}")

    if args.display_fft:
        c.plot_fft()

    if args.display_wave:
        c.plot_wave()
