# ===========================================================================
# Copyright (C) 2021-2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import argparse
import numpy as np

from ifxAvian import Avian
from scipy import signal
from scipy import constants
import matplotlib.pyplot as plt
import time
import csv
from internal.fft_spectrum import *
# -------------------------------------------------
# Computation
# -------------------------------------------------
class DistanceFFT_Algo:
    # Algorithm for computation of distance fft from raw data

    def __init__(self, config : Avian.DeviceConfig):
        # Common values initiation
        # config: dictionary with configuration for device used by set_config() as input

        self._numchirps = config.num_chirps_per_frame
        chirpsamples = config.num_samples_per_chirp

        # compute Blackman-Harris Window matrix over chirp samples(range)
        self._range_window = signal.blackmanharris(chirpsamples).reshape(1, chirpsamples)

        start_frequency_Hz = config.start_frequency_Hz
        end_frequency_Hz = config.end_frequency_Hz
        bandwidth_hz = abs(end_frequency_Hz-start_frequency_Hz)
        fft_size = chirpsamples * 2
        self._range_bin_length = (constants.c) / (2 * bandwidth_hz * fft_size / chirpsamples)

    def compute_distance(self, data):
        # Computes a distance for one chirp of data
        # data: single chirp data for single antenna

        # Step 1 - calculate range fft spectrum of the frame
        range_fft = fft_spectrum(data, self._range_window)

        # Step 2 - convert to absolute spectrum
        fft_spec_abs = abs(range_fft)

        # Step 3 - coherent integration of all chirps
        data = np.divide(fft_spec_abs.sum(axis=0), self._numchirps)

        # Step 4 - peak search and distance calculation
        skip = 8
        max = np.argmax(data[skip:])

        dist = self._range_bin_length * (max + skip)
        return dist, data

# -------------------------------------------------
# Presentation
# -------------------------------------------------
class Draw:
    # Draws plots for data - each antenna is in separated plot

    def __init__(self, config, max_range_m, num_ant):
        # Common values init
        # config:          dictionary with configuration for device used by set_config() as input
        # max_range_m:  maximum supported range
        # num_ant:      number of available antennas

        self._num_ant = num_ant;
        chirpsamples = config.num_samples_per_chirp
        self._pln = []

        self._fig, self._axs = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant+1)//2,2))
        self._fig.canvas.manager.set_window_title("Range FFT")
        self._fig.set_size_inches(17/3*num_ant, 4)

        self._dist_points = np.linspace(
            0,
            max_range_m,
            chirpsamples)

        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True


    def _draw_first_time(self, data_all_antennas):
        # Create common plots as well scale it in same way
        # data_all_antennas: array of raw data for each antenna
        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(0, self._num_ant):
            # This is a workaround: If there is only one plot then self._axs is
            # an object of the blass type AxesSubplot. However, if multiple
            # axes is available (if more than one RX antenna is activated),
            # then self._axs is an numpy.ndarray of AxesSubplot.
            # The code above gets in both cases the axis.
            if type(self._axs) == np.ndarray:
                ax = self._axs[i_ant]
            else:
                ax = self._axs

            data = data_all_antennas[i_ant]
            pln, = ax.plot(self._dist_points, data)
            ax.set_ylim(minmin, 1.05 *  maxmax)
            self._pln.append(pln)

            ax.set_xlabel("distance (m)")
            ax.set_ylabel("FFT magnitude")
            ax.set_title("Antenna #"+str(i_ant))
        self._fig.tight_layout()
        plt.ion()
        plt.show()

    def _draw_next_time(self, data_all_antennas):
        # Update plots
        # data_all_antennas: array of raw data for each antenna

        for i_ant in range(0, self._num_ant):
            data = data_all_antennas[i_ant]
            self._pln[i_ant].set_ydata(data)

    def draw(self, data_all_antennas):
        # Draw plots for all antennas
        # data_all_antennas: array of raw data for each antenna
        if self._is_window_open:
            if len(self._pln)==0:
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            # Needed for Matplotlib ver: 3.4.0 and 3.4.1 helps with capture closing event
            plt.draw() 
            plt.pause(1e-3)
            
  
    def close(self, event = None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all') # Needed for Matplotlib ver: 3.4.0 and 3.4.1
            print('Application closed!')

    def is_open(self):
        return self._is_window_open


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def parse_attr_nframes_frate(
        description,
        def_nframes,
        def_frate):
    # Parse all program attributes
    # description:   describes program
    # def_nframes:   default number of frames
    # def_frate:     default frame rate in Hz

    parser = argparse.ArgumentParser(
        description=description)

    parser.add_argument('-n', '--nframes', type=int,
                        default=def_nframes, help="number of frames, default "+str(def_nframes))
    parser.add_argument('-f', '--frate', type=int, default=def_frate,
                        help="frame rate in Hz, default "+str(def_frate))

    return parser.parse_args()


# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == '__main__':
    args = parse_attr_nframes_frate(
        '''Displays distance plot from Radar Data''',
        def_nframes = 50,
        def_frate = 20)

    with Avian.Device() as device:
        # activate all RX antennas
        num_rx_antennas = 1
        rx_mask = 1

        # metric = Avian.DeviceMetrics(
        #     sample_rate_Hz =           1000000,
        #     range_resolution_m =       0.05,
        #     max_range_m =              2.0,
        #     max_speed_m_s =            3,
        #     speed_resolution_m_s =     0.2,
        #     frame_repetition_time_s =  1/args.frate,
        #     center_frequency_Hz =      60750000000,
        #     rx_mask =                  rx_mask,
        #     tx_mask =                  1,
        #     tx_power_level =           31,
        #     if_gain_dB =               33)
        config = Avian.DeviceConfig(
            sample_rate_Hz=2e6,  # ADC sample rate of 2MHz
            rx_mask=1,  # RX antenna 1 activated
            tx_mask=1,  # TX antenna 1 activated
            tx_power_level=31,  # TX power level of 31
            if_gain_dB=33,  # 33dB if gain
            start_frequency_Hz=58e9,  # start frequency: 58.0 GHz
            end_frequency_Hz=63.5e9,  # end frequency: 63.5 GHz
            num_samples_per_chirp=256,  # 256 samples per chirp
            num_chirps_per_frame=1,  # 32 chirps per frame
            chirp_repetition_time_s=0.000150,  # Chirp repetition time (or pulse repetition time) of 150us
            frame_repetition_time_s=1 / args.frate,  # Frame repetition time default 0.005s (frame rate of 200Hz)
            mimo_mode="off")  # MIMO disabled

        config = config
        device.set_config(config)

        sample_rate_Hz = config.sample_rate_Hz
        rx_mask = config.rx_mask
        tx_mask = config.tx_mask
        tx_power_level = config.tx_power_level
        if_gain_dB = config.if_gain_dB
        start_frequency_Hz = config.start_frequency_Hz
        end_frequency_Hz = config.end_frequency_Hz
        num_samples_per_chirp = config.num_samples_per_chirp
        num_chirps_per_frame = config.num_chirps_per_frame
        chirp_repetition_time_s = config.chirp_repetition_time_s
        frame_repetition_time_s = config.frame_repetition_time_s
        # center_frequency_Hz = config.center_frequency_Hz

        print("Configuration \n")
        print(config)
        distance = DistanceFFT_Algo(config)
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        timestr = timestr + ".csv"
        # draw = Draw(config, config.max_range_m, num_rx_antennas)
        with open(timestr, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sample_rate_Hz, rx_mask, tx_mask, tx_power_level, if_gain_dB, start_frequency_Hz, end_frequency_Hz, num_samples_per_chirp, num_chirps_per_frame, chirp_repetition_time_s, frame_repetition_time_s])
            dist_peak_m_values = []  # List to store dist_peak_m values
            for frame_number in range(args.nframes): # For each frame
                # if not draw.is_open():
                #     break
                frame_data = device.get_next_frame()

                dist_data_all_antennas = []
                dist_peak_m_4_all_ant = []

                for i_ant in range(0, num_rx_antennas): #For each antenna
                    data = frame_data[i_ant]
                    flatten_frame_data = data .flatten()
                    writer.writerow(flatten_frame_data)
                    print("Jumlah Data : ", data.shape)
                    dist_peak_m, dist_data = distance.compute_distance(data)

                    dist_data_all_antennas.append(dist_data)
                    dist_peak_m_4_all_ant.append(dist_peak_m)
                    dist_peak_m_values.append(dist_peak_m)  # Append dist_peak_m to the list


                    print("Distance antenna # " + str(i_ant) + ": " +
                        format(dist_peak_m, "^05.3f") + "m")
                    print('Number sample per-chirp :', num_samples_per_chirp)
                    print('number chirp per frame : ', num_chirps_per_frame)
        #     draw.draw(dist_data_all_antennas)
    # Save dist_peak_m values to a text file
            dist_peak_m_filename = "dist_peak_m_values.txt"
            np.savetxt(dist_peak_m_filename, dist_peak_m_values, fmt="%0.5f")

        # draw.close()
    print(timestr)