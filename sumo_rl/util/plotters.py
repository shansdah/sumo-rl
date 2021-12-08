import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

def Average(lst):
    return sum(lst)/float(len(lst))

class PlotMetrics:

    def __init__(self, plot_source_path_ts, plot_source_path_veh, plot_destination_folder):
        self.plot_source_path_ts = plot_source_path_ts
        self.plot_source_path_veh = plot_source_path_veh
        self.plot_destination_folder = plot_destination_folder
        
    def plot_episodes(self, episodes, veh_columns, ts_columns):
        df_dict = []
        for i in range(episodes):
            df = pd.read_csv(self.plot_source_path_veh+"_"+str(i)+".csv", usecols=veh_columns, sep=';')
            df_dict.append({})
            for col, vals in df.iteritems():
                df_dict[i][col] = Average(vals)
        
        for i in veh_columns:
            plt_lst = []
            for j in range(episodes):
                plt_lst.append(df_dict[j][i])
            plt.figure()
            plt.plot(range(episodes), plt_lst)

            plt.xlabel('Episode')

            plt.ylabel(i)
            plt.savefig(self.plot_destination_folder+str(i))