#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
print('Basic imports successful')
from src.frequency_analyzer import FrequencyAnalyzer
print('FrequencyAnalyzer import successful')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
print('All imports successful')
