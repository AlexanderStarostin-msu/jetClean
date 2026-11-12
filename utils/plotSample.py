#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# plotSample.py
#
# Purpose
# - Plot sampled OpenFOAM postProcessing tables (and optional reference tables)
#   for a selected x/y column pair.
#
# Expected layout (run from case directory)
# - Simulation data: ./postProcessing/<functionName>/<time>/<sampleSet>
# - Reference data:  ./reference/<functionName>/<time>/<sampleSet>
#
# Input arguments
# 1) sample set name, e.g. centralLine.xy
#    - Supports "functionName,sampleSet" format.
# 2) x column index (optionally suffixed with "s" to sort by x descending)
# 3) y column index (optionally suffixed with "s" to sort by y descending)
# 4) optional "save" flag to save figure instead of showing window.
#
# Requirements:
# - Reads latest available time directories and plots each time series.
# - Converts vector-valued text tuples "(a b c)" to magnitude before parsing.
# - Tries to load matching reference data and overlays it when available.
# - Performs a simple simulation-vs-reference consistency check and highlights
#   violations.
# - Applies scienceplots/LaTeX styling when available.
# - If profiles are shown on the plot (files with extensions xy,yz, or zy), 
#   the names of the time folders on the plots should be rounded to the nearest second and displayed with unit " s"
# - name plot after second column name, if it is not one of coordinates ("x", "y", "z"),
#   or "distance", or time ("t"),  otherwise it names after the first column
#
# Output
# - Interactive plot window by default, or saved figure when "save" is passed.
# - Console diagnostics about reference availability and comparison checks.
#
# Quick usage examples
# - python3 $FOAM_INST_DIR/BakerFoam/utils/plotSample.py centralLine.xy 1 6
# - python3 $FOAM_INST_DIR/BakerFoam/utils/plotSample.py sample,outputLine.xy 0 2 save
#
# If required wheels are missing:
# - pip install -r requirements.txt
#
# Always, keep requirements at the top of script
#
# -----------------------------------------------------------------------------

import numpy as np
from numpy import array
from numpy.linalg import norm
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import re
import math
import importlib
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl

# Check if scienceplots is installed
spec = importlib.util.find_spec('scienceplots')
if spec is not None:
    import scienceplots

    # Check if LaTeX is available
    if shutil.which("latex") is None:
        # No LaTeX → disable it
        mpl.rcParams['text.usetex'] = False
        plt.style.use(['science', {'text.usetex': False}])
    else:
        # LaTeX available → full scienceplots
        plt.style.use('science')

#format for latex if scienceplotwheel exists
def formatStringForLatex(str):

   latexString = str.replace('distance','x')

   spec = importlib.util.find_spec('scienceplots')
   if spec is None:
      return latexString

   latexString = latexString.replace('point_x','x')
   latexString = latexString.replace('point_y','y')
   latexString = latexString.replace('point_z','z')

   latexString = latexString.replace('_','\\_')
#guess greek letter
   latexString = latexString.replace('alpha','\\alpha ')
   latexString = latexString.replace('omega','\\omega ')
   latexString = latexString.replace('eps','\\varepsilon ')
   latexString = latexString.replace('lambda','\\lambda ')
   latexString = latexString.replace('Uf','U_f')
   latexString = latexString.replace('Us','U_s')
   latexString = latexString.replace('Time','t')

#guess math symbols
   latexString = latexString.replace('grad','\\nabla ')

#brackets for mean
   meanWordLBrack = 'areaNormalAverage('
   if(latexString.find(meanWordLBrack)>=0):
      latexString = latexString.replace(meanWordLBrack,'\\langle ')
      latexString = latexString.replace(')',' \\rangle')

   meanWordLBrack = 'volAverage('
   if(latexString.find(meanWordLBrack)>=0):
      latexString = latexString.replace(meanWordLBrack,'\\langle ')
      latexString = latexString.replace(')',' \\rangle')      

   latexString = "$" + latexString + "$"

   return latexString

def transform_string(s):
    # Pattern to match three floats inside round brackets
    pattern = r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[,\s]+" \
              r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[,\s]+" \
              r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"
    
    def replacer(match):
        nums = [float(match.group(i)) for i in range(1, 4)]
        result = math.sqrt(sum(x**2 for x in nums))
        return f"{result:.6f}"  # Replace with rounded result, no brackets

    return re.sub(pattern, replacer, s)


def format_time_folder_label(folder_name: str) -> str:
    try:
        tval = float(folder_name)

        # Check number of digits after decimal
        s = folder_name.strip()
        if "." in s:
            decimals = len(s.split(".", 1)[1])
        else:
            decimals = 0

        # Only format if >3 digits after the point
        if decimals > 3:
            tsec = int(math.floor(tval + 0.5))
            return f"{tsec} s"
        else:
            return f"{folder_name} s"

    except (ValueError, TypeError):
        return f"{folder_name} s"

def select_plot_name_column(x_name: str, y_name: str) -> str:
   """Use y-name unless it is coordinate/distance/time; then fallback to x-name."""

   def norm(name: str) -> str:
      return name.strip().lower()

   forbidden = {"x", "y", "z", "distance", "t"}
   y_norm = norm(y_name)

   return x_name if y_norm in forbidden else y_name

#read table
def readTxtTable(fileName):
   fulldata=[]
   colNames=[]
   
   if(not os.path.isfile(fileName)):
      return fulldata, colNames

   with open(fileName, "r") as f:
      linestoken = [line.strip() for line in f if line.strip()]

   cntHeader=1
   while linestoken[cntHeader][0]=="#":
      cntHeader+=1
   colNames = linestoken[cntHeader-1].split()[1:]

   fulldata=[]
   for x in linestoken[cntHeader:]:
      resulttoken=[]
      x=transform_string(x) # group 3 component to the absolute value
      xx=x.split()

      def safe_float(token):
         try:
            return float(token)
         except ValueError:
            return float('nan')

      resulttoken.append([ safe_float(token) for token in xx ] )
      fulldata.append(resulttoken[0])
   return fulldata, colNames

# ================ Executions starts ===============

sampleFuncName = "sample"
sampleSetName = "centralLine.xy"
complexName = [sampleSetName]

if len(sys.argv) > 1:
    sampleSetName = sys.argv[1]
    complexName = sampleSetName.split(',')
    if len(complexName) > 1:
        sampleFuncName = complexName[0]
        sampleSetName = complexName[1]

samplePath = os.getcwd() + "/postProcessing/"+sampleFuncName
print(samplePath)
refPath = os.getcwd() + "/reference/"+sampleFuncName
print(refPath)

xind=1
yind=4
xsort=False
ysort=False
savePlot=False
if len(sys.argv)>2:
   if(sys.argv[2][-1]=='s'):
      xsort=True
      sys.argv[2]=sys.argv[2][:-1]
   xind=int(sys.argv[2])
if len(sys.argv)>3:
   if(sys.argv[3][-1]=='s'):
      ysort=True
      sys.argv[3]=sys.argv[3][:-1]
   yind=int(sys.argv[3])
if len(sys.argv)>4:
   if(sys.argv[4].lower()=="save"):
      savePlot=True

#Put marker at the end of plot assumanbly parameterized with time
isLastMarker = xind>1 and yind>1 and len(complexName)>1

dataPath = Path(samplePath)
print("dataPath")
print(dataPath)
timeDirs=[f for f in dataPath.iterdir() if f.is_dir()]
timeDirs.sort(key=lambda x: os.path.getmtime(x))
timeDirs=reversed(timeDirs)

fig = plt.figure(figsize=(6,4.5))
 
token = []
flagRefData = False  # stays false if no reference found
cnt=0
for tDir in timeDirs:
   fullPath = (samplePath + "/" + tDir.name +"/" + sampleSetName)
   fullRefPath = (refPath + "/" + tDir.name +"/" + sampleSetName)
   timeLabel = format_time_folder_label(tDir.name)

   # Protect against missing sampled file
   if not os.path.isfile(fullPath):
      print(f"Skipping {tDir.name}: missing {sampleSetName}")
      continue

   fullData, colNames = readTxtTable(fullPath)
   if not fullData:
      print(f"Skipping {tDir.name}: empty or unreadable {sampleSetName}")
      continue

 #  print(fullPath)

   fullData, colNames = readTxtTable(fullPath)
   if fullData:
      cnt=cnt+1
#      print(fullData)

      xdata = [x[xind] for x in fullData[:]]
      ydata = [x[yind] for x in fullData[:]]

   fullRefData, colNamesRef = readTxtTable(fullRefPath)

   refDataExists = fullRefData and colNames[xind] in colNamesRef and colNames[yind] in colNamesRef
   if refDataExists:
      xindRef = colNamesRef.index(colNames[xind])
      yindRef = colNamesRef.index(colNames[yind])
#      print(fullRefData)
      flagRefData = True
      print(fullRefPath)
      xrefdata = [x[xindRef] for x in fullRefData[:]]
      yrefdata = [x[yindRef] for x in fullRefData[:]]
      plt.plot(xrefdata, yrefdata, marker='o', markersize=3, color='black', linewidth=0, label=timeLabel+" ref")

   if fullData:
      if xsort:
         xdata, ydata = zip(*sorted(zip(xdata, ydata),  reverse=True))
      elif ysort:
         ydata, xdata = zip(*sorted(zip(ydata, xdata),  reverse=True))
      plt.plot(xdata, ydata, label=timeLabel, linewidth=(1+2*(cnt==1)))
      if len(complexName)>1:
         plt.plot(xdata[-1], ydata[-1], marker='o', markersize=10, fillstyle='none', markeredgecolor='darkcyan' )
 
      plt.xlabel(formatStringForLatex(colNames[xind]))
      plt.ylabel(formatStringForLatex(colNames[yind]))
   #   print(colNames[xind])
   #   print(formatStringForLatex(colNames[xind]))

#Quantative comparison with reference data
   if refDataExists:
      if len(xdata)==len(xrefdata) and len(ydata)==len(yrefdata) :
         deltaData = array(ydata)-array(yrefdata) 
         deltaData = [x if x>0 else -x for x in deltaData]
         absTol = 0.05*array( max( deltaData, [1.0 for _ in yrefdata] ) )
         indViolated = [i for i,v in enumerate(deltaData>absTol) if v ]
         if len(indViolated):
            print("ERROR: Calculations! Simulation and reference differ by 1%.")
            xrefdata = [xrefdata[i] for i in indViolated]
            yrefdata = [yrefdata[i] for i in indViolated]
            plt.plot(xrefdata, yrefdata, marker='o', markersize=4, color='red', linewidth=0)
         else:
            print("Simulation and reference are close enough")
      else:
         print("ERROR: Data integrity! Simulation and reference have different sizes.")

if (not flagRefData):
   print("WARNING: The case has no reference data.")

print(colNames)
print("Saved in total "+str(len(colNames)))
plotTitle = os.getcwd().rsplit('/')[-1]
plt.title(formatStringForLatex(plotTitle))
plt.legend(loc="best")

#if changes in file format needed
#import matplotlib
#matplotlib.rcParams['savefig.format'] = 'png'

nameCol = select_plot_name_column(colNames[xind], colNames[yind])
nameCol = nameCol.replace("...", "")
safeNameCol = nameCol.replace(".", "_")
figFileName = (plotTitle + "_"+ sampleSetName.rsplit('.')[0] + "_"+ safeNameCol+ ".png")
fig.canvas.manager.set_window_title(formatStringForLatex(figFileName))

if (savePlot):
   plt.savefig(figFileName)
else:
   plt.show()

plt.close('all')

