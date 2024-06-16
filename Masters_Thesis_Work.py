
# -*- coding: utf-8 -*-
"""
Spyder Editor

Program reads a .pfc-file and extracts Force Curve data, calculating
various values and printing graphs from the data.
NOTE: Needs a program PointInPolygon.py to run properly.

TODO:   Optional: Limit measured values to improve graphs
        Optional: Optimization For Speed
        Add/Update Comments

@author: Niklas Valjakka
@version: 20230424

"""
# System imports

import struct
import collections
import time
import math
import numpy as np
import scipy as sp
from scipy import stats
import pylab as pl
import matplotlib.pyplot as plt
import os
import errno
from numpy.polynomial import polynomial

# Local source tree imports

from PointInPolygon import wn_PnPoly

# Constants

DEF_VALUE = 0.0  # Default value if errors encountered
# Data limit to differentiate two different versions (before 1.1.2019 and after 31.12.2018)
DATA_LIMIT = 2**10
CMAX_INDEX = 128
MAX_DISTANCE = 10**-9 # Used in functions to prevent dividing by zero
KEV_CONVERSION = 6.241509 / 1000 # Conversion factor from (nN*nm) to keV
PI_CONSTANT = math.pi
SIN_FUNC = math.sin
F_VALUES = 256 # Number of force data points in a Force Curve
BG_REMOVAL_DEGREES = [2,2] # Degrees for polyfit2d-function for background removal
ZLEVEL_START = 5 # The index of the point where the calculations for the z-level begin
ZLEVEL_END = 65 # The index of the point where the calculations for the z-level end
SLOPE_START_OFFSET = 4
SLOPE_END_OFFSET = 5
# Typical Force Fit Boundary values.
# Values taken from Dimension Icon User Guide
SLOPE_FIT_START = 0.90
SLOPE_FIT_END = 0.30
DISSIPATION_THRESHOLD = 25
# Radius of the AFM-tip in use (RTESPA525). Sourced from:
# https://www.brukerafmprobes.com/p-3915-rtespa-525.aspx
R_TIP = 8
HISTOGRAM_BINS = 50 # Number of bins in the histograms


def main():
    """ 
    The main program

    Returns
    -------
    None

    """
    # Starts a timer to test
    #start = time.time()
    
    # READING THE DATA FROM THE FILE
    
    # The user is asked for the name of a .pfc-file
    filename = input("File to read: ")
    # The file is located in system and the file path is returned
    filepath = find(filename,"..")
    filesize = os.path.getsize(filepath) # Tutkittavan tiedoston koko
    # Reads the header of the file, returns necessary info
    header_data = read_header(filepath)
    # Reads topography & Force Curve data
    topography_data_bin,fcurve_data_bin = read_height_and_fc_data(filepath,filesize,header_data[0])
    
    # Make the save folders, if necessary.
    savepath = filepath[:-4] + "_File/";
    make_folder(savepath)
    savepath_fc = savepath + 'Force_Curves/'
    savepath_me = savepath + 'FC_mean/'
    
    make_folder(savepath_fc)
    make_folder(savepath_me)
    
    # Change binary height data to float
    if (filesize/(header_data[0]**2) > DATA_LIMIT):
        topography_data_iter = struct.iter_unpack("i", topography_data_bin)
    else:
        topography_data_iter = struct.iter_unpack("h", topography_data_bin)
    z_mult = header_data[5] * header_data[7] # Multiplication factor for height from [raw data] to [nm]
    topography_data = z_mult * np.array(list(topography_data_iter)) # Unit: [nm]
    
    # Plot the topography data
    plot_data(topography_data,header_data[0],savepath=savepath,data_type="Topography")
    # Ask the user if background is to be calculated & removed
    char = input("Remove background? (Useful for flat surfaces) [Y/N]: ")
    if (char == 'Y' or char == 'y'):
        topography_data = remove_background(topography_data,header_data[0])
        # Plot topography minus background
        plot_data(topography_data,header_data[0],savepath=savepath,data_type="Topography")
    
    # Change binary Force Curve data to float
    if (filesize/(header_data[0]**2) > DATA_LIMIT):
        fcurve_data_iter = struct.iter_unpack("i", fcurve_data_bin)
    else:
        fcurve_data_iter = struct.iter_unpack("h", fcurve_data_bin)
    # Change values to nN
    force_mult = header_data[4] * header_data[8] * header_data[9] # Conversion factor from [raw data] to [nN]
    fcurve_data = force_mult * np.array(list(fcurve_data_iter)) # Unit: nN
    
    # Specify areas, if necessary
    
    char = input("Do you want to specify an area? [Y/N]: ")
    if (char == 'Y' or char == 'y'):
        # Set up area specifying loop
        areaint = 1;
        area_points = [ask_for_coordinates()];
        savepath_fc = savepath + 'Force_Curves/Area_' + str(areaint) + '/';
        make_folder(savepath_fc)
        while True:
            char = input("Do you want to specify another area? [Y/N]: ")
            if not (char == 'Y' or char == 'y'):
                # Done asking points -> Plot area, end loop
                plot_data(topography_data,header_data[0],area_points,savepath=savepath,data_type="Topography_Areas")
                break
            else:
                area_points.append(ask_for_coordinates())
                areaint += 1
                savepath_fc = savepath + 'Force_Curves/Area_' + str(areaint) + '/';
                make_folder(savepath_fc)
    else:
        area_points = None;
    
    # Initialize result arrays
    adh_arr = np.zeros(header_data[0]**2)
    defor_arr = np.zeros(header_data[0]**2)
    pfor_arr = np.zeros(header_data[0]**2)
    diss_arr = np.zeros(header_data[0]**2)
    slope_arr = np.zeros(header_data[0]**2)
    #dmt_arr = np.zeros(header_data[0]**2)
    
    # ITERATE THROUGH EVERY FORCE CURVE, PUT VALUES IN ARRAYS
    
    savepath_fc = savepath + 'Force_Curves/'
    for i in range(header_data[0]**2):
        # Process a force curve and get values
        value_arr = process_force_curve(i,fcurve_data[i*F_VALUES:(i+1)*F_VALUES],
                                        header_data,area_points,savepath_fc)
        # Put values into appropriate arrays
        adh_arr[i] = value_arr[0]; defor_arr[i] = value_arr[1];
        pfor_arr[i] = value_arr[2]; diss_arr[i] = value_arr[3];
        slope_arr[i] = value_arr[4]; 
        #dmt_arr[i] = value_arr[5];
    
    
    # PLOT DATA INTO HISTOGRAMS
    
    if area_points is None:
        # If no areas are specified
        savepath_hg = savepath + 'Histograms/'
        make_folder(savepath_hg)
        plot_histogram(adh_arr,header_data[0],area_points,savepath_hg,"Adhesion","nN")
        plot_histogram(defor_arr,header_data[0],area_points,savepath_hg,"Deformation","nm")
        plot_histogram(pfor_arr,header_data[0],area_points,savepath_hg,"Peak_Force","nN")
        plot_histogram(diss_arr,header_data[0],area_points,savepath_hg,"Dissipation","keV")
        plot_histogram(slope_arr,header_data[0],area_points,savepath_hg,"Slope","N/m")
        #plot_histogram(dmt_arr,header_data[0],area_points,savepath_hg,"Modulus")
    else:
        # For every area specified
        for i in range(len(area_points)):
            savepath_hg = savepath + 'Histograms/Area_' + str(i + 1) + '/'
            make_folder(savepath_hg)
            plot_histogram(adh_arr,header_data[0],area_points[i],savepath_hg,"Adhesion","nN")
            plot_histogram(defor_arr,header_data[0],area_points[i],savepath_hg,"Deformation","nm")
            plot_histogram(pfor_arr,header_data[0],area_points[i],savepath_hg,"Peak_Force","nN")
            plot_histogram(diss_arr,header_data[0],area_points[i],savepath_hg,"Dissipation","keV")
            plot_histogram(slope_arr,header_data[0],area_points[i],savepath_hg,"Slope","N/m")
            #plot_histogram(dmt_arr,header_data[0],area_points[i],savepath_hg,"Modulus")
    
    # MAP DATA TO IMAGES
    
    savepath_pl = savepath + 'Plots/'
    make_folder(savepath_pl)
    plot_data(adh_arr,header_data[0],area_points,savepath_pl,"Adhesion")
    plot_data(defor_arr,header_data[0],area_points,savepath_pl,"Deformation")
    plot_data(pfor_arr,header_data[0],area_points,savepath_pl,"Peak_Force")
    plot_data(diss_arr,header_data[0],area_points,savepath_pl,"Dissipation")
    plot_data(slope_arr,header_data[0],area_points,savepath_pl,"Slope")
    #plot_data(dmt_arr,header_data[0],area_points,savepath_pl,"Modulus")
    
    # Plot mean/median curves in the y-direction
    
    if area_points is None:
        savepath_ot = savepath + 'Other_Files/'
        make_folder(savepath_ot)
        plot_mean_values(adh_arr,header_data[0],area_points,savepath_ot,"Adhesion")
        plot_mean_values(defor_arr,header_data[0],area_points,savepath_ot,"Deformation")
        plot_mean_values(pfor_arr,header_data[0],area_points,savepath_ot,"Peak_Force")
        plot_mean_values(diss_arr,header_data[0],area_points,savepath_ot,"Dissipation")
        plot_mean_values(slope_arr,header_data[0],area_points,savepath_ot,"Slope")
        #plot_mean_values(dmt_arr,header_data[0],area_points,savepath_ot,"Modulus")
    else:
        for i in range(len(area_points)):
            savepath_ot = savepath + 'Other_Files/Area_' + str(i + 1) + '/'
            make_folder(savepath_ot)
            plot_mean_values(adh_arr,header_data[0],area_points[i],savepath_ot,"Adhesion")
            plot_mean_values(defor_arr,header_data[0],area_points[i],savepath_ot,"Deformation")
            plot_mean_values(pfor_arr,header_data[0],area_points[i],savepath_ot,"Peak_Force")
            plot_mean_values(diss_arr,header_data[0],area_points[i],savepath_ot,"Dissipation")
            plot_mean_values(slope_arr,header_data[0],area_points[i],savepath_ot,"Slope")
            #plot_mean_values(dmt_arr,header_data[0],area_points,savepath_ot,"Modulus")
    
    # PLOT DATA TO HEIGHT
    
    savepath_ot = savepath + 'Other_Files/'
    height_to_value(adh_arr,topography_data,savepath_ot,"Adhesion")
    height_to_value(defor_arr,topography_data,savepath_ot,"Deformation")
    height_to_value(pfor_arr,topography_data,savepath_ot,"Peak_Force")
    height_to_value(diss_arr,topography_data,savepath_ot,"Dissipation")
    height_to_value(slope_arr,topography_data,savepath_ot,"Slope")
    #height_to_value(dmt_arr,topography_data,savepath_ot,"Modulus")
    
    # Make a Force Curve from the mean of the values of FCs in the area
    if area_points is not None:
        for i in range(1,len(area_points)+1):
            savepath_fc = savepath + 'Force_Curves/Area_' + str(i) + '/'
            savepath_me = savepath + 'FC_mean/Area_' + str(i) + '/'
            make_folder(savepath_me)
            plot_mean_curve(savepath_fc,savepath_me,header_data[11],[header_data[12],header_data[13]])
    
    # Ends the timer
    #end = time.time()
    #print("Total time: %9.5f" % (end - start))
    
    
def find(name,path):
    """
    Returns a file path to the specified file

    Parameters
    ----------
    name : A name of the file that is searched.
    path : A filepath from which onward the file is searched.

    Returns
    -------
    string
        Filepath to the file.

    """
    
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
            
    # Raise Error if no file found
    if (len(result) == 0):
        raise NameError("Specified file could not be found!")
    # List multiple results, ask user to choose
    elif (len(result) > 1):
        print("Multiple files with same name found, please pick a correct file with a number");
        for i in range(len(result)):
            print("%i %s" % (i,result[i]))
        print("Please give the number of the file you want: ")
        time.sleep(1) # Separates print and input-commands to avoid possible errors
        file_num = int(input())
        return result[file_num]
    # Only one result found
    else:
        return result[0]
            
    
def make_folder(savepath):
    """
    Make a folder in the specified path

    Parameters
    ----------
    savepath : A path to where to make the folder.

    Returns
    -------
    None.

    """
    try:
        os.makedirs(savepath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            

def read_header(file_path):
    """
    Read important data from header

    Information about file format:
        http://nanophys.kth.se/nanophys/facilities/nfl/afm/icon/bruker-help/Content/Cover%20page.htm

    Parameters
    ----------
    file_path : The path to the file which is read.

    Returns
    -------
    An array containing the necessary data, in order:
        [Data resolution,
         Sync Distance QNM,
         Peak Force Amplitude,
         Frequency,
         Sens. DeflSens,
         ZSensSens,
         Soft HarmoniX Setpoint,
         Z Scale Z (Height scaling factor),
         Spring Constant,
         Z Scale Defl. (Z Deflection scaling factor),
         Peak Force Setpoint (Not in header, calculated),
         Deformation Fit Region
         Maximum Force Fit Boundary,
         Minimum Force Fit Boundary].

    """

    # Open specified file in read-mode
    with open(file_path, 'rb') as f:
        header_data = [] # Initialize array of header rows
        char_num = 0 # Initialize character size
        # Iterate through every line in a header
        while True:
            # Read next line
            # (split-command removes linebreak at the end of the row)
            line = f.readline().split(b'\r\n')[0]
            # End loop if line begins with byte "x1a"
            # (Special character that specifies the end of the header)
            if line.startswith(b'\x1a'):
                break
            # Otherwise add row to the header_data -array, and update character size
            else:
                header_data.append(line)
                char_num = char_num+len(line)+2 # +2 comes from the removed linebreak
        print("Number of characters: " + str(char_num)) # Print the number of characters
        # Get the relevant values from the header
        for i in range(len(header_data)):
            linestr = str(header_data[i])
            # print(str(i) + " " + linestr[3:-1])
            # Print rows by number. Unnecessary "b'\" ja "'" are removed
            # from the beginning and the end of the row,
            # so that rows would match ones from UNIX more-command.
            
            # Data resolution value
            if (linestr[3:-1].find("\Lines") > -1):
                resolution = int(get_value_from_header(linestr[3:-1]))
                print("Resolution = %3.1i" % (resolution))
                
            # Sync Distance value
            elif (linestr[3:-1].find("\Sync Distance QNM:") > -1):
                sdqnm = get_value_from_header(linestr[3:-1])
                print("Sync Distance QNM = %i" % (sdqnm))
                
            # Peak Force Amplitude value
            elif (linestr[3:-1].find("\Peak Force Amplitude:") > -1):
                pfa = get_value_from_header(linestr[3:-1])
                print("Peak Force Amplitude = %4.1f" % (pfa))
                
            # PFT Frequency value
            elif (linestr[3:-1].find("\PFT Freq:") > -1):
                freq = 1000 * get_value_from_header(linestr[3:-1])
                print("Frequency = %5.1f" % (freq))
                
            # Deflection Sens. value
            elif (linestr[3:-1].find("\@Sens. DeflSens") > -1):
                deflsens = get_value_from_header(linestr[3:-1])
                print("Sens. DeflSens = %3.1f" % (deflsens))
                
            # Z sens. value
            elif (linestr[3:-1].find("\@Sens. ZsensSens") > -1):
                zsens = get_value_from_header(linestr[3:-1])
                print("ZSensSens = %7.4f" % (zsens))
                
            # Soft HarmoniX Setpoint value
            elif (linestr[3:-1].find("\@2:SoftHarmoniXSetpoint:") > -1):
                shxsp = get_value_from_header(linestr[3:-1])
                print("SHXSP = %8.7f" % (shxsp))
                
            # Z Scale value
            elif (linestr[3:-1].find("\@2:Z scale:") > -1):
                istart = linestr.find('(') + 1
                iend = linestr.find(')')
                zscalez = get_value_from_header(linestr[istart:iend])
                print("ZScaleZ = %13.12f" % (zscalez))
                
            # Spring Constant value
            elif (linestr[3:-1].find("\Spring Constant:") > -1):
                k = get_value_from_header(linestr[3:-1])
                print("Spring Constant = %4.1f" % (k))
                
            # Deformation Fit Region value
            elif (linestr[3:-1].find("\Deformation Fit Region:") > -1):
                deformation_fit = get_value_from_header(linestr[3:-1])
                print("Deformation Fit Region = %3.2f" % (deformation_fit))
                
            # Z scale value
            elif (linestr[3:-1].find("\@4:Z scale:") > -1):
                istart = linestr.find('(') + 1
                iend = linestr.find(')')
                zscaledefl = get_value_from_header(linestr[istart:iend])
                print("ZScaleDefl = %13.11f" % (zscaledefl))
                
            # Maximum Force Fit Boundary Value
            elif (linestr[3:-1].find("\Max Force Fit Boundary:") > -1):
                FFB_max = get_value_from_header(linestr[3:-1]) / 100.0
                print("Maximum Force Fit Boundary = %3.2f" % (FFB_max))
                
            # Minimum Force Fit Boundary Value
            elif (linestr[3:-1].find("\Min Force Fit Boundary:") > -1):
                FFB_min = get_value_from_header(linestr[3:-1]) / 100.0
                print("Minimum Force Fit Boundary = %3.2f" % (FFB_min))
                
                
    # Calculate the Peak Force Setpoint (unit: [nN])
    pfs = shxsp * k * deflsens
    print("Peak Force Setpoint = %5.1f" % (pfs))
    # Gather the data into a single array
    header_info = [resolution,sdqnm,pfa,freq,deflsens,zsens,shxsp,zscalez,k,
                   zscaledefl,pfs,deformation_fit,FFB_max,FFB_min]
    return header_info


def get_value_from_header(header_string,default=DEF_VALUE):
    """
    Read a value from a header line

    Parameters
    ----------
    header_string : A string line to read.
    default: A value given in a case no data is read

    Returns
    -------
    float
        Value from header string.

    """
    # Define a boolean
    got_value = False
    # Split to substrings, go over them all
    for substr in header_string.split():
        try:
            # Try to get a value from substring
            retval = float(substr)
            got_value = True
        except ValueError:
            # No luck, next substring
            pass
    if got_value:
        # Program succeessful, return value
        return retval
    else:
        # Program failure, return default
        print("Error! Data could not be read! Return default value %13.9f" % (default))
        return default
    

def read_height_and_fc_data(file_path,file_size,resolution):
    """
    Read height and Force Curve data from a .pfc -file

    Parameters
    ----------
    file_path : A path to a .pfc file.
    file_size : The size of the .pcf file
    resolution : A resolution of the data.

    Returns
    -------
    height_data: Topography data in a binary array.
    fc_data: Force Curve data in a binary array.

    """
    
    # Check for the .pfc-file version and choose imagesize & datasize accordingly
    if (file_size/(resolution**2) > DATA_LIMIT):
        imagesize = (resolution**2) * 4 # Size of height data in bytes
        datasize = (resolution**2) * 2 * (2**7) * 4 # Size of force curve data in bytes
    else:
        imagesize = (resolution**2) * 2 # Size of height data in bytes
        datasize = (resolution**2) * 2 * (2**7) * 2 # Size of force curve data in bytes
    # Get height and force curve data using values above
    with open(file_path, 'rb') as f:
        f.seek(file_size - datasize - imagesize) # Skip to the start of FC data
        height_data = np.fromfile(f,'B',imagesize) # Read Height Data
        fc_data = np.fromfile(f,'B',datasize) # Read FC Data
        
    return height_data,fc_data



def plot_data(data,resolution,areas=None,savepath=None,data_type=None):
    """
    Plot data in a square (resolution**2) area

    Parameters
    ----------
    data : Data to be plotted. (NOTE: Has to be an array of length resolution**2)
    resolution : Side of a square plot area.
    area : Area to be focused on
    savepath : Path to the save file
    data_type : The type of data to be shown on the title

    Returns
    -------
    None.

    """
    # Reshape data to match the area
    data_array = data.reshape(resolution,resolution)
    # Plot the figure
    fig = plt.figure()
    ax = plt.gca()
    
    # Plot the area, if any given
    if areas is not None:
        for area in areas:
            for i in range(len(area)-1):
                xplot = [area[i].x,area[i+1].x]
                yplot = [area[i].y,area[i+1].y]
                ax.plot(xplot,yplot,'-r')
    
    imgplot = ax.imshow(data_array, origin='lower')
    fig.colorbar(imgplot, ax=ax)
    imgplot.set_cmap('nipy_spectral')
    
    # Speficy title and axis labels
    if data_type is not None:
        plt.title('2D map for ' + data_type)
    else:
        plt.title('2D map')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    
    # Save figure if save path given
    if savepath is not None:
        savefile = savepath + data_type + '_plot.png'
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
        
    # Show figure
    plt.show()
    
    
def ask_for_coordinates():
    """
    Ask user for coordinate points. Returns given points

    Returns
    -------
    Point array
        An array containing area edge points.
    
    """
    
    # Asks for the first coordinate
    print("Please give the first coordinate in form ""x y"", where x,y stand for pixels: ")
    time.sleep(1) # Separates print and input-commands to avoid possible errors
    first_point = [int(x) for x in input().split()]
    area = [Point(first_point[0],first_point[1])]
    while True:
        # Keep asking points until first one is given again
        print("Please give the next coordinate, or retype the first coordinate to close area: ")
        time.sleep(1) # Separates print and input-commands to avoid possible errors
        next_point = [int(x) for x in input().split()]
        area.append(Point(next_point[0],next_point[1]))
        if (next_point[0] == first_point[0] and next_point[1] == first_point[1]):
            # First point given again -> Return Points
            return area
    
    
def remove_background(data,resolution):
    """
    Remove background from an image by a polynomial fit

    Parameters
    ----------
    data : Data points from which background
    resolution : Resolution of data.

    Returns
    -------
    data_modified : Data with background removed.

    """
    xFit = np.linspace(0, 500, resolution)
    yFit = np.linspace(0, 500, resolution)
    zFit = [[data[j*resolution+i] for i in range(resolution)] for j in range(resolution)]
    X = np.array(np.meshgrid(xFit,yFit))
    c = polyfit2d(X[0], X[1], zFit, BG_REMOVAL_DEGREES)
    f = polynomial.polyval2d(X[0], X[1], c)
    data_modified = np.array([(zFit[j][i]-f[j][i]) for j in range(len(f)) for i in range(len(f[0]))])
    #f = np.resize(f,(1,len(dataImMod)))
    return data_modified


def polyfit2d(x, y, f, deg):
    """
    Program fits a polynomial to the data

    Parameters
    ----------
    x : X-coordinates of the data points.
    y : Y-coordinates of the data points.
    f : Data that polynomial is fitted to.
    deg : The degrees of the fitting polynomial in a vector [xdeg, ydeg].

    Returns
    -------
    TYPE
        A polynomial fit to the data.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1,vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f, rcond=-1)[0]
    return c.reshape(deg+1)

    
def process_force_curve(i,data,header_data,areas,savepath):
    """
    Process raw floating point data into a force curve

    Parameters
    ----------
    i : The pixel in question.
    data : Force data on pixel i.
    header_data : Array containing useful array data of the file.
    area : An area from which curve curves are pulled from.
    savepath : The path to the save file.

    Returns
    -------
    values : Useful values derived from force curves.

    """
    # Begin by correcting the force values and creating distance values
    # Data is currently disorganized and contains "trash bytes".
    # Data needs to be properly organized
    force_arr,dist_arr = create_curve_points(data,CMAX_INDEX - 1 - int(round(header_data[1])),header_data)
    ffb = [header_data[12],header_data[13]]
    values = get_values(force_arr,dist_arr,header_data[11],ffb)
    
    # Plot the force curves in the specified area(s).
    if areas is not None:
        for j in range(len(areas)):
            x,y = pixel2point(i,header_data[0]) # Transform index i to pixel coordinates [x,y]
            if (wn_PnPoly(Point(x,y),areas[j],len(areas[j])-1) != 0):
                savepath_full = savepath + 'Area_' + str(j + 1) + '/';
                save_fc_data(force_arr,dist_arr,[x,y],savepath_full)
    return values
    

def pixel2point(pixel,res):
    """
    Transform a pixel placement into a placement on a 2D map

    Parameters
    ----------
    pixel : A pixel value.
    res : Measurement resolution.

    Returns
    -------
    x : x-position of the pixel point.
    y : y-position of the pixel point.

    """
    x = int(pixel%res)
    y = int(round((pixel-x)/res))
    return x,y
    

def create_curve_points(data,cutoff_point,header_data):
    """
    Create force curve points from data.
    Rearranges force data in a properly order.
    (Required as the raw data in a file is not organized as-is)

    Parameters
    ----------
    data : Force values of a force curve.
    cutoff_point : Index of the point where the value is cut-off.
    header_data : Data from header

    Returns
    -------
    y : A force data organized in a way that it presesnts a force curve.
    x : A distance values to match the force values

    """
    # Rearrange force values
    # NOTE: data[127] is not a valid data point. 
    # Instead, data[256-cutoff_point-2], which corresponds 
    # to the beginning of the curve, is used twice instead
    force_arr = np.array([(data[i]) for i in range(256-cutoff_point-2,256,1)])
    force_arr = np.append(force_arr,[(data[i]) for i in range(126,-1,-1)])
    force_arr = np.append(force_arr,[(data[i]) for i in range(128,256-cutoff_point-1,1)])
    
    # Define the time steps
    t = np.linspace(0,1.0/header_data[3],len(force_arr))
    # Transform x-axis from z-piezo distance
    x_0 = [header_data[2]*(1-SIN_FUNC(2*PI_CONSTANT*header_data[3]*(t[i])-(PI_CONSTANT/2))) for i in range(len(t))]
    
    # Correct force values in data so that "tail" is near zero
    zerolevel = np.median(force_arr[ZLEVEL_START:ZLEVEL_END])
    y = force_arr - zerolevel;
    defl_max = header_data[10]/header_data[8]
    # Correct z-piezo distance to true distance by taking account cantilever movement
    x = [i-defl_max+(j/header_data[8]) for (i,j) in zip(x_0,y)]
    
    return y,x

    
def get_values(force_arr,dist_arr,deformation_fit,ffb):
    """
    Get values from force curves

    Parameters
    ----------
    force_arr : Force values of the force curve points.
    dist_arr : Distance values of the force curve points.
    deformation_fit : Deformation Fit Region value (from header)
    ffb : Force fit boundary values

    Returns
    -------
    list
        A list of values calculated from the force curve data.

    """
    
    global x_ind
    global adhes_val
    
    # Calculate peak force value (Unit: [nN])
    peakforce_val = max(force_arr)
    peakforce_val_index = np.where(force_arr == peakforce_val)[0][0]
    
    # Calculate adhesion value (Unit: [nN])
    adhes_val = min(force_arr)
    adhes_val_index = np.where(force_arr == adhes_val)[0][0]
    adhes_val_abs = abs(adhes_val)
    
    # Calculate deformation value (Unit: [nm])
    
    # Calculate first an offset value on distance (Based on Deformation Fit -value)
    defor_val = calc_deformation(force_arr,dist_arr,deformation_fit,peakforce_val)
    
    # Calculate dissipation value (Unit: [keV])
    diss_val = calc_dissipation(force_arr,dist_arr)
    
    # Calculate slope value (Unit: [N/m])
    slope_val = calc_slope(force_arr,dist_arr,peakforce_val_index,adhes_val_index,ffb)
    
    #dmt_val = calc_dmt(force_arr,dist_arr,adhes_val_index)
    
    #return [adhes_val_abs,defor_val,peakforce_val,diss_val,slope_val,dmt_val]
    return [adhes_val_abs,defor_val,peakforce_val,diss_val,slope_val]


def calc_deformation(y,x,def_fit,pf_val,default=DEF_VALUE):
    """
    Calculate deformation value of the force curve.
    (Deformation defined as the distance from the 
     Deformation Force Level position to the peak interaction force position)
    
    Calculated value is not the full deformation value, but rather 
    the Deformation Fit Region value is used.
    Same value is used by Bruker to reduce baseline noise.
    
    Unit: [nm]

    Parameters
    ----------
    y : Force values of force curve points.
    x : Distance values of force curve points.
    def_fit: Deformation Fit Region value
    pf_val : Value of Peak Force
    default : The default value returned in case of error.

    Returns
    -------
    float
        The deformation value of the force curve.

    """
    # Initialize variables to control while-loop
    points_above = True; find_value = True;
    #i = np.where(y == pf_val)[0][0]
    i = np.where(x == min(x))[0][0]
    j = i
    cut_off = (1.00 - def_fit) * pf_val
    while points_above:
        if j < 0:
            # An end was reached before a suitable index j was found!
            find_value = False
            points_above = False
        elif (y[j] < cut_off and y[j] > y[j-1]):
            # A Suitable index j was found: Loop can be ended
            # Calculate Deformation using secant method
            points_above = False
            y_modified = y - ((1.00 - def_fit) * pf_val)
            # In case y-values are too close
            if (abs(y_modified[j]-y_modified[j+1]) < MAX_DISTANCE):
                x_end = (x[j]+x[j+1])/2.0;
            else:
                x_end = x[j] - y_modified[j] * (x[j] - x[j+1])/(y_modified[j] - y_modified[j+1])
        else:
            # Step to the next index j
            j = j-1
    if not find_value:
        # Value not Found -> Error, return DEFAULT
        return default
    else:
        # Return deformation value
        return x_end - x[i]
        

def calc_dissipation(y,x):
    """
    Calculate the dissipation value of the curve, defined as the 
    area between approach and retract curves. 
    
    Unit: [keV]

    Parameters
    ----------
    y : Force values of force curve points.
    x : Distance values of force curve points.

    Returns
    -------
    Float
        The dissipation value of the force curve.

    """
    # Set a start index and index range for the analysis
    start_index = np.where(x == min(x))[0][0]
    
    index_range = min(255-start_index,start_index-1)
    ret_val_nm = num_integral(y,x,start_index,index_range)
    ret_val = KEV_CONVERSION * ret_val_nm
    return max(0,ret_val)
    
    
def num_integral(y,x,i_start,i_range,default=DEF_VALUE):
    """
    Calculate the area under the FC Extend and Retract -curves numerically

    Parameters
    ----------
    y : y-coordinates of the FC points.
    x : x-coordinates of the FC points.
    i_start : Start index of integral.
    i_end : End index of integral.

    Returns
    -------
    area : The area under the FC.

    """
    # Set index points, set total area to 0.0
    j0 = i_start; j1 = i_start - 1; j2 = i_start + 1; area = 0.0;
    while True:
        # Test whether the en index has been reached for either of the subindeces
        if ( (j2 - i_start) == i_range or (i_start - j1) == i_range):
            return default
        elif (y[j1] < y[j2] and (i_start - j1) >= DISSIPATION_THRESHOLD and (j2 - i_start) >= DISSIPATION_THRESHOLD):
            # End reached: Break loop
            break
        else:
            # End reached: Break loop
            # End not reached: Calculate Tri. A and add to total
            area = area + calc_triangle([x[j0],y[j0]],[x[j1],y[j1]],[x[j2],y[j2]])
            # Move the indeces to the closest point in either direction
            j_apu = min(x[j1-1],x[j2+1])
            if j_apu == x[j1-1]:
                j0 = j1
                j1 = j1 - 1
            else:
                j0 = j2
                j2 = j2 + 1
    # Return integral or total area under the
    return area
    

def calc_triangle(x0,x1,x2):
    """
    Calculate the area of a triangle using its vertices

    Parameters
    ----------
    x0 : First vertice of a triangle
    x1 : Second vertice of a triangle
    x2 : Third vertice of a triangle

    Returns
    -------
    TYPE
        Area of a triangle.

    """
    return abs((1/2) * ( x0[0]*(x1[1]-x2[1]) + x1[0]*(x2[1]-x0[1]) + x2[0]*(x0[1]-x1[1]) ) )
    
    
def calc_slope(y,x,start_index,end_index,ffb,default=DEF_VALUE):
    """
    Calculate the slope of a force curve

    Unit : [N/m]

    Parameters
    ----------
    y : Force values of the force curve.
    x : Distance values of the force curve.
    start_index : The index of the maximum of the force curve.
    end_index : The index of the minimum of the force curve.
    ffb: Force Fit Boundary values
    default : The default value returned in case of error.

    Returns
    -------
    float
        The value of the slope.

    """
    # Set maximum and minimum force values
    y_pf = y[start_index]; y_adh = y[end_index]
    
    # Set maximum and minimum force values for slope value analysis
    #y_max = calc_FFB(ffb[0],y_adh,y_pf)
    #y_min = calc_FFB(ffb[1],y_adh,y_pf)
    y_max = calc_FFB(SLOPE_FIT_START,y_adh,y_pf)
    y_min = calc_FFB(SLOPE_FIT_END,y_adh,y_pf)
    
    # Calculate start index of the slope scan
    # (PeakForceValue * DeformationFitRegion (value from header))
    i = start_index
    while True:
        if (i == 256):
            x0 = start_index
            break
        if (y[i] < y_max):
            x0 = i
            break
        else:
            i = i + 1

    # Calculate end index of slope scan
    # (AdhesionValue * SLOPE_FIT_END (custom value))
    i = end_index
    while True:
        if (y[i] > y_min):
            xn = i + 1
            break
        else:
            i = i - 1
    
    # Fit a slope to the points [x[x0:xn],y[x0:xn]]
    if (xn - x0 < 3):
        # Not enough points for an analysis -> Error, return default
        return default
    else:
        y_slope = y[x0:xn]
        x_slope = x[x0:xn]
        # Fit a 1-degree polynomial to the points
        slope,y0 = np.polyfit(x_slope,y_slope,1)
        if (slope > 0):
            # Slope should be negative, positive result -> Error, return default
            return default
        else:
            return slope
        
    
def calc_FFB(x,F_Adh,F_PF):
    """
    Calculates Force Fit Boundary value for multiplier x

    Parameters
    ----------
    x : A multiplier, 0 <= x <= 1
    F_Adh : Adhesion Force value
    F_PF : Peak Force value

    Returns
    -------
    Force Fit Boundary for multiplier x.

    """
    return F_Adh + (F_PF - F_Adh) * x
    

def calc_dmt(y,x,indMax,default=DEF_VALUE):
    """
    
    Calculate the young modulus value in GPa using DMT model
    
    (NOTE: NOT IN USE: WILL SIGNIFICANTLY SLOW DOWN THE PROGRAM & NEEDS PRESET VALUES
     TO WORK!)

    Parameters
    ----------
    y : Force values of the force curve.
    x : Distance values of the force curve.
    indMax : The index of the minimum point of the force curve
    default : Default return value, if program encounters an error.

    Returns
    -------
    float
        Value of calculated Young's Modulus.
        
    """
    try:
        Arr = sp.optimize.curve_fit(myfunc,
                                    x[CMAX_INDEX+SLOPE_START_OFFSET:indMax-SLOPE_END_OFFSET],
                                    y[CMAX_INDEX+SLOPE_START_OFFSET:indMax-SLOPE_END_OFFSET],
                                    p0=[100.0],bounds=([1.0],[300.0]))
        
    except ValueError:
        #print("There's something wrong with the values! Check the program!")
        return default
    
    except RuntimeError:
        #print("Optimal parameters not found! Check the program!")
        return default
    
    return Arr[0][0]
    

def myfunc(x,e):
    """Funktio DMT-fittauksen tekoon
        (Lähde: https://pubs.acs.org/doi/10.1021/la302706b)
    
    Parametrit:
    x -- Muuttujan arvo
    e -- (Redusoidun) Youngin Moduluksen arvo
    
    Palautus:
    Funktion arvo pisteessä x
    """
    
    return (4/3) * e * np.sqrt(R_TIP*(abs(x_ind-x)**3)) - adhes_val


def save_fc_data(y_arr,x_arr,point=None,savepath=None):
    """
    Save Force Curve data in a text file, ASCII format

    Parameters
    ----------
    y_arr : The y-coordinate values of the points
    x_arr : The x-coordinate values of the points.
    point : The location of the data point in pixel units.
    savepath : The path to the save file.

    Returns
    -------
    None.

    """
    
    if point is not None:
        file_name = savepath + 'Data_y_' + str(point[1]) + '_x_' + str(point[0]) + '.txt'
    else:
        file_name = savepath + '_Value_Curve.txt'
    with open(file_name,'w') as text_file:
        text_file.write("  i nm            nN \n")
        for i in range(len(x_arr)):
            text_file.write("%3i %12.9f %8.5f\n" % (i, x_arr[i], y_arr[i]))
            

def plot_histogram(data,resolution,area,savepath,data_type,data_unit):
    """
    Plot the calculated values in a histogram.

    Parameters
    ----------
    data : Data to be presented.
    resolution : The resolution of the data.
    area : The area given by user, None if none given.
    savepath : Path to the save folder.
    data_type : The type of data in a string.
    data_unit : The unit of the data in a string

    Returns
    -------
    None.

    """
    # Remove data points outside the area
    if area is not None:
        data = [data[i] for i in range(resolution**2) 
                if (wn_PnPoly(Point(pixel2point(i,
                        resolution)[0],pixel2point(i,resolution)[1]),area,len(area)-1) != 0)]
    
    data_sorted = sorted(data)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Fit a gauss cirve to the data
    fit = stats.norm.pdf(data_sorted, np.mean(data_sorted), np.std(data_sorted))
    # Plot gauss curve and make a histogram
    pl.plot(data_sorted,fit,'-o')
    pl.hist(data_sorted,density=True,bins=HISTOGRAM_BINS)
    # Convert values to a string
    str_median = "Median: " + str(np.median(data_sorted))
    str_mean = "Mean: " + str(np.mean(data_sorted))
    str_sd = "SD: " + str(np.std(data_sorted))
    iqr = np.percentile(data_sorted, 75) - np.percentile(data_sorted, 25)
    str_iqr = "IQR: " + str(iqr)
    str_count = "Count: " + str(len(data_sorted))
    # Write mean values to a corner
    ax.text(0.99, 0.99, str_median, horizontalalignment='right',
            verticalalignment='top', transform=ax.transAxes)
    ax.text(0.99, 0.92, str_mean, horizontalalignment='right',
            verticalalignment='top', transform=ax.transAxes)
    ax.text(0.99, 0.85, str_sd, horizontalalignment='right',
            verticalalignment='top', transform=ax.transAxes)
    ax.text(0.99, 0.78, str_iqr, horizontalalignment='right',
            verticalalignment='top', transform=ax.transAxes)
    ax.text(0.99, 0.71, str_count, horizontalalignment='right',
            verticalalignment='top', transform=ax.transAxes)
    plt.xlabel(data_type + " [" + data_unit + "]")
    pl.title("Histogram for " + data_type)
    savefile = savepath + 'Histogram_' + data_type + '.png'
    pl.savefig(savefile, dpi=300, bbox_inches='tight')
    pl.show()
    # Write histogram data to a text file
    datafile = savepath + 'Histogram_data_' + data_type + '.txt'
    with open(datafile,"w") as text_file:
        for i in range(len(data_sorted)):
            text_file.write("%13.9f \n" % (data_sorted[i]))
        

def plot_mean_values(data,resolution,area,savepath,data_type):
    """
    Plot the mean and median values for every y-pixel in the area.

    Parameters
    ----------
    data : Array of values.
    resolution : Data resolution.
    area : Points of a polygon.
    savepath : Path to the save file.
    data_type : Value type.

    Returns
    -------
    None.

    """
    # Reshape data array into map
    data_array = data.reshape(resolution,resolution)
    # Initialize arrays
    x = []; data_means = []; data_medians = [];
    # If area is defined
    if area is not None:
        for i in range(resolution):
            # Gather values inside area
            row_insides = [data_array[i][j]
                           for j in range(resolution)
                           if (wn_PnPoly(Point(j,i),area,len(area)-1) != 0)]
            # Add a value from row_insides if not NULL
            if row_insides:
                x.append(i)
                # Calculate the mean and median, add to array
                row_mean = np.mean(row_insides)
                row_median = np.median(row_insides)
                data_means.append(row_mean);
                data_medians.append(row_median)
    # If area is not defined
    else:
        for i in range(resolution):
            row_insides = [data_array[i][j]
                           for j in range(resolution)]
            x.append(i)
            row_mean = np.mean(row_insides)
            row_median = np.median(row_insides)
            data_means.append(row_mean);
            data_medians.append(row_median)
    
    # Create a figure
    plt.figure()
    plt.plot(x,data_means,'or',x,data_medians,'ob')
    plt.xlabel("Y position")
    plt.ylabel(data_type)
    pl.title("Curves for mean (red) and median (blue)")
    savefile = savepath + 'Mean_Graphs_For_' + data_type + '.png'
    pl.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def height_to_value(y,x,path,data_type):
    """
    Plots value vs. height graph

    Parameters
    ----------
    y : Force values of the force curve.
    x : Distance values of the force curve.
    path : Path to the save file.
    data_type : Value type.

    Returns
    -------
    None.

    """
    plt.figure()
    plt.plot(x,y,'or')
    plt.xlabel("Height")
    plt.ylabel(data_type)
    pl.title("Value vs. height data for " + data_type)
    savefile = path + data_type + '_Height.png'
    pl.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_mean_curve(curves_path,path,deformation_fit,ffb):
    """
    Plot a mean and median curves of every FC in an area.

    Parameters
    ----------
    curves_path : Path to the file with the force curves.
    path : Path to save file.
    deformation_fit : fitting area of deformation, needed to calculate values 
                        in save_values 
    ffb: Force Fit Boundary values
    
    Returns
    -------
    None.

    """
    
    # Get every force curve file
    folder_files = [f for f in os.listdir(curves_path) if f[-3:] == 'txt']
    # Initialize useful variables
    N = len(folder_files); M = 256; i= 0;
    # Initialize arrays to gather values
    xs = np.zeros([N,M]); ys = np.zeros([N,M])
    for file in folder_files:
        with open(curves_path + file, 'r') as f:
            lines = f.readlines()
            x0 = []; y0 = [];
            for x in lines[1:]:
                x0.append(float(x.split()[1])); # The x-values of a force curve
                y0.append(float(x.split()[2])); # The y-values of a force curve
            # Add the x- and y-positions of the curve to the total array
            xs[i] = x0; ys[i] = y0; i += 1;
                
    # Create mean/median force curves from data
    x1 = np.mean(xs, axis = 0); y1 = np.mean(ys, axis = 0)
    x2 = np.median(xs, axis = 0); y2 = np.median(ys, axis = 0)
    save_fc_data(y1,x1,savepath=path+"Mean");
    save_fc_data(y2,x2,savepath=path+"Median");
    save_values(y2,x2,deformation_fit,ffb,path)
    
    
def save_values(y,x,deformation_fit,ffb,savepath):
    """
    Save median force curve values to a file

    Parameters
    ----------
    y : Force values of the force curve.
    x : Distance values of the force curve.
    deformation_fit : fitting area of deformation
    ffb: Force Fit Boundary values
    savepath : Path to the save file.

    Returns
    -------
    None.
    
    """
    # Get values of a force curve
    values = get_values(y,x,deformation_fit,ffb)
    strings = ["Adhesion","Deformation","Peak Force","Dissipation","Slope"]
    # Write values to a file
    datafile = savepath + 'Median_FC_Data.txt'
    with open(datafile,"w") as text_file:
        for i in range(5):
            text_file.write("%s %13.9f\n" % (strings[i], values[i]))


if __name__ == "__main__":
    """ 
    Starts up the program
    """
    # Specify a Point, which is needed to use PointInPolygon.py
    Point = collections.namedtuple('Point', ['x','y'])
    # Start the main program
    main()