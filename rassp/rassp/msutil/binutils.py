# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Mass-spectrometry-specific helper utilities for binning, formula/mass operations, and evaluation helpers.

"""
Generic utilities for binning spectra with exact precision. 

"""
import numpy as np
import importlib

# Try to load the compiled fast helper; if unavailable (different Python ABI
# or extension not built in this environment), fall back to pure-Python code.
fast = None
for module_name in ('rassp.msutil.binutils_fast', 'rassp.binutils_fast'):
    try:
        fast = importlib.import_module(module_name)
        break
    except Exception:
        fast = None

# Class overview: SpectrumBins encapsulates a reusable component in this module.
class SpectrumBins:
    """
    A class that contains configuration about how we are binning
    our spectrum. This should be the canonical source of binning
    information, and all functions dependent on mapping from 
    continuous peaks ->binned spectra should use this. 

    """

    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, first_bin_center,
                 bin_width, bin_number):
        self.first_bin_center = first_bin_center
        self.bin_width = bin_width
        self.bin_number = bin_number

        self.bin_centers = np.arange(self.bin_number) * self.bin_width + self.first_bin_center
        
        # we only support partitions right now, that is
        # spectral bins with no gaps
        self._is_partition = True

    # Function overview: config handles a specific workflow step in this module.
    def config(self):
        return {'first_bin_center' : self.first_bin_center,
                'bin_width' : self.bin_width,
                'bin_number' : self.bin_number}
    
    # Function overview: get_value_range handles a specific workflow step in this module.
    def get_value_range(self):
        """
        Returns the smallest value that could go in the first bin and
        the outer edge of the largest bin
        
        that is [min, max)
        """
        return (self.first_bin_center - self.bin_width/2.0, 
                self.first_bin_center + self.bin_number * self.bin_width + self.bin_width/2.0)

    # Function overview: get_bin_width handles a specific workflow step in this module.
    def get_bin_width(self):
        return self.bin_width
    
    # Function overview: get_num_bins handles a specific workflow step in this module.
    def get_num_bins(self):
        return self.bin_number
    
    # Function overview: get_bin_centers handles a specific workflow step in this module.
    def get_bin_centers(self):
        """
        Get the centers of the bins
        """
        return np.arange(self.bin_number)*self.bin_width + self.first_bin_center
        
    # Function overview: is_partition handles a specific workflow step in this module.
    def is_partition(self):
        """
        Returns true if this is a partitioning of a range (that is, no gaps)

        Right now always returns true. 
        """
        return self._is_partition

    # Function overview: __getitem__ handles a specific workflow step in this module.
    def __getitem__(self, value):
        """
        SLOW returns the bin associated with a value, or -1 
        if none
        """
        
        return self.to_bins([value])[0]

    # Function overview: to_bins handles a specific workflow step in this module.
    def to_bins(self, values):
        """
        Returns the integer bin that a particular mass value maps to, or
        -1 for outside of the range. 
        """
        values = np.array(values)
        
        binned_value_min, binned_value_max = self.get_value_range()
        value_range = self.bin_width * self.bin_number
        
        value_bins = (values -self.first_bin_center + self.bin_width/2) / self.bin_width
        value_bins_int = np.floor(value_bins).astype(np.int32)
        value_bins_int[(value_bins_int < 0) | (value_bins_int >= self.bin_number)] = -1
        return value_bins_int

    # Function overview: histogram handles a specific workflow step in this module.
    def histogram(self, masses, intensities):
        """
        Bin "masses" into the appropriate bins weighted with intensities, 
        and then normalize the entire resulting histogrammed spectrum
        to have unit mass. 

        Returns :
         (locations of non-zero bins, 
          values at those non-zero bins, 
          dense histogram of values)
        """
        target_bins = self.to_bins(masses)

        target_bins_valid = target_bins >= 0
        h, _ = np.histogram(target_bins[target_bins_valid],
                            bins=np.arange(self.bin_number+1),
                            weights=intensities[target_bins_valid])
        idx = np.argwhere(h).flatten()
        
        total_p = max(np.sum(h[idx]), 1e-6)
        p = h[idx] / total_p 
        dense_out = h / total_p
        return idx, p, dense_out

        

# Function overview: create_spectrum_bins handles a specific workflow step in this module.
def create_spectrum_bins(**config):
    """
    Create a spectrum config. Right now just a wrapper around
    instantiating the class directly. 
    
    """
    
    return SpectrumBins(**config)



# Class overview: MassPeaksToBins encapsulates a reusable component in this module.
class MassPeaksToBins:
    """
    Take in an array of N spectra of real-valued mass/intensity pairs
    and discretize them into bins, returning index/intensity pairs. 

    
    
    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, first_bin_center, bin_width, bin_number):
        self.sb = SpectrumBins(first_bin_center, bin_width, bin_number)
        
    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(self, peaks_and_bins):
        # If compiled fast implementation is available, use it; otherwise
        # fall back to the pure-Python histogramming implementation.
        if fast is not None and hasattr(fast, 'mass_bin_peaks_fast'):
            output_peak_idx, output_peak_val = fast.mass_bin_peaks_fast(
                peaks_and_bins,
                self.first_bin_center,
                self.bin_width,
                self.bin_number,
            )
            return output_peak_idx, output_peak_val

        # Pure-Python fallback
        BATCH_N, MAX_PEAK, _ = peaks_and_bins.shape
        output_peak_idx = np.zeros((BATCH_N, MAX_PEAK), dtype=np.int64)
        output_val = np.zeros((BATCH_N, MAX_PEAK), dtype=np.float32)
        for fi, f in enumerate(peaks_and_bins):
            idx, p, _ = self.sb.histogram(f[:, 0], f[:, 1])
            output_peak_idx[fi, :len(idx)] = idx
            output_val[fi, :len(idx)] = p
        return output_peak_idx, output_val
        assert peaks_and_bins.shape[2] == 2
        
        output_peak_idx = np.zeros((BATCH_N, MAX_PEAK), dtype=np.int64)
        output_val = np.zeros((BATCH_N, MAX_PEAK), dtype=np.float32)
        for fi, f in enumerate(peaks_and_bins):
            idx, p, _ = self.sb.histogram(f[:, 0], f[:, 1])
            output_peak_idx[fi, :len(idx)] = idx
            output_val[fi, :len(idx)] = p
        return output_peak_idx, output_val



# Class overview: MassPeaksToBinsFast encapsulates a reusable component in this module.
class MassPeaksToBinsFast:
    """
    Take in an array of N spectra of real-valued mass/intensity pairs
    and discretize them into bins, returning index/intensity pairs. 


    This is the fast cython version of MassPeaksToBins
    
    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, first_bin_center, bin_width, bin_number):
        self.first_bin_center = first_bin_center
        self.bin_width = bin_width
        self.bin_number = bin_number
        
    
    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(self, peaks_and_bins):
        
        output_peak_idx, output_peak_val = fast.mass_bin_peaks_fast(peaks_and_bins, 
                                                               self.first_bin_center,
                                                               self.bin_width,
                                                               self.bin_number)

            
        return output_peak_idx, output_peak_val     
        

# Function overview: create_peaks_to_bins handles a specific workflow step in this module.
def create_peaks_to_bins(spectrum_bins):
    """
    Factory for creating the mass-peaks-to-bins object, 
    for future expansion/configuraiton
    """

    first_bin_center = spectrum_bins.get_bin_centers()[0]
    bin_width = spectrum_bins.get_bin_width()
    bin_number = spectrum_bins.get_num_bins()

    if fast is not None and hasattr(fast, 'mass_bin_peaks_fast'):
        return MassPeaksToBinsFast(first_bin_center, bin_width, bin_number)
    return MassPeaksToBins(first_bin_center, bin_width, bin_number)
