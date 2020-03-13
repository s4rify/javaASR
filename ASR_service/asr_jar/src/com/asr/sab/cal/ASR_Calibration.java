package com.asr.sab.cal;


import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;

import com.asr.sab.filter.FilterFactory;
import com.asr.sab.filter.FilterState;
import com.asr.sab.filter.IFilter;
import com.asr.sab.utils.Calib_Utils;
import com.asr.sab.utils.MyMatrixUtils;
import com.asr.sab.utils.Proc_Utils;

/**
 * Created by Sarah Blum on 9/21/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class ASR_Calibration {

	/**
	 * Value which is used to determine how we separate the incoming datachunk 
	 * for the calculation of the sample covariance matrix.
	 */
	int blocksize = 10;
	
	/**
	 * Window length for calculating thresholds for the artifact detection.
	 *  Window length that is used to check the data for artifact content. This is 
	 *  ideally as long as the expected time scale of the artifacts but short enough to 
	 *	allow for several 1000 windows to compute statistics over. Default: 0.5. 
	 */
    public static double window_len = 0.5;
    
    
    public static double window_overlap = 0.66;
    
    /**
     * The sampling rate in Hz of the incoming signal
     */
    public static double sRate = 250;
    
    /**
     * N is used to set over how many samples we want to average in the calculation
     * of the amplitude rms per channelpair.
     */
    public static int N = (int)Math.round(window_len * sRate) ;
    
    /*
     * This value is used in the processing method only. It states whether the last
     * processed chunk contained an artifact. Since the calibration data is 
     * artifact-free by definition, we set last_trivial to true.
     */
    public boolean last_trivial = true;
    
    /**
     * The mixing matrix. This is the matrix which is calculated from the sample
     * covariance matrix of the clean calibration data. It is calculated in the calibration,
     * passed to the processing and then never changed again. It is being used for the 
     * reconstruction of clean data in case we find an artifact. 
     */
    public double[][] M;
    
    /**
     * The threshold matrix T. This matrix is computed during the calibration on clean data
     * and then never updated again. It contains rms-based thresholds for every channelpair 
     * (dimension is channel x channel) and is used in the detection of artifacts during the
     * processing. 
     * The threshold operator which is calculated based on this threshold matrix is 
     * updated during the processing, though. 
     */
    public double[][] T;

	private FilterState filterState;

    /**
     * Coefficients of an IIR filter that is used to shape the spectrum of the signal
     *  when calculating artifact statistics. The output signal does not go through
     *  this filter. This is an optional way to tune the sensitivity of the algorithm
     *  to each frequency component of the signal. The default filter is less
     *  sensitive at alpha and beta frequencies and more sensitive at delta (blinks)
     *  and gamma (muscle) frequencies. Default: 
     *  <code>[b,a] = 
     *         yulewalk(8,[[0 2 3 13 16 40 min(80,srate/2-1)]*2/srate 1],[3 0.75 0.33 0.33 1 1 3 3]) </code>
     */


	
	/**
	 * Main calibration method. 
	 * 
	 * 
	 * Matlab Documentation:
	 	* The input to this data is a multi-channel time series of calibration data. In typical uses the
		% calibration data is clean resting EEG data of ca. 1 minute duration (can also be longer). One can
		% also use on-task data if the fraction of artifact content is below the breakdown point of the
		% robust statistics used for estimation (50% theoretical, ~30% practical). If the data has a
		% proportion of more than 30-50% artifacts then bad time windows should be removed beforehand. This
		% data is used to estimate the thresholds that are used by the ASR processing function to identify
		% and remove artifact components.
		%
		% The calibration data must have been recorded for the same cap design from which data for cleanup
		% will be recorded, and ideally should be from the same session and same subject, but it is possible
		% to reuse the calibration data from a previous session and montage to the extent that the cap is
		% placed in the same location (where loss in accuracy is more or less proportional to the mismatch
		% in cap placement).
		%
		% The calibration data should have been high-pass filtered (for example at 0.5Hz or 1Hz using a
		% Butterworth IIR filter).
		%
		% In:
		%   Data : Calibration data [#channels x #samples]; *zero-mean* (e.g., high-pass filtered) and
		%          reasonably clean EEG of not much less than 30 seconds length (this method is typically
		%          used with 1 minute or more).
		
		 *  
		 * @param calibData		The data to be used for the calibration. The array is expected to have
		 * 						channels in the major dimension and samples in the minor dimension:
		 * 						calibdata[channels][samples]
		 * 						All samples are expected to have the same length.
	 * @return 
	 */
    public ASR_Calibration (double[][] calibData){

    	int C = calibData.length;
        this.M = new double[C][C];
        this.T = new double[C][C];
        IFilter filter = FilterFactory.getFilter();
        
        double[][] filtered = filter.filter_data(calibData, sRate);
        this.filterState = filter.getState();
        
        /*
         * Calculate the Mixing matrix M
         */
        this.M = Calib_Utils.calculate_mixingMatrix(filtered, this.blocksize);

        /*
         * component activations
         */
	    EigenDecomposition eig_M = new EigenDecomposition(new Array2DRowRealMatrix(M));
	    double[][] componentActivations = Calib_Utils.compute_component_activations(filtered, eig_M);
	    
	    /*
	     * calculate mean and standard deviation from rms, statistics for each channel
	     * remember: this is a shortcut-calculation. but the values are in the same range
	     *   rms_orig = X(:,c).^2; % column-wise element squaring over time
	     *   rms_orig = sqrt(sum(rms_orig(bsxfun(@plus,round(1:N*(1-window_overlap):S-N),(0:N-1)')))/N);
	     *   RMS(c,:) = rms_orig;
	     */
	    double[][] RMS = MyMatrixUtils.calculate_RMS(componentActivations, window_overlap, N);
	    double[][] channelStats = Calib_Utils.calculateStatistics(RMS);
        
        /*
         * Calculate the Threshold matrix
         */
        this.T = Calib_Utils.calculate_thresholdMatrix(channelStats, eig_M);
    }

    
    /**
     *  For testing: provide M and only compute T
     */
    public ASR_Calibration (double[][] calibData, double[][] precomputedMatrix, boolean calcT) {
    	int C = calibData.length;
    	IFilter filter = FilterFactory.getFilter();

    	/* 
    	 * M is provided, T gets computed
    	 */
    	if (calcT) {
    		this.M = precomputedMatrix;
    		this.T = new double[C][C];
			double[][] filtered = filter.filter_data(calibData, sRate);
    		EigenDecomposition eig_M = new EigenDecomposition(new Array2DRowRealMatrix(precomputedMatrix));
    		double[][] componentActivations = Calib_Utils.compute_component_activations(filtered, eig_M);
    		double[][] RMS = MyMatrixUtils.calculate_RMS(componentActivations, window_overlap, N);
    		double[][] channelStats = Calib_Utils.calculateStatistics(RMS);
    		this.T = Calib_Utils.calculate_thresholdMatrix(channelStats, eig_M);

    		/*
    		 * T is provided, M gets computed
    		 */
    	} else if (!calcT) {
    		this.T = precomputedMatrix;
            this.M = new double[C][C];
            double[][] filtered = filter.filter_data(calibData, sRate);
            this.M = Calib_Utils.calculate_mixingMatrix(filtered, this.blocksize);
    	}
        
    	// save filter state for processing
    	this.filterState = filter.getState();
    }

    
    public double[][] getMixingMatrix() {
		return M;
	}


	public double[][] getThresholdMatrix() {
		return T;
	}

	public FilterState getFilterState() {
		return this.filterState;
	}

}
