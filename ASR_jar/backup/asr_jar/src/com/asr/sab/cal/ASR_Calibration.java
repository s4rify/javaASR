package com.asr.sab.cal;


import static com.asr.sab.utils.MyMatrixUtils.calculate_mixingMatrix;
import static com.asr.sab.utils.MyMatrixUtils.calculate_thresholdMatrix;
import static com.asr.sab.utils.MyMatrixUtils.filter_one_channel;

/**
 * Created by Sarah Blum on 9/21/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class ASR_Calibration {
    int blocksize;
    double N;
    public double sRate = 250;
    
    /*
     * state object content in Matlab
     */
    public boolean last_trivial = true;
    public double[][] M;
    public double[][] T;
    public double[][] last_R;
    public double[][] cov;


    // use precomputed filter coefficients for sRate of 250Hz
    public final double[] B = {1.7587013141770287, -4.3267624394458641,  5.7999880031015953,
            -6.2396625463547508,  5.3768079046882207, -3.7938218893374835,  2.1649108095226470,
            -0.8591392569863763,  0.2569361125627988};

    public final double[] A = {1.0000000000000000, -1.7008039639301735,  1.9232830391058724,
            -2.0826929726929797,  1.5982638742557307, -1.0735854183930011,  0.5679719225652651,
            -0.1886181499768189,  0.0572954115997261};

	
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
    public ASR_Calibration calibrate(double[][] calibData){

    	int C = calibData.length;
    	int S = calibData[0].length;

        this.M = new double[C][C];
        this.T = new double[C][C];
    	
        /*
         * Filter
         */
        double[][] filtered = new double[C][S];
        double[] channelVal;
        for (int c = 0 ; c < C; c++){
            channelVal = filter_one_channel(this.B, this.A, calibData[c]);
            filtered[c] = channelVal; // rows = channels
        }

        /**
         * Mixing matrix M
         */

        this.M = calculate_mixingMatrix(filtered, this.blocksize);


        /*
         * Calculate threshold matrix
         */
        this.T = calculate_thresholdMatrix(filtered, M, sRate, this.N);
        
        return this;

    }






}
