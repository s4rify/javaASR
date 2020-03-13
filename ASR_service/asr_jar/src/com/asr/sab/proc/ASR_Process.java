package com.asr.sab.proc;

import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;

import com.asr.sab.cal.ASR_Calibration;
import com.asr.sab.filter.FilterFactory;
import com.asr.sab.filter.IFilter;
import com.asr.sab.utils.MyMatrixUtils;
import com.asr.sab.utils.Proc_Utils;

import ml.utils.Matlab;


/**
 * Created by Sarah Blum on 9/21/17
 * Carl von Ossietzky Univesity of Oldenburg.
 * 
 * This is the implementation of asr_process in Matlab. 
 * 
 */
public class ASR_Process {

	int N;
	
	/*
	 * stepsize is taken directly from Matlab, for the online case, 32 seems to be a reasonable value
	 */
	public int stepsize = 31;
	
	/*
	 * tuning parameter which is set in the constructor
	 */
	public int P;
	
	/*
	 * A portion of the last datachunk that has been cleaned. It is appended in the beginning to the 
	 * current datachunk to be cleaned. By appending this, the filter gets more samples to work with 
	 * and the calculations of the covariance matrix takes some previous state into account.  
	 */
	public double[][] carry;
	
	/*
	 * The covariance matrix of the previous call of this method.
	 * This is Xcov from the Matlab code, not state.cov.
	 * 
	 * concerning state.cov:
	 * at the moment, state.cov is not read anywhere, it gets filled up correctly but
	 * there is no call to the method at the current implementation (it is also not tested)
	 * When using the processing online (and therefore calling it over and over again), we dont need
	 * a state here anyway ( I think).
	 */
	//public double[][] prevCov = new double[0][0];
	
	
	public double[][] Xcov = new double[0][0];
	
	/*
	 * The reconstruction matrix of the last loop iteration. The reconstruction matrix is computed 
	 * over and over again during this processing, for every entry in the third dimension of the 
	 * covariance matrix. 
	 */
	public double[][] last_R;
	
	/*
	 * The sampling rate of the datachunk which is to be cleaned. 
	 */
	private final int sRate;
	
	/*
	 * Mixing matrix from calibration.
	 */
    public final double[][] M;
    
    /*
     * Threshold matrix from calibration
     */
    public double[][] T;
    
    /*
     * This parameter is used to calculate P, a parameter which determines how much samples
     * we are using for different computations during the processing.
     */
	private double lookahead;

	private final IFilter filter;

	/*
	 * For the very first call of the processing, we create a new object of the processing class.
	 * It needs to have the initially calculated values in order to be able to 
	 * make the very first calculations on the online data.
	 * 
	 * @param calibState		The calibration state object which contains initially calculated 
	 * 							values.
	 * @param sampleRate		The sample rate in Hz.
	 */
	public ASR_Process(ASR_Calibration calibState, int sampleRate) {
		this.M = calibState.getMixingMatrix();
		this.T = calibState.getThresholdMatrix();
		this.N = ASR_Calibration.N;
		this.lookahead = ASR_Calibration.window_len/2;
		this.sRate = sampleRate;
		this.P = (int)Math.round(lookahead * sRate);
		/*
		 * create a filter object and use calibration state since we do not have a 
		 * previous state. The state is handled by the filter internally for every call of asr_process
		 * after this initial one
		 */
		this.filter = FilterFactory.getFilter();
		filter.setState(calibState.getFilterState());
	}

	/*
	 * This method gets called again and again
	 */
	public double[][] asr_process(double[][] rawData) {
		
		/*
		 * extract channel and sampling rate information from the current rawData datachunk
		 */
		int C = rawData.length;
		int S = rawData[0].length;
		
		/*
		 * this is the data chunk we fill into the returned data array It gets filled up in the very last step
		 * of the processing loop, where we use the recontruction matrix R to reconstruct segments
		 * of the raw data which contain artfacts (using raised cosine blending).
		 * These segments are then filled into the finally returned array 
		 */
		double[][] reconstructedDataChunk = new double[C][stepsize];

		/*
		 * A parameter which determines the size of the steps in which we look at the data
		 * during the processing loop. 
		 * (let's take a shortcut here. 
		 * Matlab code: min(stepsize:stepsize:(size(Xcov,2)+stepsize-1),size(Xcov,2)))
		 */
		int n = stepsize-1;
		
		/*
		 * paremeter which we need in the processing loop to determine the sample range we process
		 */
		int last_n = 0;
		
		/*
		 * initialize the carry matrix. This matrix contains data from the previous 
		 * calculation with an earlier data segment. It is used, to enlarge the current
		 * data chunk for the filter, so that more information is given into it.
		 */
		this.carry = Proc_Utils.init_carry(rawData, P);

		/*
		 * Put the carry portion of the data (the previous data chunk) in front of the current
		 * data chunk so that the filter gets more values to work with. 
		 */
		double[][] paddedData = Proc_Utils.pad_data_with_previous_segment(rawData, this.carry);
		
		/*
		 * The array, we return in the end. It gets filled up iteratively in stepsize-chunks with reconstructed data
		 * Whenever we don't find an artefact, the original data is returned, so we fill it in here and 
		 * only reconstruct the artefactual parts.
		 */
		double[][] reconstructedDataTotal = paddedData;

		/*
		 * we only put a lookahead portion of padded data into the filter
		 */
		double[][] paddedDataSegment = new double[C][S];
		for (int c = 0; c < C; c++) {
			for (int splus = P, s = 0; splus < S+P && s < S; splus++, s++) {
				paddedDataSegment[c][s] = paddedData[c][splus];
			}
		}
		
		/*
		 * Filter the padded current data chunk given the filter coefficients and the filter state from the previous call 
		 * to this method. In the very first run, when the constructor is called, the filter state is
		 * initialized using the calibration state object.
		 */
		double[][] X = filter.filter_data(paddedDataSegment, (double) sRate);

		
		/*
		 * Compute the current sample covariance matrix. This is the basis for all 
		 * following computations.
		 * This fills two fields of the processing object:
		 * - the Xcov field which is further used for the PCA and stuff
		 * - in the original code: the prevcov field which contains a chunk of Xcov from the last iteration
		 * 		(note that this never applies in the online case where asr_process() gets called
		 * 		over and over again, without splitting the data inside asr_process!!)
		 */
        this.Xcov = Proc_Utils.compute_cross_cov(N, X);
        
        /*
         * last_R is used in the processing loop for the cleaning of the current datachunk
         * If in the previous data chunk, the reconstruction matrix has not been determied (for example because
         * the previous data chunk was the calibration data and we just started with the online processing),
         * the reconstruction matrix is initialized with the identity matrix of the same channel dimensions. 
         */
        if(this.last_R == null) {
    		this.last_R = Matlab.eye(C).getData();
    	}
        
        /*
         * This is the computation of a stable covariance matrix, where incrementally samples are taken and used
         * altough they are *stepsize* apart. This leads to a more stable representation, omitting 
         * sharp but short outliers.
         */
        double[][][] Xcov3 = Proc_Utils.extract_Xcov_at_stepsize(this.Xcov, stepsize, C);
        
        

        double maxdims = 0.66;
        maxdims = Math.round(C * maxdims);
        boolean last_foundArtifact = false;

        /*
         * This is the main processing loop which computes an eigendecompsition on the covariance matrix
         * and determines in the component space, whether a component contains an artifact. 
         *  
         * This loop runs until we cleaned everything in the passed datachunk, in steps of stepsize. 
         */
        for (int u = 0; u < Xcov3.length; u++) { //stepsize-1

        	// Xcov(:,:,u)
        	double[][] curXcov = Proc_Utils.computeCurrentXcov(C, Xcov3, u);
        	
        	/*
        	 * The eigendecomposition in Matlab consitently finds right eigenvectors, whereas the Java eigendecomposition
        	 * does not give any assertions about finding left or right eigenvectors. 
        	 * In the test, this yields the correct results: transposing the input for the decomposition.
        	 * This is based on:  a left eigenvector of A is the same as the transpose of a right eigenvector of AT, 
        	 * with the same eigenvalue (wikipedia).
        	 */
        	// eig(Xcov(:,:,u))
        	EigenDecomposition XcovEigen = new EigenDecomposition(MatrixUtils.createRealMatrix(curXcov)); 
        	
        	/*
        	 * [D,order] = sort(reshape(diag(D),1,C)); 
        	 * V = V(:,order)
        	 * since D comes sorted out of the library, the sorting vector here is 1:C, and for compatibility, it will be C:1
        	 */
        	double[] diagD = Proc_Utils.sortD(C, XcovEigen.getD().getData());  
        	double[][] Vsort = Proc_Utils.sortV(C, XcovEigen.getV().getData());
        	
        	/*
    		 * Threshold value which is used to compute the threshold operator 'keep'. 
    		 * The threshold operator keep is used to indicate whether a component contains artifacts.
    		 * If keep is one, then this component can be kept, it does not contain an artifact.
    		 * If keep is zero, then this component cannot be kept, because it contains an artifact.
    		 */
        	double[] keep = Proc_Utils.compute_threshold_operator(C, T, maxdims, Vsort, diagD);
        	
        	/*
        	 * trivial = all(keep), true if all elements are true, false otherwise:
			 * if keep does not contain a '1' at every entry, we found an artifact!
			 */
        	double keepSum = DoubleStream.of(keep).sum();
			boolean foundArtifact = keepSum < C; // Java is way too sensitive for artifacts

			/*
			 * check which reconstruction matrix we want to use
			 */
    		double[][] R;
    		if (foundArtifact) {
    			R = Proc_Utils.compute_R(Vsort, keep, M, C); 
    			//System.out.println("reconstruction in " + u);
    		} else {
    			R = Matlab.eye(C).getData();
    		}

    		/*
			 * determine which range of rawData we clean in this iteration
			 */ 
			int[] subrange = (last_n + stepsize > S)
							? IntStream.rangeClosed(last_n, S-1).toArray()	// boundary condition: only in last iteration
							: IntStream.rangeClosed(last_n, n).toArray(); 	// major case

			
			if (foundArtifact || last_foundArtifact) {
				/*
				 * reconstruct the current data segment using the reconstruction matrix R. 
				 * In case we don't have an artifact, the reconstructedData is nevertheless created
				 * by multiplying the input signal with the reconstruction matrix, where the reconstruction matrix is the identity matrix. 
				 */
	    		reconstructedDataChunk = Proc_Utils.reconstruct_data(C, paddedData, R, last_R, subrange);
	    		
	    		/*
	    		 * fill the reconstructed data into the output structure 
	    		 */
	    		int i = 0;
	    		int from = subrange[0];
	    		int to = subrange[subrange.length-1]; 
	    		for (int c = 0; c < C; c++) {
					for (int s = from; s <= to; s++) {
						reconstructedDataTotal[c][s] = reconstructedDataChunk[i][c];
						i++;
					}
					i = 0;
				}
			}
    		
			/*
			 * In the very first case of the loop we have a boundary condition and do not want to update n
			 * This is due to the way Matlab computes the update_at field
			 */
			if (u != 0) {
				last_n = n+1;
				n += stepsize;
			}
    		last_R = R;
    		last_foundArtifact = foundArtifact;
    		if (subrange[subrange.length-1] >= S-1) break;
        }
        
        /*
         * return cleaned up, unfiltered data minus lookahead portion:
         * outdata = data(:, 1:(end-P));
         * copyofrange reads the array in the wrong direction, this is why we transpose the input.
         * it also fills the result up the wrong way around, this is why we transpose the output (fail). 
         */
        double[][] outdata = Arrays.copyOfRange(MyMatrixUtils.transpose(reconstructedDataTotal), 0, reconstructedDataTotal[0].length-P); 
        return MyMatrixUtils.transpose(outdata);
	}
}
	
	

