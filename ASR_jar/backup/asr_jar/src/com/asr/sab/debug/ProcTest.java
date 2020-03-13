package com.asr.sab.debug;

import static com.asr.sab.utils.MyMatrixUtils.calculate_mixingMatrix;
import static com.asr.sab.utils.MyMatrixUtils.calculate_thresholdMatrix;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import com.asr.sab.cal.ASR_Calibration;
import com.asr.sab.proc.ASR_Process;
import com.asr.sab.utils.MyMatlabUtils;
import com.asr.sab.utils.MyMatrixUtils;
import com.asr.sab.utils.Proc_Utils;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import ml.utils.Matlab;



public class ProcTest {

	private static final double CONTAINS_ARTIFACT = 0.0;
	private static final double IS_CLEAN = 1.0;
	private static final DoubleBinaryOperator TIMES = (double a, double b) -> a * b;


	//private static final DoubleBinaryOperator TIMES = (double a, double b) -> a * b;

	public static void main(String[] args) {
		
		int sRate = 100;
		int C = 3;
		int S = 50;
		
        /* 
         * compose input X
         * which has samples in minor dim X[0].length and channels ind major X.length
         */
    	double[][] data = new double[C][S];

        for (int i = 0; i < 50; i++) {
            double addend = 0.01 * i;
            data[0][i] = 1.01 + addend;
            data[1][i] = 4.01 + addend;
            data[2][i] = 6.01 + addend;
        }
        
        /*
         * ------------------------- CALIBRATION -----------------------------------------
         * Here the single methods are called seperately, but only for debugging purposes
         */
        int blocksize = 5;
        int N = 5;
        int stepsize = 32;
        

        double[][] M = calculate_mixingMatrix(data, blocksize);
		double window_overlap = 0.66;
		double[][] T = calculate_thresholdMatrix(data, M, window_overlap, N);
        
		/*
		 * because of the shortcut in the calculation of mu and sigma (and only because of that) my
		 * threshold matrix is different than the one calculated in matlab
		 */
		ASR_Calibration state = new ASR_Calibration();
		state.M = M;
		state.T = T;
		// -------------------------------------------------------------------------------
		
		
		
		// ----------------------------PROCESSING ----------------------------------------
		ASR_Process proc = new ASR_Process(state);
		int P = 10;
		proc.carry = Proc_Utils.init_carry(data, P);

		Matrix datMat = new DenseMatrix(data).transpose();
		Matrix carryMat = new DenseMatrix(proc.carry).transpose();
		
		// put carry data in front of data 
		double[][] paddedData = Matlab.cat(1, carryMat, datMat).getData();
		
		
		/*
		 * filter data 
		 */
		double[][] X = new double[paddedData.length][paddedData[0].length];
        double[] channelVal;
        for (int c = 0 ; c < paddedData.length; c++){
            channelVal = MyMatrixUtils.filter_one_channel(state.B, state.A, paddedData[c]);
            X[c] = channelVal; // rows = channels
        } 
        
        // -------------------- cov matrix for running mean -----------------
        // this is done with the unfiltered input again, but only for debugging purposes.

        
        // the state object gets its cov field handled in here
        double[][] Xcov = MyMatrixUtils.moving_average(N, MyMatrixUtils.transpose(X), state);
        
        // let's take a shortcut here 
        // min(stepsize:stepsize:(size(Xcov,2)+stepsize-1),size(Xcov,2))
        int n = stepsize-1;
        
        if(state.last_R == null) {
    		//prevCov = zeros(size(X,1),N)
    		state.last_R = Matlab.eye(C).getData();
    	}
        
        /*
         * We want the loop to run as often as we can look into the samples with a distance of stepsize-1 intervals
         */
        
        double[][] Xcov_update_at = Proc_Utils.extract_Xcov_at_stepsize(Xcov, stepsize);
        
        
		double[][][] Xcov3 = MyMatlabUtils.reshape_2_to_3(MyMatrixUtils.transpose(Xcov_update_at), Xcov_update_at.length, C, C);
        
        double maxdims = 0.66;
        maxdims = Math.round(C * maxdims);
        
        // ----------------------- PCA to find artefacts ------------------------------------------------------
        /*
         * this loop is not inside another loop in this code, because we only treat the online processing where we 
         * pass a small chunk of data to the processing method and then clean everything at once (in chunks of 
         * stepsize sample-ranges). In the Matlab code, we also treat the case that incoming data is a huge 
         * dataset which needs to be split into smaller chunks before processing. 
         * 
         * This loop runs until we cleaned everything in the passed datachunk, in steps of stepsize. 
         */
        
        int last_n = 0;

        for (int u = 0; u < Xcov_update_at.length; u++) {
             /*
              * The Eigendecomposition will always have the form C*C.
              * 
              * d = eigenvalues, v = eigenvectors
              * 
              * D is of the form:  
              *    [lambda, mu    ]
              *    [   -mu, lambda]
              *    
              * The columns of V represent the eigenvectors in the sense that A*V = V*D,
              * i.e. A.multiply(V) equals V.multiply(D).
              *    
              * - sign does not matter: the set of eigenvectors is a linear subspace and is closed under scalar multiplication
              * - the order in different implementations may vary: this one is different than in Matlab
              * 
              */
        	EigenDecomposition XcovEigen = new EigenDecomposition(new Array2DRowRealMatrix(Xcov3[u]));
        	double[] keep = Proc_Utils.compute_threshold_operator(C, T, maxdims, XcovEigen);
        	/*
        	 *  trivial = all(keep), true if all elements are true, false otherwise:
        	 *  if keep.contains(CONTAINS_ARTIFACT): foundArtifact = true
        	 */
    		boolean foundArtifact = Arrays.asList(keep).contains(CONTAINS_ARTIFACT);
    		
    		/*
    		 * if we found artifacts (~trivial)
    		 * 		R = real(M*pinv(bsxfun(@times,keep',V'*M))*V');
    		 * 
    		 * if we did not find artifacts, keep data as is and compose R as:
    		 * 		R = eye(C);
    		 */
    		
    		double[][] R; // = new double[X.length][X[0].length];
    		if (foundArtifact) {
    			R = Proc_Utils.compute_R(XcovEigen, keep, M, C);
    		} else {
    			R = Matlab.eye(C).getData();
    		}
    		
    		/*
    		 * apply R to intermediate samples (raised-cosine blending)
    		 * use update_at as index: always take stepsize-1 samples and go stepsize-1 samples further 
    		 * blend = (1-cos(pi*(1:(n-last_n))/(n-last_n)))/2;
    		 */
    		int[] subrange = IntStream.rangeClosed(last_n, n).toArray(); 
    		double[] blend = Arrays.stream(subrange).mapToDouble(t -> (t * Math.PI)/stepsize)
    												.map(t -> (1 - Math.cos(t)) / 2)
    												.toArray();
    		
    		// rows = channels, columns = samples
    		/*
    		 *  firstPart = bsxfun(@times, blend, R*X(:,subrange))
    		 */
    		RealMatrix dataRange = new Array2DRowRealMatrix(MyMatrixUtils.transpose(X)).getSubMatrix(0, C-1, last_n, n);
    		double[][] RX = new Array2DRowRealMatrix(R).multiply(dataRange).getData();
    		double[][] firstPart = MyMatlabUtils.bsxfun(MyMatlabUtils.reshape_1_to_2(blend, C), RX, TIMES);
    		
    		
    		/*
    		 * secondPart = bsxfun(@times,1-blend,state.last_R*X(:,subrange)
    		 */
    		double[][] last_RX = new Array2DRowRealMatrix(state.last_R).multiply(dataRange).getData();
    		//FIXME 1-blend
    		double[] invertedBlend = MyMatlabUtils.subtract_constant_1d(1.0, blend);
    		double[][] secondPart = MyMatlabUtils.bsxfun(MyMatlabUtils.reshape_1_to_2(invertedBlend,  C), last_RX ,TIMES);
    		
    		/*
    		 * X(:,subrange) = bsxfun(@times, blend, R*X(:,subrange)) + bsxfun(@times,1-blend,state.last_R*X(:,subrange));
    		 * X(:,subrange) = firstPart + secondPart;
    		 */
    		double[][] Xsub =  new Array2DRowRealMatrix(firstPart).add(new Array2DRowRealMatrix(secondPart)).getData();
    		
    		/*
    		 * change the content of X and replace some samples with the content of Xsub which contains the multiplied
    		 * values of blend and R.
    		 */
    		for (int i = 0; i < subrange.length; i++) {
    			new Array2DRowRealMatrix(X).setRow(subrange[i], MyMatrixUtils.transpose(Xsub)[i]); 
			}
    		
    		// update fields
    		last_n = n;
    		n += stepsize; // always increase by stepsize
    		state.last_R = R;
    		
    		// FIXME this is not correct, is it
    		if (n > X[0].length) break;
    		
        }
        
        /*
         * finalize outputs 
         */
        
        
        
	}

	
	
	
	


}
