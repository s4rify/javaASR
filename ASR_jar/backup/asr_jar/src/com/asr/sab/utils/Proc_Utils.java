package com.asr.sab.utils;

import java.util.LinkedList;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

public class Proc_Utils {
	
	private static final double CONTAINS_ARTIFACT = 0.0;
	private static final double IS_CLEAN = 1.0;
	private static final DoubleBinaryOperator TIMES = (double a, double b) -> a * b;
	
	/**
	 * When we are in the very first iteration of processing, the carry part is empty.
	 * Therefore, we interpolate a few samples from the exisiting data that we are 
	 * processing and initialize carry with it:
	 * state.carry = repmat(2*data(:,1),1,P) - data(:,1+mod(((P+1):-1:2)-1,S))
	 * 
	 * 
	 * @param data the current data chunk
	 * @return	carry
	 */
	public static double[][] init_carry(double[][] data, int P) {
		/*
		 * repmat(2*data(:,1),1,P)
		 */
		// double each value from the first sample
		RealMatrix dataM = new Array2DRowRealMatrix(data);
		dataM = dataM.getColumnMatrix(0).scalarMultiply(2);
		// replicate the first sample P times 
		double[][] replicatedSample = new double[P][data[0].length];
		for (int i = 0; i < P; i++) {
			replicatedSample[i] = dataM.getColumn(0);
		}
		
		/*
		 * data(:,1+mod(((P+1):-1:2)-1,S)
		 * -> for us: get P-1 samples sorted from new (index P) to old (index 0) 
		 */
		int i = 0;
		double[][] samplesOldToNew = new double[data.length][P];
		for (int c = 0; c < data.length; c++) {
			i = 0;
			for (int s = P; s > 0; s--) {
				samplesOldToNew[c][i] = data[c][s];
				i++;
			}
		}
		
		/* 
		 * subtract elements: repmat(2*data(:,1),1,P) - data(:,1+mod(((P+1):-1:2)-1,S))
		 * -> replicated duplicated samples minus newest to oldest samples 
		 */
		return new Array2DRowRealMatrix(replicatedSample)
				.transpose()
				.subtract(new Array2DRowRealMatrix(samplesOldToNew))
				.getData(); 
		
	}
	
	/*
	 * update_at = min(stepsize:stepsize:(size(Xcov,2)+stepsize-1) , size(Xcov,2))
	 * Xcov(:,update_at)
	 * 
	 * Iterate through the Xcov matrix in stepsize - bins.
	 * The first and last entry of Xcov is always chosen for further use and collected in Xsegmnts.
	 * The intermediate samples are taken successively with a distance of stepsize.
	 * 
	 */
	public static double[][] extract_Xcov_at_stepsize(double[][] in, int stepsize) {
		List<double[]> collectedSegments = new LinkedList<>();
        
		// the first entry contains the first entry of Xcov 
		collectedSegments.add(in[0]);
        
		// treat all the entries except the first one here
        int i = stepsize - 1;
        for (; i < in.length; i += stepsize) {
        	collectedSegments.add(in[i]);
		}
        // if the last entry is not exactly the last entry of Xcov, put the last entry in Xsegemnts anyway
        if (i != in.length - 1 + stepsize) {
        	collectedSegments.add(in[in.length-1]);
        }
        
        // we need an array in the end
        return collectedSegments.toArray(new double[collectedSegments.size()][]);
	}
	
	

	/**
	 * The reconstruction matrix R is computed using the Eigendecomposition of M, the threshold operator and
	 * the mixing matrix M of the signal X.
	 * 
	 * In the paper: 			R = V * M * pinv( M x U) * V'
	 * In the Matlab code: 		R = M * pinv( (V'*M) x U')	* V'
	 * Hm. What do we make of that?
	 * 
	 * @param e		Eigendecomposition of the covariance matrix of X
	 * @param keep	The threshold operator (this is called U in the paper).
	 * @param M		The mixing matrix which is the square root matrix of the sample covariance matrix of X
	 * @param C		The number of channels in X
	 * @return
	 * 				The reconstruction matrix R which is applied to X in a later step.
	 */
	public static double[][] compute_R(EigenDecomposition e, double[] keep, double[][] M, int C) {
		final DoubleBinaryOperator TIMES = (double a, double b) -> a * b;

		//V'*M
		RealMatrix VM = new Array2DRowRealMatrix(M).multiply(e.getV()); // or getVT()
		// keep'* (V'*M)
		double[][] kVM = MyMatlabUtils.bsxfun(MyMatrixUtils.transpose(MyMatlabUtils.reshape_1_to_2(keep, C)), VM.getData(), TIMES);
		
		// pinv = pinvfun(bsxfun(@times,keep',V'*M))
		// note thatt the LUDecomposer is only defined for square matrices!
		double[][] pinv = new LUDecomposition(new Array2DRowRealMatrix(MyMatrixUtils.transpose(kVM))).getSolver().getInverse().getData();
		
		//M * pinv * V
		double[][] MPV = (new Array2DRowRealMatrix(M).multiply(new Array2DRowRealMatrix(pinv))).multiply(e.getVT()).getData();

		// omg.
		return MPV;
	}



	/**
	 *TODO Put in ProcessUtils
	 *
	 * The calculation of the threshold operator U. 
	 * U = 0 if std of the current segment k > t(k), where t element threshold matrix T 
	 * U = 1 else.
	 * 
	 * This threshold operator is used to decide which components to keep for later use and which
	 * components contain artefacts and must be replaced by the content of R.
	 * 
	 * @param C			Number of channels in the signal
	 * @param T			The threshold matrix, calculated either in the calibration or an earlier run of the processing
	 * @param maxdims	I have no idea.
	 * @param e			The Eigendecomposition of the covariance matrix of the signal X.
	 * @return	
	 * 					The threshold operator U (here keep) which is a double vector that contains
	 * 					a 1.0 (true) for every channel which does not contain artifacts and can be kept as is
	 * 					and a 0.0 for every channel that contains artifacts and cannot be kept as is. 
	 */
	public static double[] compute_threshold_operator(int C, double[][] T, double maxdims, EigenDecomposition e) {
		/*
		 * determine which components to keep (variance below directional threshold or not admissible for rejection)
		 * keep = D<sum((T*V).^2) | (1:C)<(C-maxdims);
		 * keep = D < sumsqTV	  |
		 */
		double[][] TV = MyMatrixUtils.transpose(MyMatrixUtils.array_multiplication(T, e.getV().getData()));
		double[] sumsqTV = new double[Math.max(TV.length, TV[0].length)]; 
		
		/*
		 * sum((T*V).^2)
		 */
		for (int i = 0; i < TV.length; i++) {
			for (int j = 0; j < TV[0].length; j++) {
				sumsqTV[i] += TV[i][j] * TV[i][j];
			}
		}
		
		/*
		 * keep = D < sumsqTV
		 */
		double[] keep = new double[sumsqTV.length];
		int[] channelRange = IntStream.rangeClosed(0,C).toArray();
		double dim = C - maxdims;

		for (int j = 0; j < TV[0].length; j++) {
			if(e.getD().getData()[0][j] < sumsqTV[j] || channelRange[j] < dim) {
				keep[j] = IS_CLEAN;
			} else {
				keep[j] = CONTAINS_ARTIFACT;
			}
		}
		return keep;
	}
	

}
