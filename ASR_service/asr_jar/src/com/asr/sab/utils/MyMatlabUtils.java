package com.asr.sab.utils;

import java.util.function.DoubleBinaryOperator;

public class MyMatlabUtils {
	
	
	
	/**
	 * A reshaping method which takes a 1-dim input and distributes the content in 2 dimensions as specified by the 
	 * dimension parameter. This behaves like the Matlab method reshape() and produces a square-matrix
	 * 
     *  MATLAB: 
     * 	 reshape(X,M,N,P,...) or reshape(X,[M,N,P,...]) returns an 
     *   N-D array with the same elements as X but reshaped to have 
     *   the size M-by-N-by-P. The product of the specified
     *   dimensions, M*N*P* must be the same as NUMEL(X).
	 * 
	 * @param X
	 * @param M
	 * @return
	 */
    public static double[][] reshape_1_to_2(double[] X, int M) {
    	double[][] Y = new double[M][M];

    	int c = 0,r = 0;
    	for (int i = 0; i < M * M; i++) {
    		Y[c][r] = X[i];
    		r++; 
    		
    		if ((i+1) % M == 0) {				// zero-based: M-1
    			c++;							// continue with next column
    			r = 0;							// and begin to fill it at the beginning
    		}
    	}
    	return Y;
	}
	
	
	
	/**
	 * A reshaping method which takes a 2-dim input and distributes the content in 3 dimensions as specified by the 
	 * three dimension parameters. This behaves like the Matlab method reshape(): 
	 * 
     * MATLAB: 
     * 	 reshape(X,M,N,P,...) or reshape(X,[M,N,P,...]) returns an 
     *   N-D array with the same elements as X but reshaped to have 
     *   the size M-by-N-by-P. The product of the specified
     *   dimensions, M*N*P* must be the same as NUMEL(X).
     *   
     *   Usage in Matlab:	f = reshape(X,          1, C, [])
     *   Usage in Java:		f = reshape(X, samples, 1, C);
     *   Note that the space where Matlab determines the new dimension itself must be specified here and the position
     *   is not the same!
     *   
     * 
	 * @param in		The input array to be reshaped. The elements are taken columnwise (major dim index runs faster than minor index)
	 * @param dOuter	The major [first] dimension of the output array.
	 * @param dMiddle	The middle dimension of the output array.
	 * @param dInner	The minor [third] dimension of the output array.
	 * @return			The output array which is of the form: [dOuter][dMiddle][dInner]
	 */
	public static double[][][] reshape_2_to_3(double[][] in, int dOuter, int dMiddle, int dInner){
		if (dOuter*dInner*dMiddle != in.length * in[0].length) 
			throw new IllegalArgumentException("dOuter * dMiddle * dInner must be equal to the number of elements in X!");
		double[][][] out = new double[dOuter][dMiddle][dInner];
		int srci = 0, srcj = 0;
		
		for (int i = 0; i < dOuter; i++) {
			for (int j = 0; j < dMiddle; j++) {
				for (int k = 0; k < dInner; k++) {
					out[i][j][k] = in[srci][srcj];
					srci++;
					if(srci >= in.length) {
						srci = 0;
						srcj++;
					}
				}
			}
		}
		return out;
	}
	
	
	/**
	 * A reshaping method which takes a3-dim input and distributes the content in 2 dimensions as specified by the 
	 * dimension parameters. This behaves like the Matlab method reshape(): 
	 *
     * MATLAB: 
     * reshape(X,M,N) or reshape(X,[M,N]) returns the M-by-N matrix 
     * whose elements are taken columnwise from X. An error results 
     * if X does not have M*N elements.
     *   
     *   Usage in Matlab:	f = reshape(X,          1, C, [])
     *   Usage in Java:		f = reshape(X, samples, 1, C);
     *   Note that the space where Matlab determines the new dimension itself must be specified here and the position
     *   is not the same!
	 * 
	 * @param in		The input array to be reshaped. The elements are taken as follows: 
	 * 					the minor dim index runs fastest.
	 * @param d1		The major [first] dimension of the output array.
	 * @param d2		The minor dimension of the output array.
	 * @return			The output array which is of the form: [d1][d2] where d2 contains the combined input of 
	 * 					in[][j][k] 
	 */
	public static double[][] reshape_3_to_2(double[][][] in, int d1, int d2){
		if (d1*d2 != in.length * in[0].length * in[0][0].length)
			throw new IllegalArgumentException("d1 * d2 must be number of elements in input!");
		double[][] out = new double[d1][d2];
		int samples = -1, cc = 0;
		for (int i = 0; i < in.length; i++) {
			cc =0;
			samples++;
			for (int j = 0; j < in[0].length; j++) {
				for (int k = 0; k < in[0][0].length; k++) {
					out[samples][cc] = in[i][j][k];
					cc++;
				}
			}
		}
		return out;
	}
	
	
	

	/**
	 * Implementation of the super helpful but also very painful bsxfun() method in Matlab. 
	 * C = bsxfun(A,B, FUNC) applies the element-by-element binary operation
     * specified by the function handle FUNC to arrays A and B, with implicit
     * expansion enabled. 
     * 
     * The output will in every dimension be as big as the respective biggest input dimension.
     * 
     * No safe guards in place, so watch your steps.
     * 
	 * @param a		The input array a.
	 * @param b		The input array b.
	 * @param operator	A binary operator which is applied element-by-element to every element in a and b.
	 * 					e.g.: DoubleBinaryOperator TIMES = (double a, double b) -> a * b;
	 * @return 		An output array which contains all the combined elements of a and b and is in every dimension
	 * 				as big as Math.max(dimA, dimB) returns.
	 */
	public static double[][][] bsxfun(double[][][] a, double[][][] b, DoubleBinaryOperator operator) {
		int d1 = Math.max(a.length, b.length);
		int d2 = Math.max(a[0].length, b[0].length);
		int d3 = Math.max(a[0][0].length, b[0][0].length);
		double[][][] out = new double[d1][d2][d3];
		
		for (int i = 0, ai = 0, bi = 0; i < d1; i++) {
			for (int j = 0, aj = 0, bj = 0; j < d2 ; j++) {
				for (int k = 0, ak = 0, bk = 0; k < d3; k++) {
				
					double operandA = a[ai][aj][ak];
					double operandB = b[bi][bj][bk];
					double cellresult = operator.applyAsDouble(operandA, operandB);
					out[i][j][k] = cellresult;
					
					ak = (ak+1) % a[0][0].length;
					bk = (bk+1) % b[0][0].length;
				}
				aj = (aj+1) % a[0].length;
				bj = (bj+1) % b[0].length;
			}
			ai = (ai+1) % a.length;
			bi = (bi+1) % b.length;
		}
		return out;
	}
	
	
	public static double[] bsxfun(double[] a, double[] b, DoubleBinaryOperator operator) {
		int d1 = Math.max(a.length, b.length);
		double[] out = new double[d1];
		
		for (int i = 0, ai = 0, bi = 0; i < d1; i++) {
				
					double operandA = a[ai];
					double operandB = b[bi];
					double cellresult = operator.applyAsDouble(operandA, operandB);
					out[i] = cellresult;
					
			ai = (ai+1) % a.length;
			bi = (bi+1) % b.length;
		}
		return out;
	}

	
	
	/**
	 * If only I would be so smart as to parameterize this better. But here you go.
	 * 
	 * Implementation of the super helpful but also very painful bsxfun() method in Matlab. 
	 * C = bsxfun(A,B, FUNC) applies the element-by-element binary operation
     * specified by the function handle FUNC to arrays A and B, with implicit
     * expansion enabled. 
     * 
     * The output will in every dimension be as big as the respective biggest input dimension.
     * 
     * No safe guards in place, so watch your steps.
     * 
	 * @param a		The input array a.
	 * @param b		The input array b.
	 * @param operator	A binary operator which is applied element-by-element to every element in a and b.
	 * 					e.g.: DoubleBinaryOperator TIMES = (double a, double b) -> a * b;
	 * @return 		An output array which contains all the combined elements of a and b and is in every dimension
	 * 				as big as Math.max(dimA, dimB) returns.
	 */
	public static double[][] bsxfun(double[][] a, double[][] b, DoubleBinaryOperator operator) {
		int d1 = Math.max(a.length, b.length);
		int d2 = Math.max(a[0].length, b[0].length);
		double[][] out = new double[d1][d2];
		
		for (int i = 0, ai = 0, bi = 0; i < d1; i++) {
			
			for (int j = 0, aj = 0, bj = 0; j < d2 ; j++) {
				
					double operandA = a[ai][aj];
					double operandB = b[bi][bj];
					double cellresult = operator.applyAsDouble(operandA, operandB);
					out[i][j] = cellresult;
				
				aj = (aj+1) % a[0].length;
				bj = (bj+1) % b[0].length;
			}
			ai = (ai+1) % a.length;
			bi = (bi+1) % b.length;
		}
		
		return out;
	}

	
	/**
	 * A very not-helpful method who lays out all the elements of the input x, columnwise. That is, the major dimension
	 * is handled faster than the minor dimension and in the end, all major elements come after each other, than the next
	 * and so on.
	 * 
	 * @param x 	The input array to be flattened out.
	 * @return		An out out array with the length NUMEL(x).
	 */
	public static double[] flatten(double[][] x) {
		// new row should have samples times channels in length
		double[] out = new double[x[0].length * x.length ];
		int i = 0;
		for (int s = 0; s < x[0].length; s++) {
			for (int c = 0; c < x.length; c++) {
				out[i] = x[c][s];
				i++;
			}
		}
		return out;
	}
	
	
	
	public static int[] flatten(int[][] x) {
		// new row should have samples times channels in length
		int[] out = new int[x[0].length * x.length ];
		int i = 0;
		for (int s = 0; s < x[0].length; s++) {
			for (int c = 0; c < x.length; c++) {
				out[i] = x[c][s];
				i++;
			}
		}
		return out;
	}
	
	
	/**
	 * TODO debug me
	 * 
	 * @param in
	 * @param S
	 * @return
	 */
	public static double[][] subtract_constant_2d(double[][] in, double S){
		double[][] out = new double[in.length][in[0].length];
		
		for (int i = 0; i < in.length; i++) {
			for (int j = 0; j < in[0].length; j++) {
				out[i][j] = in[i][j] - S;
			}
		}
		return out;
	}
	
	/**
	 * TODO debug me
	 * 
	 * @param in
	 * @param S
	 * @return
	 */
	public static double[] subtract_constant_1d(double[] in, double S){
		double[] out = new double[in.length];
		
		for (int i = 0; i < in.length; i++) {
				out[i] = in[i] - S;
			}
		
		return out;
	}
	
	
	/**
	 * Subtract FROM the scalar S the elements in the array in. 
	 * The order of the parameters defines the direction of the subtraction.
	 * 
	 * @param in	the array which contains the elements to be subtracted from S
	 * @param S		the scalar we want to subtract the elements from
	 * @return		an array out with the same dimensions as in, which contains the result of S - in[i]
	 */
	public static double[] subtract_constant_1d(double S, double[] in){
		double[] out = new double[in.length];
		
		for (int i = 0; i < in.length; i++) {
				out[i] = S - in[i];
			}
		return out;
	}
	
	
	
    /**
     * Reshape a vector X to a symmetric matrix from dimension M x M.
     * This is a special case of Matlab's reshape() function which produces square matrices.
     * Elements are taken column-wise from X.
     * The number of elements in input must be M * M!
     * 
     * Example: 
     * 		a = [1,2,3,4]
     * 		reshape_to_symmetric_matrix(a, 2,2) =
     * 			
     * 				1	3
     * 				2	4
     * 
     * @param X The vector to be reshaped
     * @param M The length and width of the new matrix.
     * @return	Y The reshaped array, now a square matrix.
     */
	@Deprecated
    public static double[][] reshape_to_symmetric_matrix(double[] X, int M) {
    	double[][] Y = new double[M][M];
    	/*
    	 * check if length of input is dividable by channelCount
    	 */
    	if(X.length % M*M != 0)
    		throw new IllegalArgumentException("Number of Elements in X must be M*M.");
    	
    	/*
    	 * reshape the vector to  a symmetrix matrix of channel by channel dimension
    	 */
    	int c = 0,r = 0;
    	for (int i = 0; i < X.length; i++) {
    		Y[c][r] = X[i];
    		r++;
    		
    		if (i % (M-1)  == 0 && i > 0) {	// switch to next dim 
    			c++;			// continue with next column
    			r = 0;			// and begin to fill it at the beginning
    		}
    	}
    	return Y;
	}
    
    
}
