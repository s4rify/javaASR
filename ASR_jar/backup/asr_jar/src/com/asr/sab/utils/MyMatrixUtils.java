package com.asr.sab.utils;

import static com.asr.sab.utils.MyMatlabUtils.bsxfun;
import static com.asr.sab.utils.MyMatlabUtils.reshape_2_to_3;
import static com.asr.sab.utils.MyMatlabUtils.reshape_3_to_2;

import java.awt.geom.FlatteningPathIterator;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;
import java.util.stream.IntStream;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DiagonalMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import com.asr.sab.cal.ASR_Calibration;

import la.matrix.DenseMatrix;
import la.matrix.Matrix;
import la.vector.Vector;
import ml.utils.Matlab;

/**
 * Created by Sarah Blum on 9/29/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class MyMatrixUtils {

    private static final double median_percentile = 50.0;
	private static final int maxiter = 500;

	/**
     * Why is nothing ever done in Java.
     *
     * @param array the array to transpose
     * @return  the transposed array.
     */
    public static double[][] transpose (double[][] array) {
        if (array == null || array.length == 0) throw new IllegalArgumentException("The array is emtpy!");
        int width = array.length;
        int height = array[0].length;
        double[][] array_new = new double[height][width];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                array_new[y][x] = array[x][y];
            }
        }
        return array_new;
    }
    
    
	/**
     * Why is nothing ever done in Java.
     *
     * @param array the array to transpose
     * @return  the transposed array.
     */
    public static int[][] transpose(int[][] array){
        if (array == null || array.length == 0) throw new IllegalArgumentException("The array is emtpy!");
        int width = array.length;
        int height = array[0].length;
        int[][] array_new = new int[height][width];
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                array_new[y][x] = array[x][y];
            }
        }
        return array_new;
    }
    

    /**
     * Matrix- Utils: get absolute matrix from given matrix m by
     * taking the abs value of every element of the matrix.
     * 
     * @param m The matrix from which we need the absolute values.
     * @return abs(m).
     */
    public static double[][] matrix_abs(double[][] m) {
        int rowLen = m.length;
        int colLen = m[0].length;
        double[][] m_abs = new double[rowLen][colLen];
        for (int i = 0; i < rowLen ; i++) {
            for (int j = 0; j < colLen ; j++) {
                m_abs[i][j] = Math.abs(m[i][j]);
            }
        }
        return m_abs;
    }



    /**
     * Matrix multiplication method like in the books. This takes
     * pairwise elements, multiplies them and then adds them up.  The
     * resulting matrix will be much smaller than the inputs!
     * 
     * @param m1 Multiplicand
     * @param m2 Multiplier
     * @return Element-wise product
     */
    public static double[][] array_multiplication(double[][] m1, double[][] m2) {
        int m1ColLength = m1[0].length; // m1 columns length
        int m2RowLength = m2.length;    // m2 rows length
        if(m1ColLength != m2RowLength) throw new IllegalArgumentException("Dimensions must match.");
        int mRRowLength = m1.length;    // m result rows length
        int mRColLength = m2[0].length; // m result columns length
        double[][] mResult = new double[mRRowLength][mRColLength];
        for(int i = 0; i < mRRowLength; i++) {         // rows from m1
            for(int j = 0; j < mRColLength; j++) {     // columns from m2
                for(int k = 0; k < m1ColLength; k++) { // columns from m1
                    mResult[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
        return mResult;
    }


    /**
     * Implementation of the difference equation of a
     * Direct Form II Transposed IIR filter.
     *
     * @param b
     * 		The array of b-coefficients
     * @param a
     * 		The array of a-coefficients, where the first value must be 1
     * @param X
     * 		The signal to filter
     * @return
     * 		A double array containing the filtered signal
     */
    public static double[] filter_one_channel (double[] b, double[] a, double[] X) {
        int n = a.length;
        double[] z = new double[n];
        double[] Y = new double[X.length];
        for (int m = 0; m < Y.length; m++) {
            Y[m] = b[0] * X[m] + z[0];
            for (int i = 1; i < n; i++) {
                z[i-1] = b[i] * X[m] + z[i] - a[i] * Y[m];
            }
        }
        return Y;
    }
    
    

    /**
     * Calculates the mixing matrix on the input data. The input data is the filtered channel x samples.
     * This method adds up #blocksize elements and normalizes the content of the expanded covariance
     * matrix U after all the input has been computed.
     *
     *
     * @param input filtered [channel][samples]
     * @param blocksize how many samples are going to be aggregated
     *
     * @return the mixing matrix.
     */
    public static double[][] calculate_mixingMatrix (double[][] input, int blocksize){
        /*
        * for k=1:blocksize
        *      range = min(S,k:blocksize:(S+k-1));
        *      U = U +reshape(bsxfun(@times,reshape(X(range,:),[],1,C),reshape(X(range,:),[],C,1)),size(U));
        * end
        * TODO we assume that samples is dividable by blocksize, make sure that it is!
        */
        int samples = input[0].length;
        int channels = input.length;

        double[][] U = compute_sample_cov_matrix(input, blocksize, samples, channels);
        //System.out.println(Arrays.deepToString(U).replace("], ", "]\n"));

        /*
         * normalize U by dividing by blocksize.
         * we have to do this here because blocksize != samples. 
         */
        for (int i = 0; i < U.length; i++) {
			for (int j = 0; j < U[i].length; j++) 
				U[i][j] /= blocksize;
		}
        
        
        /*
         * channel medians over all blocks. each channel gets a median which is then compared to
         * a global median to see whether the channel is way off
         */
        double[] UM = geometric_mean(U);
        // TODO is this the same?
        // new GeometricMean().setData(U[1]);
        
        
        /*
         * square root of matrix median
         * 1) reshape in CxC
         * 2) make RealMatrix
         * 3) compute squareRoot on RealMatrix: 
         * 		X = sqrtm(A) is the principal square root of the square matrix A. 
    	 * 		That is, X*X = A.
         */
        //Matlab.reshape(new DenseMatrix(UM), channels, channels);
        //double[][] sUM =  MyMatlabUtils.reshape_to_symmetric_matrix(UM, channels);
        double[][] sUM = MyMatlabUtils.reshape_1_to_2(UM, channels);
        RealMatrix rm = new Array2DRowRealMatrix(sUM);
        EigenDecomposition e = new EigenDecomposition(rm);
        double[][] mixingMatrix = e.getSquareRoot().getData();
        
        /*
         * OMG, it is happening.
         */
        return mixingMatrix;
    }
    

    /**
     * Compute the covariance matrix of all channel combinations with the possibility of aggregating
     * the samples over #blocksize samples.
     * 
     * @param input the data with which we want to compute the covariance
     * @param blocksize the blocksize over which we want to average. If blocksize == samples, all
     * 			samples from the channel are treated together
     * @param samples how many samples does one channel have
     * @param channels	how many channels does the input have
     * @return	U, the covariance super - matrix which contains all covariance matrices for all channels.
     * 			It will be [samples][channels*channels] in size! 
     */
    public static double[][] compute_sample_cov_matrix(double[][] input, int blocksize, int samples, int channels) {
    	if(blocksize == samples) throw new IllegalArgumentException("The blocksize must be smaller than the samples!");
    	double[][] U = new double[samples/blocksize][channels * channels];

    	for (int k = 0; k < blocksize ; k++) {
    		for (int block = 0, r = k; r < samples; r += blocksize, block++) {

    			// all channel combinations
    			for (int c1 = 0; c1 < channels; c1++) {
    				for (int c2 = 0; c2 < channels ; c2++) {
    					// multiply channel elements
    					// U += reshape(bsxfun(@times,reshape(X,1,C,[]),reshape(X,C,1,[]))
    					U[block][c1*channels+c2] += input[c1][r] * input[c2][r];
    				}
    			}
    		}
    	}
    	return U;
    }

   
    
    
    /**
     * Calculate the threshold matrix which will contain one threshold for every channel-vector.
     * Threshold is defined over multiplies of sigma.
     * 
     * @param X filtered input.
     * @param M mixing matrix
     * @return	threshold matrix with dimensions from X.
     */
    public static double[][] calculate_thresholdMatrix(double[][] X, double[][]M, double window_overlap, double N) {
    	//int channel = X.length;
    	//int samples = X[0].length;
    	
    	/*
    	 * Eigenvalue decomposition of mixing matrix M.
    	 * This gives the same results as eig() in Matlab, except for the order!
    	 * 
    	 * [V,D] = eig(M);
    	 */
        EigenDecomposition eig_M = new EigenDecomposition(new Array2DRowRealMatrix(M));
        RealMatrix Xr = MatrixUtils.createRealMatrix(transpose(X));
        
        /*
         * X = abs(X*V);
         * Note: XV has the correct elements in the columns but the columns are swapped (compared to matlab)
         */
        double[][] XV = matrix_abs(transpose(Xr.multiply(eig_M.getV()).getData()));
        
        /*
         * calculate mean and standard deviation from rms, statistics for each channel
         * remember: this is a shortcut-calculation. but the values are in the same range
         *   rms_orig = X(:,c).^2; % column-wise element squaring over time
         *   rms_orig = sqrt(sum(rms_orig(bsxfun(@plus,round(1:N*(1-window_overlap):S-N),(0:N-1)')))/N);
         *   RMS(c,:) = rms_orig;
         */
        double[][] RMS = calculate_RMS(XV, window_overlap, N);
        
        double[] mu = new double[RMS.length], sig = new double[RMS.length];
        for (int c = 0; c < RMS.length; c++) {
        	DescriptiveStatistics ds = new DescriptiveStatistics(RMS[c]);
        	mu[c] = ds.getMean();
        	sig[c] = ds.getStandardDeviation();
		}
        
        
        /*
         * T = diag(mu + cutoff * sig) * V';
         */
        int cutoff = 5;
        // multiply every element by cutoff
        sig = Arrays.stream(sig).map(i -> i * cutoff).toArray();
        for (int m = 0; m < mu.length; m++) {
			mu[m] += sig[m];
		}
        /*
         *  create diagonal matrix which contains the given data on the diagonal and 0s else
         *  this has dimensions channel x channel
         *  T = diag(mu + cutoff*sig)*V';
         */
        double[][] T = MyMatrixUtils.transpose(new DiagonalMatrix(mu).multiply(eig_M.getV().transpose()).getData());
        return T;
    }



	/**
     * Implementation of the geometric median.
     * this is basically finding a median vector in a matrix full  of vectors; this is the equivalent of 
     * the median in a set of single values, only for vectors.
     * 
     * Smart description for smart people: Calculate the geometric median for a set of observations 
     * (mean under a Laplacian noise distribution). This is using Weiszfeld's algorithm.
     * 
     * @param X the matrix in which to find a median vector (e.g. one column)
     * @return	Y a vector which contains the median vector (here a sample across all channels).
     */
    public static double[] geometric_mean(final double[][] X) {
    	/*
    	 * when called, U is handed over (X=U), and it has [samples][channels], so it is
    	 * the other way around. this is why we transpose it after a few lines
    	 */
    	
    	final int channelPairCount = X[0].length;
    	final int blockCount = X.length;
    	
    	// transpose in.
    	// before: in[blocks][channelpairs]
    	// after: in[channelpairs][blocks]
    	double[][] Xt = MyMatrixUtils.transpose(X);
    	
    	/*
    	 *   y = median(X);
    	 */
    	double[] mediansPerChannelPair = new double[channelPairCount];
    	for (int i = 0; i < channelPairCount; i++) {
    		mediansPerChannelPair[i] = StatUtils.percentile(Xt[i], median_percentile);
    	}
    	
    	double[] y = mediansPerChannelPair;
    	for (int iteration = 0; iteration < maxiter; iteration++) {
			/*
			 * invnorms = 1./sqrt(sum( bsxfun(@minus,X,y) .^2,2));
			 * 
    		 * 
    		 * X: Input matrix of dimensions blocks x channelpairs
    		 * y: (Intermediate) result vector, one entry per channelpair
    		 *    Is repeatedly updated to converge to the desired result.
    		 * 
    		 * Calculate differences:
    		 * b: block index
    		 * 
    		 * for each b, calculate invnorm[b] as follows:
    		 * 
    		 *   difference[cp] = X[b][cp] - y[cp]  // bsxfun(@minus,X,y)
    		 * 
    		 *   squares[cp] = dfference[cp] * difference[cp];
    		 * 
    		 *   invnorm[b] = sum(squares)
    		 *   
    		 * end invnorm calc.
    		 */ 
    		double[] invnorm = new double[blockCount];
    		double sumInvnorms = 0.0;
    		double[] sumOfProducts = new double[channelPairCount];

    		for (int b = 0; b < blockCount; b++) {
    			double sum_squares = 0.0; 
    			for (int cp = 0; cp < channelPairCount; cp++) {
					double diff = X[b][cp] - y[cp];
					sum_squares += diff * diff;
    			}
    			/*
    			 *  sum(invnorms) 
    			 */
    			sumInvnorms += invnorm[b] = 1.0 / Math.sqrt(sum_squares);
			
    			/*
    			 *  sum(bsxfun(@times,X,invnorms))
    			 *  funny side-effect: this sums up all the values a little bit in each iteration.
    			 *  Each sum gets one summand per iteration, the sum is only done after the outer loop
    			 *  is finished.    			 
    			 */
    			for (int cp = 0; cp < channelPairCount; cp++) {
    				sumOfProducts[cp] += X[b][cp] * invnorm[b];
    			}
    		}
    		
    		/*
    		 *  sum(bsxfun(@times,X,invnorms)) / sum(invnorms)
    		 *  where sum(bsxfun(..)) is the result of the previous loop which is
    		 *  contained in sumOfProducts.
    		 */
//    		assert y.length == channelPairCount : "Wrong dimension: y should have an entry for each pair of channels.";
    		for (int cp = 0; cp < channelPairCount; cp++) {
    			y[cp] = sumOfProducts[cp] / sumInvnorms;
    		}
		}
    	return y;
    }


    /**
     * Calculates the root mean square (RMS) for an array of double values.
     * 
     * first part: RMS = sqrt(sum(X(:,c).^2)/N)
     *
     * second part: sqrt(sum(RMS(bsxfun(@plus,round(1:N*(1-window_overlap):S-N),(0:N-1)')))/N);
     * 
     * @param values the array which contains the values of whom we calculate the RMS
     * @return the RMS value for this array. The output will have dimensions [channels][window.len]
     */
    public static double[][] calculate_RMS (double[][] X, double window_overlap, double N) {
    	/*
    	 * first part: rms = X(:,c).^2;
    	 */
    	double[][] sumSquareRoot = new double[X.length][X[0].length];
    	// major dim: columns
    	for (int c = 0; c < X.length; c++) {
    		for (int r = 0; r < X[0].length; r++) {
    			// sum up all the squares; this is rms = X(:,c).^2
    			sumSquareRoot[c][r] = X[c][r]*X[c][r]; 
    		}
    	}

    	/*
    	 * second part: sqrt(sum(RMS(win))/N);
    	 */
    	
    	/*
    	 *  this fills win: 
    	 *    w1 = round(1:N*(1-window_overlap):S-N)
    	 *    w2 = (0:N-1)'
    	 *    win = bsxfun(@plus,range1,range2)
    	 */
    	int[][] win = fill_index_window(X, window_overlap, N);
    	double[][] RMS = new double[X.length][win.length];
    	double[] interim = new double[win.length];
    	for (int c = 0; c < X.length; c++) {
    		for (int i = 0; i < win.length; i++) {
    			for (int j = 0; j < win[0].length; j++) {
    				// use the index window to take out some values from every channel
    				interim[i] += sumSquareRoot[c][win[i][j]]/N;
    			}
    		}
    		// do sqrt(sum(rms)) as soon as we have all values from one channel
    		for (int r = 0; r < interim.length; r++) {
    			RMS[c][r] = Math.sqrt(interim[r]);
    			interim[r] = 0;
    		}
    	}
    	// the result is a vector of a couple of values per channel! we do not get one rms value per channel
    	// (as you would think) but several (determined by the size of the index window)
    	return RMS;
    }

    
    /**
     * In Matlab, for every channel, the rms is calculated for a range of indices that are calculated as follows:
     *   range1 = round(1:N*(1-window_overlap):S-N)
     *   range2 = (0:N-1)'
     * This means that we get a whole range of indices that are later used to chose some squared values
     * per channel, not just average over all of them. This is comparable to the calculation of the 
     * mixing matrix and its sample covariance matrices, where we get a matrix per sample window. Equally,
     * here we get several indices per channel, so that we generate the root mean square value over some
     * samples from every channel.
     *   
     *    
     * @param X
     * @param window_overlap
     * @param N
     * @return
     */
	private static int[][] fill_index_window(double[][] X, double window_overlap, double N) {
		int S = X[0].length;
		int range1_len = (int)Math.round((S-N)/(N*(1-window_overlap)));
		int range2_len = (int)N;
		double[] range1 = new double[range1_len];
		double[] range2 = new double[range2_len];
		
		double val = (N*(1-window_overlap));
		for (int i = 1; i < range1.length; i++ ) { // let first value = zero
			range1[i] += range1[i-1] + val;
		}
		// after all the computations have been completed, round to nearest int
		for (int i = 0; i < range1.length; i++) range1[i] = Math.round(range1[i]);
		
		for (int i = 0; i < N; i++) range2[i] = i;
		
		int[][] win =  new int[range1_len][range2_len]; 
		// bsxfun(@plus,range1,range2)
		for (int r1 = 0; r1 < range1_len; r1++) {
			for (int r2 = 0; r2 < range2_len; r2++) {
				win[r1][r2] = (int) (range1[r1] + range2[r2]);
			}
		}
		return win;
	}

    
    /**
     * Little helper method which expands a 2-dim array to a 3-dim array (with
     * one singleton dimension).
     * 
     * @param 	in The array we want to expand
     * @return	new double[][][] {in}
     */
    public static double[][][] extend_3(double[][] in){
    	return new double[][][] {in};
    }
    
    

	/**
	 * This method corresponds to this line in Matlab: 
	 * [Xcov,state.cov] = moving_average(N,reshape(bsxfun(@times,reshape(X,1,C,[]),reshape(X,C,1,[])),C*C,[]),state.cov);
	 * It computes a smoothing average over the covariance matrix.
	 * 
	 * 
	 * @param N			N = round(windowlen*srate);
	 * @param X			input which is filtered and padded with the carry content 
	 * @param prevCov	Optional input of a previous covariance matrix. If this is the first call to
	 * 					the method, omit the input. If we have been processing before, provide the previous 
	 * 					covariance matrix here. The content of prevCov is simulated (by padding zeros to X) if
	 * 					this parameter is empty.
	 * 
	 * @return			A smoothed covariance matrix with the same dimensions as the input X.
	 */
	public static double[][] moving_average(int N, double[][] X, ASR_Calibration state, double[] ... prevCov){
		/* 
		 * transfrom input because why not:
		 * U = reshape(bsxfun(@times,reshape(X,1,C,[]),reshape(X,C,1,[])),C*C,[])
		 */
		int C = X.length;
        int samples = X[0].length;
        
        // f = (reshape(X,1,C,[]))
        double[][][] f = reshape_2_to_3(X, samples, 1, C);
        
        // g = (reshape(X,C,1,[]))
        double[][][] g = reshape_2_to_3(X, samples, C, 1);
        
        // m = bsxfun(@times,f,g)
        final DoubleBinaryOperator TIMES = (double a, double b) -> a * b;
        double[][][] m = bsxfun(f, g, TIMES);
        
        // U = reshape(m,C*C,[])
        double[][] U = reshape_3_to_2(m, samples, C*C);
        
		/*
		 * check if we have a previous Xcov matrix in the state object
		 * and handle the shit out of it:
		 * - prepend the old Xcov matrix to U
		 * - if we don't have a previous cov matrix: initialize fake-state.cov  and prepend that
		 * 
		 */
    	if(prevCov.length == 0) {
    		//prevCov = zeros(size(X,1),N)
    		prevCov = new double[N][U[0].length];
    	}
    	
    	/*
    	 * prepend Xcov: Y = [prevCov X]; 
    	 */
    	double[][] Y = Matlab.cat(1, new DenseMatrix(prevCov), new DenseMatrix(U)).getData();
    	// M = size(Y,2);
        int M = Y.length-1;
		
        /*
		 * compute some index shit because why not
		 * I = [1:M-N; 1+N:M];
		 */
        int[] range1 = IntStream.rangeClosed(0, M-N).toArray();
        int[] range2 = IntStream.rangeClosed(0+N, M).toArray();
		int[][] I = {range1, range2}; 
		I = transpose(I);
		
		/*
		 * this is a matrix in matlab: S = [-ones(1,M-N); ones(1,M-N)]/N;
		 */
		double S1 = 1.0/N;
		double S2 = -1.0/N;
		
		/*
		 * Use I to index Y: Y(:,I(:)).
		 * By doing this, we take some samples (columns) from Y and smooth them by multiplying them
		 * with the divisor stored in S (which in Matlab is a matrix for itself).
		 * This is basically smoothing.
		 * 
		 */
		double[][] YI = new double[I.length*2][Y[0].length];
		int n1 = 0;
		double divisor = S1;
		for (int i = 0; i < I.length; i++) {
			
			for (int j = 0; j < I[0].length; j++) {
				for (int n = 0; n < Y[0].length; n++) {
					YI[n1][n] = Y[I[i][j]][n];
					YI[n1][n] *= divisor;
				}
				n1++;
				/*
				 * multiply the columns alternatingly with the content of S
				 * bsxfun(@times,YI, S(:)') where 
				 * S contains 1.0/N and -1.0/N in its columns
				 */
				if (n1 %2 != 0) {
					divisor = S2;
				} else {
					divisor = S1;
				}
			}
		}

		YI = cumsumfun(YI);
		
		// keep every second column.
		double[][] out = new double[YI.length/2][YI[0].length];
		for (int i = 1, i1 = 0; i1 < YI.length/2; i+=2, i1++) {
			for (int j = 0; j < YI[0].length; j++) {
				out[i1][j] = Math.abs(YI[i][j]);
			}
		}
		
		
		/*
		 * fill the cov field of the state object :
		 * zf = -(X(:,end)*N-Y(:,end-N+1))
		 * zg = Y(:,end-N+2:end)  
		 * Zf = [zf, zg];
		 */
		//int[] rangeCols = IntStream.rangeClosed(Y.length-N , Y.length-1).toArray();
		double[][] zg = MatrixUtils.createRealMatrix(Y).getSubMatrix(Y.length-N, Y.length-1, 0, Y[0].length-1).getData();
		
		double[] endX = MatrixUtils.createRealMatrix(out).scalarMultiply(-1).scalarMultiply(N).getRow(out.length-1);
		double[] endY = MatrixUtils.createRealMatrix(Y).getRow(Y.length - N);
		double[] zf = new double[endX.length];
		for (int i = 0; i < zf.length; i++) {
			zf[i] = endX[i] - endY[i];
		}
		
		zg[0] = zf;
		state.cov = zg;
		
		return out; // there you go.
	}
	

	
	/**
	 * Y = cumsum(X) computes the cumulative sum along the first non-singleton
     * dimension of X. Y is the same size as X.
     * This (like in Matlab) computes the cumulative sum row-wise, that is, we take
     * the same element from every column and sum them up. 
     * 
	 * @param in	the array in which the elements are supposed to be summed up
	 * @return 		an array with the same dimensions as in where the elements are summed
	 * 				up while keeping the interim results.
	 */
	public static double[][] cumsumfun(double[][] in){
				
		int dx = in[0].length;
		int dy = in.length;


		double[][] out = new double[dy][dx];

		// copy first line
		for (int x=0; x < dx; x++) {
			out[0][x] = in[0][x];
		}


		// special if only one line (e.g. cumsum([1,2,3]) -> [1,3,6]
		if (dy==1){
			for (int x = 1; x < dx; x++){
				out[0][x] = out[0][x] + out[0][x-1];
			}
		}

		// start with second line. start with y=1 because we look back
		for (int y = 1; y < dy; y++){
			for (int x = 0; x < dx; x++){
				out[y][x] = in[y][x] + out[y-1][x];
			}
		} 
		return out;
	}
	
	
		
	/**
	 * Really?
	 * 
	 * TODO find a better way to do this. 
	 * 
	 * @param X
	 * @param scalar
	 * @return
	 */
	public static double[][] divide_by_scalar(double[][] X, int scalar){
		double[][] out = new double[X.length][X[0].length];
		for (int i = 0; i < X.length; i++) {
			for (int j = 0; j < X[0].length; j++) {
				out[i][j] = X[i][j]/scalar;
			}
		}
		return out;
	}
	
	
	
    /**
     * 
     * TODO THIS IS NOT TESTED SO FAR!!
     * 
     * Reshape the matrix into a new form. New size should have the same number of elements as current size.
     *
     * @param m    new number of rows
     * @param n    new number of columns
     * @return array reshaped values
     */
    public static double[][] reshape(double[][] A, int m, int n) {
        int origM = A.length;
        int origN = A[0].length;
        if(origM*origN != m*n){
            throw new IllegalArgumentException("New matrix must be of same area as matrix A");
        }
        double[][] B = new double[m][n];
        double[] A1D = new double[A.length * A[0].length];

        int index = 0;
        for (double[] aA : A) {
            for (int j = 0; j < A[0].length; j++) {
                A1D[index++] = aA[j];
            }
        }

        index = 0;
        for(int i = 0;i<n;i++){
            for(int j = 0;j<m;j++){
                B[j][i] = A1D[index++];
            }
        }
        return B;
    }
    
    
    
    /*    static class Range implements Iterable<Integer> {

    final int start, end;

    public Range(int start, int end) {
        this.start = start;
        this.end = end;
    }

    public static Range range(int start, int end) {
        return new Range(start, end);
    }

    @Override
    public Iterator<Integer> iterator() {
        return IntStream.rangeClosed(start,end).iterator();
    }
}*/

}
