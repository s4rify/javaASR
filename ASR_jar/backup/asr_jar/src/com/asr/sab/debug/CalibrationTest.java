package com.asr.sab.debug;

import static com.asr.sab.utils.MyMatrixUtils.*;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;

import com.asr.sab.cal.ASR_Calibration;

import java.util.Arrays;
import java.util.stream.DoubleStream;


/**
 * Created by Sarah Blum on 9/29/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class CalibrationTest {

    public static void main (String[] args){

    	//ASR_Calibration cal = new ASR_Calibration(buffer.getChannelCount());
    	//cal.asr_calibrate(buffer, sR);

        /* 
         * compose input X
         */
    	double[][] X = new double[2][20];

        for (int i = 0; i < 20; i++) {
            double addend = 0.01 * i;
            X[0][i] = 1.01 + addend;
            X[1][i] = 4.01 + addend;
        }
        /*
         * filter input. -> filter is tested, omit here
         */

        /*
         * calculate sample covariance matrix and mixing matrix
         * [this is tested and works just like in matlab <3]
         */
        int blocksize = 5;
		double[][] M = calculate_mixingMatrix(X, blocksize);
        
		/*
		 * calculate threshold matrix T
		 */
		double window_overlap = 0.66;
		double window_len = 0.5;
		double srate = 100;
		double N = Math.round(window_len*srate);
		//TODO debug value
			N = 5;
		double[][] T = calculate_thresholdMatrix(X, M, window_overlap, N);
        
		

    }



}
