package com.asr.sab.filter;

public class IIRFilter implements IFilter {

	private double[] x_state = {};
	private double[] y_state = {};

	IIRFilter() {}
	
	@Override
	public double[][] filter_data(double[][] data, double sRate) {
		double[] B;
		double[] A;

		switch ((int)sRate) {
		case 250:
		    double[] bs_250 = {1.73133310854258,-4.16813353295698,5.37379900844171,-5.57212564343885,4.70122651316514,
		    		-3.34208799655247,1.95045488724909,-0.766909658912075,0.233281060974836};
		    double[] as_250 = {1,-1.63849492766660,1.73987814299055,-1.83638657883456,1.39241775367980,-0.953780426622200,
		    		0.505158779550748,-0.159504514603056,0.0545278399847980};
		    B = bs_250;
		    A = as_250;
			break;

		default: // srate == 100
			double[] bs_100 = {0.9314233528641650, -1.0023683814963549, -0.4125359862018213, 
					0.7631567476327510,  0.4160430392910331, -0.6549131038692215, -0.0372583518046807, 
					0.1916268458752655,  0.0462411971592346};
			double[] as_100 = {1.0000000000000000, -0.4544220180303844, -1.0007038682936749, 
					0.5374925521337940,  0.4905013360991340, -0.4861062879351137, -0.1995986490699414, 
					0.1830048420730026,  0.0457678549234644};
			B = bs_100;
			A = as_100;
			break;
		}
		
	    
		/*
		 * filter data 
		 */
		double[][] Y = new double[data.length][data[0].length];
        for (int c = 0 ; c < data.length; c++){
        	double[] filteredChannel = filter_one_channel(B, A, data[c]);
            Y[c] = filteredChannel;
        }
		return Y;
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
	// from https://stackoverflow.com/questions/8504858/matlabs-filter-in-java
	double[] filter_one_channel(double[] b, double[] a, double[] x) {
	    final int nx = x.length;
	    final int na = a.length;
	    final int nb = b.length;
	    
	    if (x_state.length != nb-1) {
	    	x_state = new double[nb-1];
	    }
	    
	    if (y_state.length != na-1) {
	    	y_state = new double[na-1];
	    }
	
	    double[] y = new double[nx];
	    /*
	     * every loop iteration (k) treats one sample
	     */
	    for (int k = 0; k < nx; k++) {
	        y[k] = 0;
	        
	        // B
	        int i = 0;
	        int boundary_i = Math.min(k, nb-1);
	        /*
	         * go back nb-1 samples in x and compute first part of y
	         */
	        for (; i <= boundary_i; i++) {
	        	y[k] += b[i] * x[k-i];
	        }
	        
	        /*
	         * use state in ramp-up phase
	         */
	        for (; i < nb; i++) {
	        	y[k] += b[i] * x_state[x_state.length + (k-i)];
	        }
	        
	        // A
	        int j = 1;
	        int boundary_j = Math.min(k, na-1);
	        /*
	         * go back na samples in x and compute second part of y
	         */
	        for (; j <= boundary_j; j++) {
                y[k] -= a[j] * y[k-j];
	        }
	        
	        /*
	         * use state in ramp-up phase
	         */
	        for (; j < na; j++) {
                y[k] -= a[j] * y_state [y_state.length + (k-j)];
	        }
	        
	        if (Math.abs(a[0] - 1) > 1.e-9) {
	            y[k] /= a[0];
	        }
	    }
	    
	    /*
	     * update x_state: put the last nb-1 samples in there
	     */
	    for (int xi = 0; xi < x_state.length; xi++) {
			x_state[xi] = x[xi + x.length - x_state.length];
		}

	    /*
	     * update y_state: put the last nb-1 samples in there
	     */
	    for (int yi = 0; yi < y_state.length; yi++) {
			y_state[yi] = y[yi + y.length - y_state.length];
		}
	    
	    return y;
	}

	@Override
	public FilterState getState() {
		FilterState state = new FilterState();
		state.x_state = this.x_state;
		state.y_state = this.y_state;
		return state ;
	}

	@Override
	public void setState(FilterState state) {
		this.x_state = state.x_state;
		this.y_state = state.y_state;
	}

}
