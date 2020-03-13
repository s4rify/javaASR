package com.asr.sab.filter;

public interface IFilter {
	
	 double[][] filter_data(double[][] data, double sRate);
	 
	 FilterState getState();
	 
	 void setState(FilterState state);
}
