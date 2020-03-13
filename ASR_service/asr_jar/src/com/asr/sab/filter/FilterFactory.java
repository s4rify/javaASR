package com.asr.sab.filter;

public class FilterFactory {
	
	public static IFilter getOldFilter() {
		return new FilterOldImpl();
	}
	
	public static IFilter getFilter() {
		return new IIRFilter();
	}

}
