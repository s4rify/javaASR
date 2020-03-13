package com.asr.sab.filter;

import static org.junit.Assert.assertArrayEquals;

import java.util.Arrays;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class UT_ASR_IIRFilter {

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void testFilter_data_with_state() {
		IIRFilter filter = new IIRFilter();
		
		double[][] unfiltered = createUnfilteredData(1);
		double[] expected = filter.filter_data(unfiltered, 250)[0];
		
		double[][] firstPart = { Arrays.copyOfRange(unfiltered[0], 0, 50) };
		double[][] secPart = { Arrays.copyOfRange(unfiltered[0], 50, 100) };
		
		IIRFilter filter2 = new IIRFilter();
		
		double[] f1 = filter2.filter_data(firstPart, 250)[0];
		double[] f2 = filter2.filter_data(secPart, 250)[0];
		
		double[] joined = Arrays.copyOf(f1, 100);
		System.arraycopy(f2, 0, joined, 50, 50);
		
		assertArrayEquals("chunked filtering is different", expected, joined, 0.01);
	}

	private double[][] createUnfilteredData(int channels) {
		double[][] unfiltered = new double[channels][100];
		for (int i = 0; i < unfiltered.length; i++) {
			for (int j = 0; j < unfiltered[0].length; j++) {
				unfiltered[i][j] = j;
			}
		}
		return unfiltered;
	}

}
