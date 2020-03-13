/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.1
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package edu.ucsd.sccn.lsl;

public class lslAndroid {
	public static double getIRREGULAR_RATE() {
		return lslAndroidJNI.IRREGULAR_RATE_get();
	}

	public static double getDEDUCED_TIMESTAMP() {
		return lslAndroidJNI.DEDUCED_TIMESTAMP_get();
	}

	public static double getFOREVER() {
		return lslAndroidJNI.FOREVER_get();
	}

	public static int protocol_version() {
		return lslAndroidJNI.protocol_version();
	}

	public static int library_version() {
		return lslAndroidJNI.library_version();
	}

	public static double local_clock() {
		return lslAndroidJNI.local_clock();
	}

	public static vectorinfo resolve_streams(double wait_time) {
		return new vectorinfo(lslAndroidJNI.resolve_streams__SWIG_0(wait_time), true);
	}

	public static vectorinfo resolve_streams() {
		return new vectorinfo(lslAndroidJNI.resolve_streams__SWIG_1(), true);
	}

	public static vectorinfo resolve_stream(String prop, String value, int minimum, double timeout) {
		return new vectorinfo(lslAndroidJNI.resolve_stream__SWIG_0(prop, value, minimum, timeout), true);
	}

	public static vectorinfo resolve_stream(String prop, String value, int minimum) {
		return new vectorinfo(lslAndroidJNI.resolve_stream__SWIG_1(prop, value, minimum), true);
	}

	public static vectorinfo resolve_stream(String prop, String value) {
		return new vectorinfo(lslAndroidJNI.resolve_stream__SWIG_2(prop, value), true);
	}

	public static vectorinfo resolve_stream(String pred, int minimum, double timeout) {
		return new vectorinfo(lslAndroidJNI.resolve_stream__SWIG_3(pred, minimum, timeout), true);
	}

	public static vectorinfo resolve_stream(String pred, int minimum) {
		return new vectorinfo(lslAndroidJNI.resolve_stream__SWIG_4(pred, minimum), true);
	}

	public static vectorinfo resolve_stream(String pred) {
		return new vectorinfo(lslAndroidJNI.resolve_stream__SWIG_5(pred), true);
	}

	public static void check_error(int ec) {
		lslAndroidJNI.check_error(ec);
	}

}
