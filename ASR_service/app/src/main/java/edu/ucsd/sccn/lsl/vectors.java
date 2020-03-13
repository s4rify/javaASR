/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.1
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package edu.ucsd.sccn.lsl;

public class vectors {
	private long swigCPtr;
	protected boolean swigCMemOwn;

	protected vectors(long cPtr, boolean cMemoryOwn) {
		swigCMemOwn = cMemoryOwn;
		swigCPtr = cPtr;
	}

	protected static long getCPtr(vectors obj) {
		return (obj == null) ? 0 : obj.swigCPtr;
	}

	protected void finalize() {
		delete();
	}

	public synchronized void delete() {
		if (swigCPtr != 0) {
			if (swigCMemOwn) {
				swigCMemOwn = false;
				lslAndroidJNI.delete_vectors(swigCPtr);
			}
			swigCPtr = 0;
		}
	}

	public vectors() {
		this(lslAndroidJNI.new_vectors__SWIG_0(), true);
	}

	public vectors(long n) {
		this(lslAndroidJNI.new_vectors__SWIG_1(n), true);
	}

	public long size() {
		return lslAndroidJNI.vectors_size(swigCPtr, this);
	}

	public long capacity() {
		return lslAndroidJNI.vectors_capacity(swigCPtr, this);
	}

	public void reserve(long n) {
		lslAndroidJNI.vectors_reserve(swigCPtr, this, n);
	}

	public boolean isEmpty() {
		return lslAndroidJNI.vectors_isEmpty(swigCPtr, this);
	}

	public void clear() {
		lslAndroidJNI.vectors_clear(swigCPtr, this);
	}

	public void add(short x) {
		lslAndroidJNI.vectors_add(swigCPtr, this, x);
	}

	public short get(int i) {
		return lslAndroidJNI.vectors_get(swigCPtr, this, i);
	}

	public void set(int i, short val) {
		lslAndroidJNI.vectors_set(swigCPtr, this, i, val);
	}

}
