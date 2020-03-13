package com.asr.sab.asr;

/**
 * Exception which is thrown whenever the buffer is overfull!
 * 
 * @author sarah
 *
 */
public class BufferOverflowException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	public BufferOverflowException(String string) {
		super(string);
	}

}
