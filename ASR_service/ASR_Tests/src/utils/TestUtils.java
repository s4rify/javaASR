package utils;

import org.junit.Assert;

public class TestUtils {
	
	/**
	 * This is a delegate to the assertEquals method for arrays including a delta. 
	 * @param message
	 * @param expecteds
	 * @param actuals
	 * @param delta
	 */
	public static void assertDeepEquals(String message, double[][] expecteds, double[][] actuals, double delta) {
		
		Assert.assertEquals(message, expecteds.length, actuals.length);
		
		for (int i = 0; i < actuals.length; i++) {
			Assert.assertArrayEquals(message, expecteds[i], actuals[i], delta);
		}
	}

	
	
	public static void assertDeepEquals(double[][] expecteds, double[][] actuals, double delta) {
		
		Assert.assertEquals(expecteds.length, actuals.length);
		
		for (int i = 0; i < actuals.length; i++) {
			Assert.assertArrayEquals(expecteds[i], actuals[i], delta);
		}
	}

	
	
}
