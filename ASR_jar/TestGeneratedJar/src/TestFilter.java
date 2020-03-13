import java.util.Arrays;

import com.asr.sab.utils.MyMatrixUtils;
import com.asr.sab.utils.Proc_Utils;

public class TestFilter {

	public static void main(String[] args) {
		double[][] testdata = new double[2][10];
		for (int i = 0; i < testdata.length; i++) {
			for (int j = 0; j < testdata[0].length; j++) {
				testdata[i][j] = j;
			}
			
		}

		double[][] filter_data = Proc_Utils.filter_data(MyMatrixUtils.transpose(testdata), 250);
		filter_data = MyMatrixUtils.transpose(filter_data);
		System.out.println(Arrays.toString(filter_data[0]));
		System.out.println(Arrays.toString(filter_data[1]));
	}

}
