package com.asr.sab.asr;

import edu.ucsd.sccn.lsl.vectord;

/**
 * Created by Sarah Blum on 11/16/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class My_LSL_Utils {

    /**
     * Helper method to store the content of a vectord type sample into an array
     *
     * @param sample sample of the LSL library type vectord
     * @return a double array containing the content of the vectord sample
     */
    public static double[] vectord_to_array(final vectord sample) {
        double[] result = new double[(int) sample.capacity()];
        for (int i = 0; i < result.length; i++) {
            result[i] = sample.get(i);
        }
        return result;
    }



    public vectord double_to_vectord(double[] data) {
        vectord v = new vectord(data.length);

        //insert a random number into v vector
        for (int k = 0; k < v.size(); k++){
            v.set(k, data[k]);
        }
        return v;
    }


    public static vectord double_to_multiplexed(double[][] cleaned_data) {
        vectord result = new vectord(cleaned_data.length * cleaned_data[0].length);
        int o = 0;
        for (int i = 0; i < cleaned_data.length; i++) {
            for (int j = 0; j < cleaned_data[0].length; j++) {
                result.set(o, cleaned_data[i][j]);
                o++;
            }

        }

        return result;
    }
}
