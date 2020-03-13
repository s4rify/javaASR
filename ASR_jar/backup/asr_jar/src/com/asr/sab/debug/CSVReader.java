package com.asr.sab.debug;

/**
 * Created by Sarah Blum on 9/25/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Pattern;

/*
 * Helper class that reads a csv file and returns a double array for further processing.
 */
public class CSVReader {

    private static final Pattern COMMA_OR_NEWLINE_DELIMITER = Pattern.compile("[,\\n\\r]"); //(, | \s)
    private static final String DEFAULT_CHARSET_NAME = "UTF-8";

    public static double[] readDoublesFromFile(String filename) {
        return readDoublesFromFile(filename, DEFAULT_CHARSET_NAME);
    }

    public static double[] readDoublesFromFile(String filename,  String charsetName) {
        List<Double> list = new LinkedList<>();

        try(Scanner s = new Scanner(new File(filename), charsetName)) {
            s.useDelimiter(COMMA_OR_NEWLINE_DELIMITER);
            while(s.hasNextDouble()) {
                list.add(s.nextDouble());
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        double[] result = new double[list.size()];
        int i = 0;
        for(double value : list) result[i++] = value;

        return result;
    }

}
