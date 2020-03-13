package com.asr.sab.debug;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class CsvWriter {

	private Path path;
	
	public CsvWriter(Path path) {
		this.path = path;
	}

	public CsvWriter(String path) {
		this.path = Paths.get(path);
	}
	
	public void writeArray(double[] vector, String basename) {
		writeArrayToPath(vector, path, basename);
	}

	public static void writeArrayToPath(double[] vector, Path path, String basename) {
		File file = path.resolve(basename + ".csv").toFile();
		try (PrintWriter pw = new PrintWriter(file, "ISO-8859-1")) {
			for (int i = 0; i < vector.length; i++) {
				pw.println(vector[i]);
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	public static void writeMatrixToPath(double[][] matrix, Path path, String basename) {
		File file = path.resolve(basename + ".csv").toFile();
		try (PrintWriter pw = new PrintWriter(file, "ISO-8859-1")) {
			for (double[] row : matrix) {
				for (double value : row) {
					pw.print(value + ",");
				}
				pw.println();
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}
}
