# Artifact Subspace Reconstruction Android Project

Artifact Subspace Reconstruction (ASR, Mullen et al., 2015) is an adaptive artifact correction method which repeatedly computes a principal component analysis on covariance matrices of the incoming data. It is a supervised method, which needs clean calibration data to estimate statistics for clean reference data. After a calibration phase, the online processing begins. During the processing, chunks of data are checked for artifactual segements which are then reconstructed. See [0] for more details.

## ASR Service
This porject contains 
1. a library which implements the signal processing, called asr_jar
  The source files for the library lie in asr_jar/src/. These source files make up the core of the ASR library, they are structured as follows:
  - /cal contains the Calibration class which is used for the calibration and needs approximately 1 minute of resting calibration data
  - /proc contains the processing class which needs the information computed during the processing. During the online cleaning, it gets small chunks of possibly unclean EEG data which is then cleaned and returned.

2. an Android Service, which provides data input and output (e.g. via LSL) and uses the library for the online cleaning
  The source files for the Android Service lie in /app/src. This Service has a dependency to the asr library which is referenced in /app/libs. The source files of the library are only part of this project for convenience reasons, there is no dependency to the source files themselves.

3. The JUnit test files in ASR_Tests


# Example Usage
After initializing the objects for the calibration and the processing, a loop repeatedly gets data (e.g. from a stream or a file) for the processing.

```
for (int i = 0; i < calibData.length; i++) {
  // read in some calibration data, here via CSV
  double[][] calibData = CSVReader.readDoublesFromFile2d(cfile.getAbsolutePath());
  
  // read in some processing data, here via CSV, can also be LSL of course
  double[][] procData = CSVReader.readDoublesFromFile2d(pfile.getAbsolutePath());
  
  // pre-allocate an output array
  double[][] out = new double[procData.length][procData[0].length];
    
    /*
     *  initialize the calibration and processing objects  
     */
  ASR_Calibration state = new ASR_Calibration(calibData);
  
  ASR_Process proc = new ASR_Process(state, srate);
      
  /*
   * repeatedly get chunk of data for processing
   */
  int last_i = 1;
  for (int k = 1000; k < procData[0].length; k+=1000) {
    double[][] currentChunk = getChunk(procData, last_i, k);
    double[][] tmp = proc.asr_process(currentChunk);
    out = setChunk(tmp, out, last_i, k);
    last_i = k;
  }
}
```

## License
Copyright 2018,  Sarah Blum

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


# References
[0] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4710679/