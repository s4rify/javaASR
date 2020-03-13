package com.asr.sab.asr;

import java.util.concurrent.BlockingQueue;
import edu.ucsd.sccn.lsl.vectord;
import com.asr.sab.proc.*;

import static com.asr.sab.asr.ForegroundCalibService.samplingRate;

/**
 * Created by Sarah Blum on 11/16/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class Consumer implements Runnable{

    protected BlockingQueue<double[][]> queue = null;
    private final ASR_Process proc;

    public Consumer(BlockingQueue<double[][]> queue) {
        this.queue = queue;
        proc = new ASR_Process(ForegroundCalibService.state, samplingRate);
    }

    public void run() {
        try {
            while(true){
                double[][] chunk = queue.take();
                double[][] cleaned_data_chunk = proc.asr_process(chunk);
                vectord push_me = My_LSL_Utils.double_to_multiplexed(cleaned_data_chunk);
                ForegroundProcService.outlet.push_chunk_multiplexed(push_me);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }




}