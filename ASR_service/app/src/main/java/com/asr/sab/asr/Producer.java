package com.asr.sab.asr;

import java.util.concurrent.BlockingQueue;
import edu.ucsd.sccn.lsl.vectord;

/**
 * Created by Sarah Blum on 11/16/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class Producer implements Runnable{

    protected BlockingQueue<double[][]> queue = null;


    public Producer(BlockingQueue<double[][]> queue) {
        this.queue = queue;
    }

    public void run() {
//        vectord samples = new vectord(ForegroundCalibService.channelCount * ForegroundCalibService.capacity);
//        try {
//            // pull samples
//            while(true) {
//                ForegroundCalibService.inlet.pull_sample(samples, 1.0);
//                queue.put(samples);
//            }
//        } catch (InterruptedException e) {
//            e.printStackTrace();
//        }
    }
}