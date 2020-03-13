package com.asr.sab.asr;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.support.annotation.Nullable;
import android.util.Log;
import android.widget.Toast;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import edu.ucsd.sccn.lsl.channel_format_t;
import edu.ucsd.sccn.lsl.stream_info;
import edu.ucsd.sccn.lsl.stream_outlet;


/**
 * Created by Sarah Blum on 9/20/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class ForegroundProcService extends Service {
    private String LOG_TAG = "ForegroundProcService";
    public static stream_info out_info;
    public static stream_outlet outlet;
    public static int online_capacity = 1024;

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent.getAction().equals("android.intent.action.START_PROC")) {
            Toast.makeText(this, "Start Calibration", Toast.LENGTH_SHORT).show();
            Log.i(LOG_TAG, "Received Start Processing Foreground Intent");


            out_info = new stream_info("CleanedData","EEG",ForegroundCalibService.channelCount,
                    250, channel_format_t.cf_double64,"cleaneddata12345");
            outlet = new stream_outlet(out_info);
            /*
             * We have a producer-consumer problem here. The produces takes samples from the LSL
             * stream and puts them into the BlockingQueue. The Consumer is the class which
             * is performing the data processing.
             *
             * Question to self: what happens to the next 128 data points? are they stored
             * somewhere? Are they put into the queue as well while the consumer takes out the
             * last ones?
             */
            BlockingQueue<double[][]> queue = new ArrayBlockingQueue<>(online_capacity); // half a second worth of data
            // this one pulls the samples from the LSL stream
            Producer producer = new Producer(queue);
            // this one processes the samples and pushes them out
            Consumer consumer = new Consumer(queue);
            new Thread(producer).start();
            new Thread(consumer).start();


        } else if (intent.getAction().equals("android.intent.action.STOP_CAL")){
            Toast.makeText(this,"Stopped ASR Service",Toast.LENGTH_SHORT).show();
            Log.i(LOG_TAG, "Received Stop Foreground Intent");
            stopForeground(true);
            stopSelf();
        }

        return START_STICKY;
    }




    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

}
