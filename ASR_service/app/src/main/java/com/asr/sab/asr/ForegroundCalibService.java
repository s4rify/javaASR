package com.asr.sab.asr;

import android.app.Notification;
import android.app.Service;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.IBinder;
import android.support.v4.app.NotificationCompat;
import android.util.Log;
import android.widget.RemoteViews;
import android.widget.Toast;


import edu.ucsd.sccn.lsl.lslAndroid;
import edu.ucsd.sccn.lsl.stream_info;
import edu.ucsd.sccn.lsl.stream_inlet;
import edu.ucsd.sccn.lsl.vectord;
import edu.ucsd.sccn.lsl.vectorinfo;
import com.asr.sab.cal.ASR_Calibration;


/**
 * Created by Sarah Blum on 9/20/17
 * Carl von Ossietzky Univesity of Oldenburg.
 */

public class ForegroundCalibService extends Service{

    private static final String LOG_TAG = "ForegroundCalibService";

    SampleBuffer buffer;
    public static int samplingRate;
    public static int capacity =  750; // let it be 3 seconds for testing. afterwards: 15000; // 1min, 250hz
    public static int channelCount;

    public static stream_info info;
    public static stream_inlet inlet;

    public static ASR_Calibration state;

    @Override
        public void onCreate() {
            super.onCreate();
        }

        @Override
        public int onStartCommand(Intent intent, int flags, int startId) {
            if (intent.getAction().equals("android.intent.action.START_CAL")) {
                Toast.makeText(this,"Start Calibration",Toast.LENGTH_SHORT).show();
                Log.i(LOG_TAG, "Received Start Calibration Foreground Intent");

                Bitmap icon = BitmapFactory.decodeResource(getResources(), R.mipmap.ic_launcher);

                RemoteViews notificationView = new RemoteViews(this.getPackageName(),R.layout.notification);
                Notification notification = new NotificationCompat.Builder(this)
                        .setContentTitle("ASR")
                        .setTicker("ASR")
                        .setContentText("ASR")
                        .setSmallIcon(R.mipmap.ic_launcher)
                        .setLargeIcon(
                                Bitmap.createScaledBitmap(icon, 128, 128, false))
                        .setContent(notificationView)
                        .setOngoing(true).build();


                Runnable runnable = new Runnable() {
                    @Override
                    public void run() {
                        Log.i(LOG_TAG, "We are in the run method.");
                        // look for EEG streams
                        vectorinfo eegStream;
                        do {
                            eegStream = lslAndroid.resolve_stream("type", "EEG");
                        } while (eegStream.isEmpty());
                        Log.i(LOG_TAG, "We have found a stream!");

                        info = eegStream.get(0);
                        inlet = new stream_inlet(info);

                        channelCount = inlet.info().channel_count();
                        samplingRate = (int)inlet.info().nominal_srate();
                        buffer = new SampleBuffer(capacity, channelCount);
                        double[][] calib_data = new double[channelCount][capacity];

                        // pull samples
                        vectord samples = new vectord(channelCount);
                        while(!buffer.isAtFullCapacity()) {
                            double t = inlet.pull_sample(samples, 1.0);
                            buffer.insertSample(My_LSL_Utils.vectord_to_array(samples));
                        }
                        Log.i(LOG_TAG, "We have a filled up buffer with samples.");
                        /*
                        put data in double array for further processing
                         */
                        for (int c = 0; c < channelCount; c++) {
                            calib_data[c] = buffer.getValuesFromOneChannel(c);
                        }

                        // begin to calibrate as soon as we have the values
                        state = new ASR_Calibration(calib_data);
                    }
                };
                //TODO does this have to be in a thread?
                new Thread(runnable).start();

                // startForeground tells Android to keep this service alive no matter what, it does
                // not really start anything
                startForeground(Constants.NOTIFICATION_ID.FOREGROUND_SERVICE, notification);
                Log.i(LOG_TAG, "started the service in the foreground");

            } else if (intent.getAction().equals("android.intent.action.INTERRUPT_CAL")){
                Toast.makeText(this,"Interrupted Calibration",Toast.LENGTH_SHORT).show();
                Log.i(LOG_TAG, "Received Stop Foreground Intent");
                stopForeground(true);
                stopSelf();
            }
            return START_STICKY;
        }


        @Override
        public void onDestroy() {
            super.onDestroy();
            Log.i(LOG_TAG, "In onDestroy");
            // do clean up stuff
        }

        @Override
        public IBinder onBind(Intent intent) {
            // Used only in case of bound services.
            return null;
        }




}
