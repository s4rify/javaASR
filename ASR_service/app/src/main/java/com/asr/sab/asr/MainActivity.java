package com.asr.sab.asr;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;


public class MainActivity extends AppCompatActivity {


    // note to self: .so files are not supported by gradle, but them into .jar file instead
    static {
        System.loadLibrary("lslAndroid");
    }

    private Button startCalibButton, startProcessButton, interruptCalibButton, stopProcessButton;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        startCalibButton = (Button)findViewById(R.id.button1);
        interruptCalibButton = (Button)findViewById(R.id.button2);
        startProcessButton = (Button)findViewById(R.id.button3);
        stopProcessButton = (Button)findViewById(R.id.button4);
        //TODO this button needs logic
        //killServiceButton = (Button)findViewById(R.id.notification_button_close);


        // when this button is pressed, a new thread is started in which we first open an inlet and
        // then pull samples from it.
        startCalibButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // new Intent(Context, Class)
                Intent startIntent = new Intent(MainActivity.this, ForegroundCalibService.class);
                startIntent.setAction("android.intent.action.START_CAL");
                startService(startIntent);
            }
        });


        // when this button is pressed, a new thread is started in which we first open an inlet and
        // then pull samples from it.
        startProcessButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent startIntent = new Intent(MainActivity.this, ForegroundProcService.class);
                startIntent.setAction("android.intent.action.START_PROC");
                startService(startIntent);
            }
        });

/*        *//** This button calls the same OnStartCommand() method as the startCalib button. In this
            method, the action which is bound to the intent is checked and the service is stopped
            in the case of the interrupt button.
        *//*
        interruptCalibButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent startIntent = new Intent(MainActivity.this, ForegroundCalibService.class);
                startIntent.setAction("android.intent.action.INTERRUPT_CAL");
                startService(startIntent);
            }
        });

        *//** This button calls the same OnStartCommand() method as the startProc button. In this
         method, the action which is bound to the intent is checked and the service is stopped
         in the case of the stop button.
         *//*
        stopProcessButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent startIntent = new Intent(MainActivity.this, ForegroundProcService.class);
                startIntent.setAction("android.intent.action.STOP_PROC");
                startService(startIntent);
            }
        });*/



    }

}