package com.example.hotspotdetection;

import android.app.Activity;

import android.graphics.Bitmap;

import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import dji.common.camera.SettingsDefinitions;
import dji.common.camera.SystemState;
import dji.common.error.DJIError;
import dji.common.mission.waypoint.WaypointMission;
import dji.common.product.Model;
import dji.common.useraccount.UserAccountState;
import dji.common.util.CommonCallbacks;
import dji.sdk.base.BaseProduct;
import dji.sdk.camera.Camera;
import dji.sdk.camera.VideoFeeder;
import dji.sdk.codec.DJICodecManager;
import dji.sdk.flightcontroller.FlightController;
import dji.sdk.useraccount.UserAccountManager;

import com.example.hotspotdetection.classifier.Classifier.Device;
import com.example.hotspotdetection.classifier.Classifier.ModelType;
import com.example.hotspotdetection.classifier.Classifier;
import com.example.hotspotdetection.segmentation.Detector;

import java.io.IOException;
import java.util.List;


public class MainActivity extends Activity implements TextureView.SurfaceTextureListener, View.OnClickListener {

    private static final String TAG = MainActivity.class.getName();
    protected VideoFeeder.VideoDataListener mReceivedVideoDataListener = null;
    private ImageView mImageSurface;

    // Codec for video live view
    protected DJICodecManager mCodecManager = null;

    protected TextureView mVideoSurface = null;
    private Button mCaptureBtn, rtlBtn;

    private Handler handler, handler_proccess;
    private HandlerThread handlerThread;

    private TextView altDisplay, labelDisplay;
    private Runnable postInferenceCallback;



    private Bitmap imageBitmap;
    private boolean loadConfig = false;

    private boolean isProcessingFrame = false;

    //initialise classifer configurations..
    private  Classifier classifier;
    private ModelType model = ModelType.HOTSPOT_MODEL;
    private Device device = Device.CPU;
    private int numThreads = -1;
    Detector detector;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.d("OPENCV_STATUS", "OpenCV loaded successfully");
                    detector = new Detector();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };





    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        handler = new Handler();

        initUI();

        recreateClassifier();

        // The callback for receiving the raw H264 video data for camera live view
        mReceivedVideoDataListener = new VideoFeeder.VideoDataListener() {

            @Override
            public void onReceive(byte[] videoBuffer, int size) {
                if (mCodecManager != null) {
                    mCodecManager.sendDataToDecoder(videoBuffer, size);
                }
            }
        };




    }

    protected void onProductChange() {
        initPreviewer();
        loginAccount();
    }

    private void loginAccount(){

        UserAccountManager.getInstance().logIntoDJIUserAccount(this,
                new CommonCallbacks.CompletionCallbackWith<UserAccountState>() {
                    @Override
                    public void onSuccess(final UserAccountState userAccountState) {
                        Log.e(TAG, "Login Success");
                    }
                    @Override
                    public void onFailure(DJIError error) {
                        showToast("Login Error:"
                                + error.getDescription());
                    }
                });
    }

    @Override
    public void onResume() {
        Log.e(TAG, "onResume");

        //OPENCV LIBRARY..

        if(!OpenCVLoader.initDebug()){

            Log.d("OPENCV_STATUS", "OPENCV NOT LOADED");
        }
        else{

            Log.d("OPENCV STATUS", "OPENCV LOADED!!!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        super.onResume();
        initPreviewer();
        onProductChange();

        if(mVideoSurface == null) {
            Log.e(TAG, "mVideoSurface is null");
        }

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler_proccess = new Handler(handlerThread.getLooper());
        //recreateClassifier();
        isProcessingFrame = false;
    }

    @Override
    public void onPause() {
        Log.e(TAG, "onPause");


        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler_proccess = null;
        } catch (final InterruptedException e) {
            Log.d(String.valueOf(e), "Exception");
        }
        uninitPreviewer();
        super.onPause();
    }

    @Override
    public void onStop() {
        Log.e(TAG, "onStop");
        super.onStop();
    }

    public void onReturn(View view){
        Log.e(TAG, "onReturn");
        this.finish();
    }

    @Override
    protected void onDestroy() {
        Log.e(TAG, "onDestroy");
        uninitPreviewer();
        super.onDestroy();
    }

    private void initUI() {
        // init mVideoSurface
        mVideoSurface = (TextureView)findViewById(R.id.video_previewer_surface);

        mCaptureBtn = (Button) findViewById(R.id.btn_capture);
        mImageSurface = (ImageView)findViewById(R.id.image_cv);
        altDisplay =  (TextView)findViewById(R.id.alt_display);
        labelDisplay =  (TextView)findViewById(R.id.label);
        rtlBtn = (Button) findViewById(R.id.btn_RTL);

        if (null != mVideoSurface) {
            mVideoSurface.setSurfaceTextureListener(this);
        }

        mCaptureBtn.setOnClickListener(this);
        rtlBtn.setOnClickListener(this);

    }

    private void initPreviewer() {

        BaseProduct product = HotSpotApplication.getProductInstance();

        if (product == null || !product.isConnected()) {
            showToast(getString(R.string.disconnected));
        } else {
            if (null != mVideoSurface) {
                mVideoSurface.setSurfaceTextureListener(this);
            }
            if (!product.getModel().equals(Model.UNKNOWN_AIRCRAFT)) {
                VideoFeeder.getInstance().getPrimaryVideoFeed().addVideoDataListener(mReceivedVideoDataListener);
            }
        }
    }

    private void uninitPreviewer() {
        Camera camera = HotSpotApplication.getCameraInstance();
        if (camera != null){
            // Reset the callback
            VideoFeeder.getInstance().getPrimaryVideoFeed().addVideoDataListener(null);
        }
    }

    @Override
    public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
        Log.e(TAG, "onSurfaceTextureAvailable");
        if (mCodecManager == null) {
            mCodecManager = new DJICodecManager(this, surface, width, height);
        }
    }

    @Override
    public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
        Log.e(TAG, "onSurfaceTextureSizeChanged");
    }

    @Override
    public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
        Log.e(TAG,"onSurfaceTextureDestroyed");
        if (mCodecManager != null) {
            mCodecManager.cleanSurface();
            mCodecManager = null;
        }

        return false;
    }

    @Override
    public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        //imageBitmap = mVideoSurface.getBitmap();
        imageBitmap = Bitmap.createScaledBitmap(mVideoSurface.getBitmap(), 300, 300, false);

        updateUiThread();

        if (isProcessingFrame){
            Log.d("PROCESS", "still processing sorry");
            return;
        }
        isProcessingFrame = true;
        postInferenceCallback =
                new Runnable() {
                    @Override
                    public void run() {
                        isProcessingFrame = false;
                    }
                };
        //process imageBitmap for classification
        processImage();




    }

    public void showToast(final String msg) {
        runOnUiThread(new Runnable() {
            public void run() {
                Toast.makeText(MainActivity.this, msg, Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    public void onClick(View v) {

        switch (v.getId()) {
            case R.id.btn_capture:{
                captureAction();
                break;
            }
            case R.id.btn_RTL:{
                goHome();
                break;
            }

            default:
                break;
        }
    }


    // Method for taking photo
    private void captureAction()
    {
        final Camera camera = HotSpotApplication.getCameraInstance();
        if (camera != null) {

            SettingsDefinitions.ShootPhotoMode photoMode = SettingsDefinitions.ShootPhotoMode.SINGLE; // Set the camera capture mode as Single mode
            camera.setShootPhotoMode(photoMode, new CommonCallbacks.CompletionCallback(){
                @Override
                public void onResult(DJIError djiError) {
                    if (null == djiError) {
                        handler.postDelayed(new Runnable() {
                            @Override
                            public void run() {
                                camera.startShootPhoto(new CommonCallbacks.CompletionCallback() {
                                    @Override
                                    public void onResult(DJIError djiError) {
                                        if (djiError == null) {
                                            showToast("take photo: success");
                                        } else {
                                            showToast(djiError.getDescription());
                                        }
                                    }
                                });
                            }
                        }, 2000);
                    }
                }
            });
        }
    }



    public void goHome() {
        HotSpotApplication.getAircraftInstance().getFlightController().startGoHome(new CommonCallbacks.CompletionCallback() {
            @Override
            public void onResult(DJIError djiError) {
                if (djiError == null) {
                    showToast("succeeded in going home");

                } else {
                    showToast("not succedded in going home");

                }
            }

        });
    }

    private synchronized void runInBackground(final Runnable r) {
        if (handler_proccess != null) {
            handler_proccess.post(r);
        }
    }

    private void readyForNextImage(){
            if (postInferenceCallback != null) {
                postInferenceCallback.run();
            }
        }

    private void processImage(){
        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        if(classifier!=null){

                            final long startTime = SystemClock.uptimeMillis();
                            final List<Classifier.Recognition> results =
                                    classifier.recognizeImage(imageBitmap);
                           final long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                           String label = results.get(0).getId() + ": "+ String.format("%.1f", results.get(0).getConfidence()*100.0f ) + "%";
                           Log.d("Detect", String.valueOf(results));
                           //grab copy of bitmap..
                           Bitmap bm = imageBitmap.copy(imageBitmap.getConfig(), true);

                            if (results.get(0).getId().equals("fire")){
                            Mat image = new Mat();
                            Utils.bitmapToMat(bm, image);

                            Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);
                            detector.process(image);
                            Imgproc.drawContours(image, detector.getContours(), -1, new Scalar(200, 200, 0, 255), 5);

                                /*
                                // draw centroids on contours...
                                List<int[]> centroids = detector.getCentroids();
                                Iterator<int[]> each = centroids.iterator();
                                while(each.hasNext()){
                                    int [] centroid = each.next();
                                    Imgproc.circle(image, new Point(centroid[0],centroid[1]), 3, new Scalar(200, 200, 0, 255), 4);

                                }
                                */

                                //convert mat to bitmap
                                Utils.matToBitmap(image, bm);

                            }

                            Bitmap finalBm = bm;
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                        labelDisplay.setText(label);
                                        mImageSurface.setImageBitmap(null);
                                        mImageSurface.setImageBitmap(finalBm);
                                }
                            });

                        }
                        readyForNextImage();
                    }
                }

        );
    }

    private void recreateClassifier(){
        if (classifier != null) {
            Log.d("CLASSIFIER", "Closing classifier.");
            classifier.close();
            classifier = null;
        }

        try{
            classifier = Classifier.create(this, device, numThreads);

        } catch(IOException e){
            Log.d("CLASSIFIER", "cannot load classifier");

        }
    }





    public void updateUiThread(){

        MainActivity.this.runOnUiThread(new Runnable() {

            @Override
            public void run() {
                double alt = HotSpotApplication.getAircraftInstance().getFlightController().getState().getAircraftLocation().getAltitude();
                String altString = String.format("%.2f", alt);
                altDisplay.setText("ALT: " +  String.valueOf(altString)+ "m");

            }
        });

    }







}