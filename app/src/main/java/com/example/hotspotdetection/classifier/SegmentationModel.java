package com.example.hotspotdetection.classifier;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import com.example.hotspotdetection.ContourDetection;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC4;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2RGBA;


public abstract class SegmentationModel {


    public enum Models{
        INDOOR_MODEL,
        OUTDOOR_MODEL

    }
    /** The runtime device type used for executing classification. */
    public enum Device {
        CPU,
        GPU
    }



    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** Image size along the x axis. */
    private final int imageSizeX;

    /** Image size along the y axis. */
    private final int imageSizeY;

    /** Optional GPU delegate for accleration. */
    private GpuDelegate gpuDelegate = null;



    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    protected Interpreter tflite;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** Input image TensorBuffer. */
    private TensorImage inputImageBuffer;

    /** Output probability TensorBuffer. */
    private final TensorBuffer outputProbabilityBuffer;

    private float [] data;
    String label;


    /** Processer to apply post processing of the output probability. */
    //private final TensorProcessor probabilityProcessor;


    /**
     * Creates a classifier with the provided configuration.
     *
     * @param activity The current Activity
     * @param device The device to use for classification.
     * @param numThreads The number of threads to use for classification.
     * @return A classifier with the desired configuration.
     * //
     */
    public static SegmentationModel create(Activity activity, Models model, Device device, int numThreads) throws IOException {
        return new Unet(activity, model, device, numThreads);

    }

    /** Initializes a {@code Classifier}. */
    protected SegmentationModel(Activity activity, Models model, Device device, int numThreads) throws IOException {


        tfliteModel = FileUtil.loadMappedFile(activity, getModelPath(model));
        switch (device) {
            case GPU:
                gpuDelegate = new GpuDelegate();
                tfliteOptions.addDelegate(gpuDelegate);
                break;
            case CPU:
                break;
        }
        tfliteOptions.setNumThreads(numThreads);
        tflite = new Interpreter(tfliteModel, tfliteOptions);




        // Reads type and shape of input and output tensors, respectively.
        //model is Float32  no need for quantiztion.
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Creates the post processor for the output probability.
        //probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

        Log.d( "CLASSIFIER", "Created a Tensorflow Lite Image Classifier.");
    }

    /** Runs inference and returns the classification results. */
    public void recognizeImage(final Bitmap bitmap) {
        // Logs this method so that it can be analyzed with systrace.

        long startTimeForLoadImage = SystemClock.uptimeMillis();
        inputImageBuffer = loadImage(bitmap);
        long endTimeForLoadImage = SystemClock.uptimeMillis();

        // Runs the inference call.

        long startTimeForReference = SystemClock.uptimeMillis();
        tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

        data = outputProbabilityBuffer.getFloatArray();

        long endTimeForReference = SystemClock.uptimeMillis();

        //LOGGER.v("Timecost to run model inference: " + (endTimeForReference - startTimeForReference));
        Log.d("CLASSIFIER", "timecost for inference: "+ (endTimeForReference - startTimeForReference));



    }

    public Bitmap smoothBlend(Bitmap fg_bmp, ContourDetection detector)
    {
        Mat fg = new Mat(fg_bmp.getHeight(),fg_bmp.getWidth(),CV_8UC4);
        Mat mskmat = new Mat(imageSizeY, imageSizeX, CV_32F); //holds data array for result which 128 * 128

        Utils.bitmapToMat(fg_bmp, fg);


        Bitmap bm = null;
        if (data != null){

            //Binarize the mask
            mskmat.put(0,0,data);
            Imgproc.threshold(mskmat,mskmat,0.02,1.0,Imgproc.THRESH_BINARY);
            mskmat.convertTo(mskmat,CV_8U,255);

            //get rid of noise in the image .../
            mskmat = remove_noise(mskmat);

            //get label for prediction here.
            label = (detector.isContour(mskmat))? "fire detected": "no fire";


            Imgproc.cvtColor(mskmat, mskmat,COLOR_GRAY2RGBA);


            // spilt the channels in Mat object ..
            List<Mat> bgr = new ArrayList<>();
            Core.split(mskmat,bgr);

            // set green and blue channels to zero..
            bgr.get(1).setTo(Scalar.all(0));
            bgr.get(2).setTo(Scalar.all(0));

            Core.merge(bgr, mskmat);


            Imgproc.resize(mskmat,mskmat, new Size(fg.cols(),fg.rows()));

            Mat dst  = new Mat();

            Core.addWeighted(fg, 0.7, mskmat, 0.3, 0.0, dst );


            // convert to bitmap:
            bm = Bitmap.createBitmap(fg_bmp.getWidth(), fg_bmp.getHeight(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(dst, bm);

            //bm = Bitmap.createBitmap(mskmat.cols(), mskmat.rows(),Bitmap.Config.ARGB_8888);
            //Utils.matToBitmap(mskmat, bm);



        }
        return bm;

    }

    public Mat remove_noise(Mat img){
        Mat kernel =Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,new Size(3,3));
        Imgproc.erode(img, img, kernel);
        Imgproc.dilate(img, img, kernel);
        return img;
    }

    public String getLabel(){

        return label;
    }






    /** Closes the interpreter and model to release resources. */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }

        tfliteModel = null;
    }

    /** Get the image size along the x axis. */
    public int getImageSizeX() {
        return imageSizeX;
    }
    /** Get the image size along the y axis. */

    public int getImageSizeY() {
        return imageSizeY;
    }

    /** Loads input image, and applies preprocessing. */
    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        //.add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }



    /** Gets the name of the model file stored in Assets. */
    protected abstract String getModelPath(Models model);

    /** Gets the name of the label file stored in Assets. */
    protected abstract String getLabelPath();

    /** Gets the TensorOperator to nomalize the input image in preprocessing. */
    protected abstract TensorOperator getPreprocessNormalizeOp();

    /**
     * Gets the TensorOperator to dequantize the output probability in post processing.
     *
     * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
     * essentially linear transformation). For float model, de-quantize is not required. But to
     * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
     * 1.0f, respectively.
     */
    protected abstract TensorOperator getPostprocessNormalizeOp();
}
