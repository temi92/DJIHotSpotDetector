package com.example.hotspotdetection.classifier;

import android.app.Activity;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

/** This TensorFlowLite classifier works with the float MobileNet model. */
public class HotspotClassifier extends Classifier {

    /** Float MobileNet requires additional normalization of the used input. */
    private static final float IMAGE_MEAN = 127.5f;

    private static final float IMAGE_STD = 127.5f;

    //private static  float[] IMAGE_MEAN = new float[]{103.939f, 116.779f, 123.68f};
    //private static float[]  IMAGE_STD = new float[] {1.0f, 1.0f, 1.0f};

    /**
     * Float model does not need dequantization in the post-processing. Setting mean and std as 0.0f
     * and 1.0f, repectively, to bypass the normalization.
     */
    private static final float PROBABILITY_MEAN = 0.0f;

    private static final float PROBABILITY_STD = 1.0f;

    /**
     * Initializes a {@code ClassifierFloatMobileNet}.
     *
     * @param activity
     */
    public HotspotClassifier(Activity activity, Device device, int numThreads)
            throws IOException {
        super(activity, device, numThreads);
    }

    @Override
    protected String getModelPath() {
        // you can download this file from
        // see build.gradle for where to obtain this file. It should be auto
        // downloaded into assets.
        //return "mobilenet_v1_1.0_224.tflite";
        return "hotspot.tflite";
    }

    @Override
    protected String getLabelPath() {
        //return "labels.txt";
        return "hotspotlabels.txt";
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp()
    {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }
}

