package com.example.hotspotdetection.classifier;

import android.app.Activity;
import android.util.Log;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

public class Unet extends SegmentationModel
{
    /**
     * Initializes a {@code Classifier}.
     *
     * @param activity
     * @param device
     * @param numThreads
     */
    private static  float[] IMAGE_MEAN = new float[]{0.0f, 0.0f, 0.0f};
    private static float[]  IMAGE_STD = new float[] {255.0f, 255.0f, 255.0f};

    public Unet(Activity activity, Models model, Device device, int numThreads)
            throws IOException
    {

        super(activity, model, device, numThreads);
    }


    @Override
    protected String getModelPath(Models model) {
        if (model == Models.INDOOR_MODEL){
            return "model_rgb_lighter.tflite";

        }
        return "model_rgb.tflite";




    }

    @Override
    protected String getLabelPath() {
        return null;
    }

    @Override
    protected TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);

    }

    @Override
    protected TensorOperator getPostprocessNormalizeOp() {
        return null;
    }
}
