package com.example.hotspotdetection;

import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class ContourDetection {
    private Mat mHierarchy = new Mat();

    public boolean  isContour(Mat img){


        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(img, contours, mHierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        boolean result = (contours.size() > 0) ? true:false;
        return result;

    }

}
