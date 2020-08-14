package com.example.hotspotdetection.segmentation;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Detector {

    private Mat mHierarchy = new Mat();

    private List<int[]> centroids = new ArrayList<int[]>();
    private List<MatOfPoint> mCountours = new ArrayList<MatOfPoint>();


    public void process(Mat img)
    {
        Mat imB = new Mat(img.cols(), img.rows(), CvType.CV_8UC4);

        Imgproc.threshold(img,imB,0,255, Imgproc.THRESH_BINARY+ Imgproc.THRESH_OTSU);


        //erosion operations..
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5,5));
        Imgproc.dilate(imB, imB, element);


        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();


        Imgproc.findContours(imB, contours, mHierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);





        Iterator<MatOfPoint> each = contours.iterator();



        while(each.hasNext()){
            MatOfPoint contour = each.next();
            if (Imgproc.contourArea(contour) > 400){
                int[] centroid = new int[2];


                mCountours.add(contour);
                Moments p = Imgproc.moments(contour);
	            centroid[0] = (int) (p.get_m10() / p.get_m00()); //get X centroid
	            centroid[1]  = (int) (p.get_m01() / p.get_m00()); //get Y centroid
                centroids.add(centroid);
                //Imgproc.circle(imB, new Point(centroid[0],centroid[1]), 10, new Scalar(200, 200, 0, 255), 4);
            }

        }


    }




    public List<int[]> getCentroids(){
        return centroids;

    }

    public List<MatOfPoint> getContours(){
        return mCountours;

    }
}
