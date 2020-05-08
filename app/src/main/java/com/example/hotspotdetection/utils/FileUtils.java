package com.example.hotspotdetection.utils;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class FileUtils {
    private static String TAG = "File INFO";


    public static String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();

        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch ( IOException ex) {
            Log.d(TAG, "Failed to upload a file");
        }
        return "";
    }

    public static void loadFile(Context context,String fileName ,double[] array) throws IOException {
        String line = null;
        InputStream stream = context.getAssets().open(fileName);
        InputStreamReader isr = new InputStreamReader(stream);
        BufferedReader bf = new BufferedReader(isr);
        String[] items = null;

        while ((line= bf.readLine()) !=null)
        {
            items = line.split(",");
            for (int i =0; i < items.length; i++)
                array[i] = Double.parseDouble(items[i]);

        }
        bf.close();


    }



}
