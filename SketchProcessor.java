package com.example.ribhance;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class SketchProcessor {

    private static final String TAG = "SketchProcessor";

    private static OrtEnvironment env;
    private static OrtSession onnxSession;

    public static void initOnnx(Context context) {
        try {
            if (env == null) env = OrtEnvironment.getEnvironment();
            if (onnxSession == null) {
                String modelPath = copyAssetToFile(context, "sketch.onnx");
                if (modelPath != null) {
                    onnxSession = env.createSession(modelPath, new OrtSession.SessionOptions());
                    Log.d(TAG, "ONNX Model loaded successfully!");
                } else {
                    Log.w(TAG, "sketch.onnx not found in assets");
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to load ONNX model: " + e.getMessage());
            onnxSession = null;
        }
    }

    public static boolean isOnnxLoaded() {
        return onnxSession != null;
    }

    public static Bitmap applyEffect(Bitmap input, String mode) {
        if (input == null) return null;
        try {
            switch (mode) {
                case "OpenCV":
                    return applyOpenCVSketch(input);
                case "ONNX":
                    return applyOnnxSketch(input);
                case "Stylize":
                    return applyStylizeEffect(input);
                default:
                    return input;
            }
        } catch (Exception e) {
            Log.e(TAG, "applyEffect error: " + e.getMessage());
            return input;
        }
    }

    // OpenCV pencil sketch
    private static Bitmap applyOpenCVSketch(Bitmap input) {
        Mat src = new Mat();
        try {
            Utils.bitmapToMat(input, src);
            if (src.channels() == 4) Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2BGR);

            Mat gray = new Mat();
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

            Mat inverted = new Mat();
            Core.bitwise_not(gray, inverted);

            Mat blur = new Mat();
            Imgproc.GaussianBlur(inverted, blur, new Size(21, 21), 0);

            Mat invertedBlur = new Mat();
            Core.bitwise_not(blur, invertedBlur);

            Mat sketch = new Mat();
            Core.divide(gray, invertedBlur, sketch, 256.0);

            Mat sketchRGBA = new Mat();
            Imgproc.cvtColor(sketch, sketchRGBA, Imgproc.COLOR_GRAY2RGBA);

            Bitmap output = Bitmap.createBitmap(sketchRGBA.cols(), sketchRGBA.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(sketchRGBA, output);

            // release
            src.release(); gray.release(); inverted.release(); blur.release(); invertedBlur.release(); sketch.release(); sketchRGBA.release();

            return output;
        } catch (Exception e) {
            Log.e(TAG, "applyOpenCVSketch error: " + e.getMessage());
            if (src != null) src.release();
            return input;
        }
    }

    // ONNX model inference (safe handling)
    private static Bitmap applyOnnxSketch(Bitmap input) {
        if (onnxSession == null) {
            Log.w(TAG, "ONNX session null, falling back to OpenCV");
            return applyOpenCVSketch(input);
        }

        try {
            final int modelSize = 256;
            Bitmap resized = Bitmap.createScaledBitmap(input, modelSize, modelSize, true);

            float[][][][] inputData = new float[1][3][modelSize][modelSize];
            for (int y = 0; y < modelSize; y++) {
                for (int x = 0; x < modelSize; x++) {
                    int px = resized.getPixel(x, y);
                    inputData[0][0][y][x] = ((px >> 16) & 0xFF) / 255.0f;
                    inputData[0][1][y][x] = ((px >> 8) & 0xFF) / 255.0f;
                    inputData[0][2][y][x] = (px & 0xFF) / 255.0f;
                }
            }

            OnnxTensor tensor = OnnxTensor.createTensor(env, inputData);
            OrtSession.Result result = onnxSession.run(
                    Collections.singletonMap(onnxSession.getInputNames().iterator().next(), tensor)
            );

            Object rawOut = result.get(0).getValue();
            Bitmap outBmp = null;
            if (rawOut instanceof float[][][][]) {
                float[][][][] out = (float[][][][]) rawOut;
                int h = out[0][0].length;
                int w = out[0][0][0].length;
                outBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int r = Math.min(255, Math.max(0, (int) (out[0][0][y][x] * 255)));
                        int g = Math.min(255, Math.max(0, (int) (out[0][1][y][x] * 255)));
                        int b = Math.min(255, Math.max(0, (int) (out[0][2][y][x] * 255)));
                        outBmp.setPixel(x, y, Color.rgb(r, g, b));
                    }
                }
                Bitmap scaled = Bitmap.createScaledBitmap(outBmp, input.getWidth(), input.getHeight(), true);
                outBmp.recycle();
                tensor.close();
                result.close();
                return scaled;
            } else {
                Log.e(TAG, "Unexpected ONNX output type: " + rawOut.getClass());
                tensor.close();
                result.close();
                return applyOpenCVSketch(input);
            }
        } catch (Exception e) {
            Log.e(TAG, "applyOnnxSketch error: " + e.getMessage());
            return applyOpenCVSketch(input);
        }
    }

    // Stylize (cartoon)
    private static Bitmap applyStylizeEffect(Bitmap input) {
        Mat src = new Mat();
        try {
            Utils.bitmapToMat(input, src);
            if (src.channels() == 4) Imgproc.cvtColor(src, src, Imgproc.COLOR_RGBA2BGR);
            Mat dst = new Mat();
            Photo.stylization(src, dst, 60f, 0.45f);
            Mat dstRGBA = new Mat();
            Imgproc.cvtColor(dst, dstRGBA, Imgproc.COLOR_BGR2RGBA);
            Bitmap output = Bitmap.createBitmap(dstRGBA.cols(), dstRGBA.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(dstRGBA, output);
            src.release(); dst.release(); dstRGBA.release();
            return output;
        } catch (Exception e) {
            Log.e(TAG, "applyStylizeEffect error: " + e.getMessage());
            if (src != null) src.release();
            return input;
        }
    }

    // copy asset helper
    private static String copyAssetToFile(Context context, String assetName) {
        try {
            File outFile = new File(context.getFilesDir(), assetName);
            if (!outFile.exists()) {
                InputStream in = context.getAssets().open(assetName);
                FileOutputStream out = new FileOutputStream(outFile);
                byte[] buffer = new byte[1024];
                int read;
                while ((read = in.read(buffer)) != -1) out.write(buffer, 0, read);
                in.close(); out.close();
            }
            return outFile.getAbsolutePath();
        } catch (Exception e) {
            Log.e(TAG, "copyAssetToFile error: " + e.getMessage());
            return null;
        }
    }
}
