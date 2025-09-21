package com.example.ribhance;

import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;

public class ImageUtils {
    private static final String TAG = "ImageUtils";

    // Convert ImageProxy (YUV_420_888) -> Bitmap (correct rotation)
    public static Bitmap imageProxyToBitmap(ImageProxy image) {
        try {
            if (image == null) return null;

            if (image.getFormat() == ImageFormat.YUV_420_888) {
                ImageProxy.PlaneProxy[] planes = image.getPlanes();
                ByteBuffer yBuffer = planes[0].getBuffer();
                ByteBuffer uBuffer = planes[1].getBuffer();
                ByteBuffer vBuffer = planes[2].getBuffer();

                int ySize = yBuffer.remaining();
                int uSize = uBuffer.remaining();
                int vSize = vBuffer.remaining();

                byte[] nv21 = new byte[ySize + uSize + vSize];
                // U and V are swapped for NV21
                yBuffer.get(nv21, 0, ySize);
                vBuffer.get(nv21, ySize, vSize);
                uBuffer.get(nv21, ySize + vSize, uSize);

                YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
                ByteArrayOutputStream out = new ByteArrayOutputStream();
                yuvImage.compressToJpeg(new Rect(0, 0, image.getWidth(), image.getHeight()), 90, out);
                byte[] jpegBytes = out.toByteArray();
                Bitmap rawBitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);

                // Rotate if needed
                Matrix matrix = new Matrix();
                int rotation = image.getImageInfo().getRotationDegrees();
                if (rotation != 0) matrix.postRotate(rotation);

                Bitmap rotated = Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.getWidth(), rawBitmap.getHeight(), matrix, true);
                rawBitmap.recycle();
                return rotated;
            } else {
                // fallback: decode first plane
                ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                byte[] bytes = new byte[buffer.remaining()];
                buffer.get(bytes);
                return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
            }
        } catch (Exception e) {
            Log.e(TAG, "imageProxyToBitmap failed: " + e.getMessage());
            return null;
        }
    }

    // Save bitmap to gallery (Android Q+ and older fallback)
    public static Uri saveBitmapToGallery(Context context, Bitmap bitmap) {
        if (bitmap == null) return null;
        Uri imageUri = null;
        OutputStream out = null;
        try {
            ContentValues values = new ContentValues();
            String displayName = "sketch_" + System.currentTimeMillis() + ".png";
            values.put(MediaStore.Images.Media.DISPLAY_NAME, displayName);
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/SketchApp");
                imageUri = context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                if (imageUri != null) {
                    out = context.getContentResolver().openOutputStream(imageUri);
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                }
            } else {
                // pre-Android Q fallback
                String imagesDir = MediaStore.Images.Media.insertImage(context.getContentResolver(), bitmap, displayName, "Saved from Ribhance");
                if (imagesDir != null) {
                    imageUri = Uri.parse(imagesDir);
                }
            }
            if (out != null) out.close();
            Log.d(TAG, "Saved image: " + imageUri);
            return imageUri;
        } catch (Exception e) {
            Log.e(TAG, "saveBitmapToGallery failed: " + e.getMessage());
            try { if (out != null) out.close(); } catch (Exception ignored) {}
            return null;
        }
    }
}
