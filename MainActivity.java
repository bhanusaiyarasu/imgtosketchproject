package com.example.ribhance;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Toast;
import android.widget.ImageView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.viewpager2.widget.ViewPager2;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";

    private RadioGroup modeGroup, inputSourceGroup;
    private Button btnChoose, btnSave, btnStop, btnCapture;
    private LinearLayout buttonPanel;
    private ViewPager2 viewPager;
    private ImageView liveView;
    private PreviewView cameraPreview;

    private String currentMode = "OpenCV";
    private Bitmap processedBitmap, originalBitmap;

    private ImageAdapter adapter;
    private ExecutorService cameraExecutor;
    private ImageCapture imageCapture;
    private ProcessCameraProvider cameraProvider;
    private boolean liveActive = false;

    private final ActivityResultLauncher<Intent> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    try {
                        originalBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                        processAndDisplay();
                    } catch (IOException e) {
                        Log.e(TAG, "Gallery load failed: " + e.getMessage());
                        Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
                    }
                }
            });

    // Load OpenCV
    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV not loaded!");
        } else {
            Log.d("OpenCV", "OpenCV loaded successfully!");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // test OpenCV quickly (optional)
        try {
            Mat testMat = new Mat(1, 1, CvType.CV_8UC1);
            Log.d(TAG, "OpenCV Mat ok: " + testMat);
        } catch (Exception e) {
            Log.e(TAG, "OpenCV test failed: " + e.getMessage());
        }

        SketchProcessor.initOnnx(this); // attempt to load ONNX model

        // find views
        modeGroup = findViewById(R.id.modeGroup);
        inputSourceGroup = findViewById(R.id.inputSourceGroup);
        btnChoose = findViewById(R.id.btnChoose);
        btnSave = findViewById(R.id.btnSave);
        btnStop = findViewById(R.id.btnStop);
        btnCapture = findViewById(R.id.btnCapture);
        buttonPanel = findViewById(R.id.buttonPanel);
        viewPager = findViewById(R.id.viewPager);
        liveView = findViewById(R.id.liveView);
        cameraPreview = findViewById(R.id.cameraPreview);

        adapter = new ImageAdapter();
        viewPager.setAdapter(adapter);

        cameraExecutor = Executors.newSingleThreadExecutor();

        // Mode selection listener
        modeGroup.setOnCheckedChangeListener((group, checkedId) -> {
            RadioButton rb = findViewById(checkedId);
            if (rb != null) {
                currentMode = rb.getText().toString();
                Log.d(TAG, "Mode selected: " + currentMode);
            }
        });

        // Choose button (handles upload / capture binding / live start)
        btnChoose.setOnClickListener(v -> {
            int checked = inputSourceGroup.getCheckedRadioButtonId();
            if (checked == R.id.radioUpload) {
                // launch gallery
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                galleryLauncher.launch(intent);
            } else if (checked == R.id.radioCapture) {
                startCameraForCapture();
            } else if (checked == R.id.radioLive) {
                startLiveCamera();
            } else {
                Toast.makeText(this, "Select a source first", Toast.LENGTH_SHORT).show();
            }
        });

        // Save processed image
        btnSave.setOnClickListener(v -> {
            if (processedBitmap == null) {
                Toast.makeText(this, "No processed image to save", Toast.LENGTH_SHORT).show();
                return;
            }
            Uri uri = ImageUtils.saveBitmapToGallery(this, processedBitmap);
            if (uri != null) {
                Toast.makeText(this, "Saved to Gallery", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Save failed", Toast.LENGTH_SHORT).show();
            }
        });

        // Stop live
        btnStop.setOnClickListener(v -> stopLiveCamera());

        // Capture snapshot when preview bound
        btnCapture.setOnClickListener(v -> captureImage());
    }

    // ----------- Capture Mode binding -----------
    private void startCameraForCapture() {
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 101);
            return;
        }

        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

                imageCapture = new ImageCapture.Builder().build();
                Preview preview = new Preview.Builder().build();

                preview.setSurfaceProvider(cameraPreview.getSurfaceProvider());

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);

                // Show preview + capture button; hide other UI to avoid overlap
                cameraPreview.setVisibility(View.VISIBLE);
                liveView.setVisibility(View.GONE);
                viewPager.setVisibility(View.GONE);

                buttonPanel.setVisibility(View.VISIBLE);
                btnCapture.setVisibility(View.VISIBLE);
                btnChoose.setVisibility(View.GONE);
                btnSave.setVisibility(View.GONE);
                btnStop.setVisibility(View.GONE);

            } catch (Exception e) {
                Log.e(TAG, "startCameraForCapture error: " + e.getMessage());
                Toast.makeText(this, "Camera failed to start for capture", Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void captureImage() {
        if (imageCapture == null) {
            Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show();
            return;
        }

        imageCapture.takePicture(ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageCapturedCallback() {
                    @Override
                    public void onCaptureSuccess(@NonNull ImageProxy image) {
                        Bitmap bmp = ImageUtils.imageProxyToBitmap(image);
                        image.close();
                        if (bmp != null) {
                            originalBitmap = bmp;
                            processAndDisplay();
                        } else {
                            Toast.makeText(MainActivity.this, "Capture returned empty bitmap", Toast.LENGTH_SHORT).show();
                        }

                        // restore UI
                        btnCapture.setVisibility(View.GONE);
                        btnChoose.setVisibility(View.VISIBLE);
                        btnSave.setVisibility(View.VISIBLE);
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e(TAG, "captureImage error: " + exception.getMessage());
                        Toast.makeText(MainActivity.this, "Capture failed", Toast.LENGTH_SHORT).show();
                    }
                });
    }

    // ----------- Live Camera with Filters -----------
    private void startLiveCamera() {
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 101);
            return;
        }

        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                liveActive = true;
                final int[] frameCounter = {0};

                analysis.setAnalyzer(cameraExecutor, image -> {
                    if (!liveActive) {
                        image.close();
                        return;
                    }

                    // throttle ONNX live processing to every 3rd frame to reduce CPU (safe fallback)
                    frameCounter[0]++;
                    boolean doProcess = true;
                    if ("ONNX".equalsIgnoreCase(currentMode)) {
                        // process fewer frames for heavy ONNX
                        doProcess = (frameCounter[0] % 3 == 0);
                    }

                    Bitmap bmp = ImageUtils.imageProxyToBitmap(image);
                    image.close();

                    if (bmp == null) return;

                    if (doProcess) {
                        Bitmap processed = SketchProcessor.applyEffect(bmp, currentMode);
                        runOnUiThread(() -> {
                            // show processed live
                            liveView.setVisibility(View.VISIBLE);
                            cameraPreview.setVisibility(View.GONE);
                            viewPager.setVisibility(View.GONE);
                            liveView.setImageBitmap(processed);
                        });
                    } else {
                        // optionally show a quick preview (raw) - we keep processing off-frame to avoid flicker
                    }
                });

                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, analysis);

                // hide panels and only show stop
                buttonPanel.setVisibility(View.GONE);
                btnStop.setVisibility(View.VISIBLE);

                Toast.makeText(this, "Live camera started", Toast.LENGTH_SHORT).show();

            } catch (Exception e) {
                Log.e(TAG, "startLiveCamera error: " + e.getMessage());
                Toast.makeText(this, "Live camera failed to start", Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void stopLiveCamera() {
        liveActive = false;
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        liveView.setVisibility(View.GONE);
        cameraPreview.setVisibility(View.GONE);
        viewPager.setVisibility(View.VISIBLE);

        // restore panel
        buttonPanel.setVisibility(View.VISIBLE);
        btnStop.setVisibility(View.GONE);

        Toast.makeText(this, "Live stopped", Toast.LENGTH_SHORT).show();
    }

    // ----------- Process and Display (Upload or Capture result) -----------
    private void processAndDisplay() {
        if (originalBitmap == null) {
            Toast.makeText(this, "No image to process", Toast.LENGTH_SHORT).show();
            return;
        }

        // If user selected ONNX but ONNX failed to load, fallback to OpenCV with a message
        if ("ONNX".equalsIgnoreCase(currentMode) && !SketchProcessor.isOnnxLoaded()) {
            Toast.makeText(this, "ONNX model not loaded — falling back to OpenCV", Toast.LENGTH_SHORT).show();
            currentMode = "OpenCV";
        }

        processedBitmap = SketchProcessor.applyEffect(originalBitmap, currentMode);

        adapter.setBitmaps(originalBitmap, processedBitmap);

        // show ViewPager (which displays both images side-by-side)
        viewPager.setVisibility(View.VISIBLE);
        liveView.setVisibility(View.GONE);
        cameraPreview.setVisibility(View.GONE);

        // ensure UI buttons restored
        buttonPanel.setVisibility(View.VISIBLE);
        btnStop.setVisibility(View.GONE);
        btnCapture.setVisibility(View.GONE);
        btnChoose.setVisibility(View.VISIBLE);
        btnSave.setVisibility(View.VISIBLE);
    }

    // ----------- Save Image -----------
    private void saveImage() {
        if (processedBitmap == null) {
            Toast.makeText(this, "No processed image to save", Toast.LENGTH_SHORT).show();
            return;
        }
        Uri uri = ImageUtils.saveBitmapToGallery(this, processedBitmap);
        if (uri != null) {
            Toast.makeText(this, "Saved to Gallery", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Save failed", Toast.LENGTH_SHORT).show();
        }
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraExecutor != null) cameraExecutor.shutdown();
    }
}
