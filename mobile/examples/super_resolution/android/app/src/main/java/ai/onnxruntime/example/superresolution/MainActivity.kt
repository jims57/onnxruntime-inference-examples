package ai.onnxruntime.example.superresolution

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Build.VERSION_CODES
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.io.*
import java.util.*


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var inputImage: ImageView? = null
    private var outputImage: ImageView? = null
    private var superResolutionButton: Button? = null
    
    companion object {
        const val TAG = "ORTSuperResolution"
        private const val STORAGE_PERMISSION_REQUEST_CODE = 1001
        private val REQUIRED_PERMISSIONS = if (Build.VERSION.SDK_INT >= 33) {
            arrayOf() // No storage permissions needed for Android 13+ (API 33+)
        } else {
            arrayOf(
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
                android.Manifest.permission.READ_EXTERNAL_STORAGE
            )
        }
    }

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImage = findViewById(R.id.imageView1)
        outputImage = findViewById(R.id.imageView2)
        superResolutionButton = findViewById(R.id.super_resolution_button)
        inputImage?.setImageBitmap(
            BitmapFactory.decodeStream(readInputImage())
        );

        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        superResolutionButton?.setOnClickListener {
            if (checkAndRequestPermissions()) {
                performSuperResolution()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
    }

    private fun updateUI(result: Result) {
        outputImage?.setImageBitmap(result.outputBitmap)
        
        // Save the result image to public Documents folder
        result.outputBitmap?.let { bitmap ->
            saveResultImageToDocuments(bitmap)
        }
    }
    
    private fun saveResultImageToDocuments(bitmap: Bitmap) {
        try {
            // Get public Documents directory
            val documentsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            
            // Create directory if it doesn't exist
            if (!documentsDir.exists()) {
                documentsDir.mkdirs()
            }
            
            // Create file with timestamp
            val timestamp = System.currentTimeMillis()
            val fileName = "super_resolution_result_${timestamp}.png"
            val outputFile = File(documentsDir, fileName)
            
            // Save bitmap to file
            FileOutputStream(outputFile).use { fos ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos)
                fos.flush()
            }
            
            Log.i(TAG, "Result image saved to: ${outputFile.absolutePath}")
            Toast.makeText(this, "Result saved to Documents: $fileName", Toast.LENGTH_LONG).show()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save result image", e)
            Toast.makeText(this, "Failed to save result image", Toast.LENGTH_SHORT).show()
        }
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.pytorch_superresolution_with_pre_post_processing_op18
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readInputImage(): InputStream {
        return assets.open("test_superresolution.png")
    }

    private fun performSuperResolution() {
        try {
            var superResPerformer = SuperResPerformer()
            var result = superResPerformer.upscale(readInputImage(), ortEnv, ortSession)
            updateUI(result)
            Toast.makeText(baseContext, "Super resolution performed!", Toast.LENGTH_SHORT)
                .show()
        } catch (e: Exception) {
            Log.e(TAG, "Exception caught when perform super resolution", e)
            Toast.makeText(baseContext, "Failed to perform super resolution", Toast.LENGTH_SHORT)
                .show()
        }
    }
    
    private fun checkAndRequestPermissions(): Boolean {
        if (REQUIRED_PERMISSIONS.isEmpty()) {
            return true // No permissions needed for Android 13+
        }
        
        val notGrantedPermissions = REQUIRED_PERMISSIONS.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        return if (notGrantedPermissions.isEmpty()) {
            true // All permissions granted
        } else {
            ActivityCompat.requestPermissions(
                this,
                notGrantedPermissions.toTypedArray(),
                STORAGE_PERMISSION_REQUEST_CODE
            )
            false
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == STORAGE_PERMISSION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                performSuperResolution()
            } else {
                Toast.makeText(
                    this,
                    "Storage permissions are required to save the result image",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }
}
