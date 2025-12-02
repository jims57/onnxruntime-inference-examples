package ai.onnxruntime.example.superresolution

import ai.onnxruntime.*
import ai.onnxruntime.extensions.OrtxPackage
import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.*
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    private var inputImage: ImageView? = null
    private var outputImage: ImageView? = null
    private var superResolutionButton: Button? = null
    private var saveToDocumentsSwitch: Switch? = null
    private var statusText: TextView? = null
    
    // 后台线程执行器
    private lateinit var executorService: ExecutorService
    
    companion object {
        const val TAG = "ORTSuperResolution"
        private const val PERMISSION_REQUEST_CODE = 100
    }

    @SuppressLint("UseCompatLoadingForDrawables")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化线程池
        executorService = Executors.newSingleThreadExecutor()

        inputImage = findViewById(R.id.imageView1)
        outputImage = findViewById(R.id.imageView2)
        superResolutionButton = findViewById(R.id.super_resolution_button)
        saveToDocumentsSwitch = findViewById(R.id.switch_save_to_documents)
        statusText = findViewById(R.id.text_status)
        
        inputImage?.setImageBitmap(
            BitmapFactory.decodeStream(readInputImage())
        )

        // Initialize Ort Session and register the onnxruntime extensions package that contains the custom operators.
        // Note: These are used to decode the input image into the format the original model requires,
        // and to encode the model output into png format
        val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
        sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
        ortSession = ortEnv.createSession(readModel(), sessionOptions)

        superResolutionButton?.setOnClickListener {
            // 禁用按钮防止重复点击
            superResolutionButton?.isEnabled = false
            updateStatus("正在处理中...")
            
            // 在后台线程执行超分辨率处理
            executorService.execute {
                try {
                    val startTime = System.currentTimeMillis()
                    val result = performSuperResolution(ortSession)
                    val processingTime = System.currentTimeMillis() - startTime
                    
                    runOnUiThread {
                        if (result.outputBitmap != null) {
                            updateUI(result)
                            
                            // 如果开关打开，保存到文档
                            if (saveToDocumentsSwitch?.isChecked == true) {
                                saveResultToDocuments(result.outputBitmap!!)
                            }
                            
                            val sizeInfo = "${result.outputBitmap!!.width}x${result.outputBitmap!!.height}"
                            updateStatus("处理完成! 耗时: ${processingTime}ms, 输出尺寸: $sizeInfo")
                            Toast.makeText(baseContext, "Super resolution performed!", Toast.LENGTH_SHORT).show()
                        } else {
                            updateStatus("处理失败")
                            Toast.makeText(baseContext, "Failed to perform super resolution", Toast.LENGTH_SHORT).show()
                        }
                        superResolutionButton?.isEnabled = true
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Exception caught when perform super resolution", e)
                    runOnUiThread {
                        updateStatus("处理失败: ${e.message}")
                        Toast.makeText(baseContext, "Failed to perform super resolution", Toast.LENGTH_SHORT).show()
                        superResolutionButton?.isEnabled = true
                    }
                }
            }
        }
        
        // 请求存储权限
        requestStoragePermissions()
    }

    override fun onDestroy() {
        super.onDestroy()
        ortEnv.close()
        ortSession.close()
        executorService.shutdown()
    }

    private fun updateUI(result: Result) {
        outputImage?.setImageBitmap(result.outputBitmap)
    }
    
    private fun updateStatus(message: String) {
        statusText?.text = message
        Log.d(TAG, "状态: $message")
    }

    private fun readModel(): ByteArray {
        val modelID = R.raw.pytorch_superresolution_with_pre_post_processing_op18
        return resources.openRawResource(modelID).readBytes()
    }

    private fun readInputImage(): InputStream {
        return assets.open("Neomix-251107-192013-imag0004-smaller-by-75-percentage-by-loveimg.jpg")

        // return assets.open("Neomix-251107-192014-imag0006.jpg")
        // return assets.open("test_superresolution.png")
    }
    
    private fun readInputBitmap(): Bitmap {
        return BitmapFactory.decodeStream(assets.open("Neomix-251107-192014-imag0006.jpg"))
        // return BitmapFactory.decodeStream(assets.open("test_superresolution.png"))
    }

    private fun performSuperResolution(ortSession: OrtSession): Result {
        val superResPerformer = SuperResPerformer()
        val inputBitmap = readInputBitmap()
        
        Log.d(TAG, "输入图像尺寸: ${inputBitmap.width}x${inputBitmap.height}")
        
        // 使用分块处理大图像
        return superResPerformer.upscaleLargeImage(inputBitmap, ortEnv, ortSession)
    }
    
    // 保存结果图像到Documents目录
    private fun saveResultToDocuments(bitmap: Bitmap) {
        try {
            val documentsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
            if (!documentsDir.exists()) {
                documentsDir.mkdirs()
            }
            
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "super_resolution_result_$timestamp.png"
            val file = File(documentsDir, fileName)
            
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
            }
            
            Log.d(TAG, "图像已保存到: ${file.absolutePath}")
            Toast.makeText(this, "已保存到: $fileName", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "保存图像失败", e)
            Toast.makeText(this, "保存失败: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    // 请求存储权限
    private fun requestStoragePermissions() {
        if (Build.VERSION.SDK_INT >= 33) {
            // Android 13+ 不需要存储权限来访问应用特定目录
            return
        }
        
        val permissions = arrayOf(
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        )
        
        val needsPermission = permissions.any {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (needsPermission) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_REQUEST_CODE)
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            val allGranted = grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (!allGranted) {
                Toast.makeText(this, "需要存储权限来保存图像", Toast.LENGTH_LONG).show()
            }
        }
    }
}
