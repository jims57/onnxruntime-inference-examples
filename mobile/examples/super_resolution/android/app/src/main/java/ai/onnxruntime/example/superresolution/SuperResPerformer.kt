package ai.onnxruntime.example.superresolution

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.util.*

internal data class Result(
    var outputBitmap: Bitmap? = null
) {}

internal class SuperResPerformer {

    companion object {
        private const val TAG = "SuperResPerformer"
        // 模型期望的输入尺寸
        private const val MODEL_INPUT_SIZE = 224
        // 超分辨率放大倍数
        private const val UPSCALE_FACTOR = 3
        // 模型输出尺寸
        private const val MODEL_OUTPUT_SIZE = MODEL_INPUT_SIZE * UPSCALE_FACTOR // 672
        // 分块重叠像素，用于消除拼接边缘
        private const val TILE_OVERLAP = 16
    }

    // 原始的单图处理方法（仅适用于224x224图像）
    fun upscale(inputStream: InputStream, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        var result = Result()

        // Step 1: convert image into byte array (raw image bytes)
        val rawImageBytes = inputStream.readBytes()

        // Step 2: get the shape of the byte array and make ort tensor
        val shape = longArrayOf(rawImageBytes.size.toLong())

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            ByteBuffer.wrap(rawImageBytes),
            shape,
            OnnxJavaType.UINT8
        )
        inputTensor.use {
            // Step 3: call ort inferenceSession run
            val output = ortSession.run(Collections.singletonMap("image", inputTensor))

            // Step 4: output analysis
            output.use {
                val rawOutput = (output?.get(0)?.value) as ByteArray
                val outputImageBitmap =
                    byteArrayToBitmap(rawOutput)

                // Step 5: set output result
                result.outputBitmap = outputImageBitmap
            }
        }
        return result
    }

    // 分块处理大图像的方法
    fun upscaleLargeImage(inputBitmap: Bitmap, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        val result = Result()
        
        val srcWidth = inputBitmap.width
        val srcHeight = inputBitmap.height
        
        Log.d(TAG, "开始分块处理大图像: ${srcWidth}x${srcHeight}")
        
        // 如果图像小于等于模型输入尺寸，直接处理
        if (srcWidth <= MODEL_INPUT_SIZE && srcHeight <= MODEL_INPUT_SIZE) {
            Log.d(TAG, "图像尺寸小于模型输入尺寸，直接处理")
            return upscaleBitmap(inputBitmap, ortEnv, ortSession)
        }
        
        // 计算有效分块尺寸（减去重叠区域）
        val effectiveTileSize = MODEL_INPUT_SIZE - TILE_OVERLAP * 2
        
        // 计算需要多少个分块
        val tilesX = Math.ceil((srcWidth - TILE_OVERLAP * 2).toDouble() / effectiveTileSize).toInt().coerceAtLeast(1)
        val tilesY = Math.ceil((srcHeight - TILE_OVERLAP * 2).toDouble() / effectiveTileSize).toInt().coerceAtLeast(1)
        
        Log.d(TAG, "分块数量: ${tilesX}x${tilesY} = ${tilesX * tilesY}")
        
        // 创建输出图像
        val outputWidth = srcWidth * UPSCALE_FACTOR
        val outputHeight = srcHeight * UPSCALE_FACTOR
        val outputBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(outputBitmap)
        val paint = Paint().apply {
            isFilterBitmap = true
            isAntiAlias = true
        }
        
        var tileCount = 0
        val totalTiles = tilesX * tilesY
        
        for (tileY in 0 until tilesY) {
            for (tileX in 0 until tilesX) {
                tileCount++
                
                // 计算分块在源图像中的位置
                var srcX = tileX * effectiveTileSize
                var srcY = tileY * effectiveTileSize
                
                // 确保不超出边界
                if (srcX + MODEL_INPUT_SIZE > srcWidth) {
                    srcX = srcWidth - MODEL_INPUT_SIZE
                }
                if (srcY + MODEL_INPUT_SIZE > srcHeight) {
                    srcY = srcHeight - MODEL_INPUT_SIZE
                }
                
                // 确保不小于0
                srcX = srcX.coerceAtLeast(0)
                srcY = srcY.coerceAtLeast(0)
                
                // 计算实际分块尺寸
                val tileWidth = minOf(MODEL_INPUT_SIZE, srcWidth - srcX)
                val tileHeight = minOf(MODEL_INPUT_SIZE, srcHeight - srcY)
                
                Log.d(TAG, "处理分块 $tileCount/$totalTiles: 位置($srcX, $srcY), 尺寸(${tileWidth}x${tileHeight})")
                
                // 提取分块
                val tileBitmap = if (tileWidth == MODEL_INPUT_SIZE && tileHeight == MODEL_INPUT_SIZE) {
                    Bitmap.createBitmap(inputBitmap, srcX, srcY, tileWidth, tileHeight)
                } else {
                    // 如果分块尺寸不足，需要填充到MODEL_INPUT_SIZE
                    val paddedTile = Bitmap.createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888)
                    val tileCanvas = Canvas(paddedTile)
                    val srcTile = Bitmap.createBitmap(inputBitmap, srcX, srcY, tileWidth, tileHeight)
                    tileCanvas.drawBitmap(srcTile, 0f, 0f, null)
                    srcTile.recycle()
                    paddedTile
                }
                
                // 对分块进行超分辨率处理
                val tileResult = upscaleBitmap(tileBitmap, ortEnv, ortSession)
                tileBitmap.recycle()
                
                if (tileResult.outputBitmap == null) {
                    Log.e(TAG, "分块 $tileCount 处理失败")
                    continue
                }
                
                val upscaledTile = tileResult.outputBitmap!!
                
                // 计算输出位置
                val dstX = srcX * UPSCALE_FACTOR
                val dstY = srcY * UPSCALE_FACTOR
                
                // 计算实际需要绘制的区域（考虑重叠和边界）
                val drawWidth = if (tileWidth < MODEL_INPUT_SIZE) tileWidth * UPSCALE_FACTOR else upscaledTile.width
                val drawHeight = if (tileHeight < MODEL_INPUT_SIZE) tileHeight * UPSCALE_FACTOR else upscaledTile.height
                
                // 绘制到输出图像
                val srcRect = Rect(0, 0, drawWidth, drawHeight)
                val dstRect = Rect(dstX, dstY, dstX + drawWidth, dstY + drawHeight)
                canvas.drawBitmap(upscaledTile, srcRect, dstRect, paint)
                
                upscaledTile.recycle()
                
                Log.d(TAG, "分块 $tileCount/$totalTiles 完成")
            }
        }
        
        Log.d(TAG, "分块处理完成，输出尺寸: ${outputWidth}x${outputHeight}")
        result.outputBitmap = outputBitmap
        return result
    }
    
    // 处理单个Bitmap的方法
    private fun upscaleBitmap(bitmap: Bitmap, ortEnv: OrtEnvironment, ortSession: OrtSession): Result {
        val result = Result()
        
        // 将Bitmap转换为PNG字节数组
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
        val rawImageBytes = outputStream.toByteArray()
        
        // 创建输入张量
        val shape = longArrayOf(rawImageBytes.size.toLong())
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            ByteBuffer.wrap(rawImageBytes),
            shape,
            OnnxJavaType.UINT8
        )
        
        inputTensor.use {
            val output = ortSession.run(Collections.singletonMap("image", inputTensor))
            output.use {
                val rawOutput = (output?.get(0)?.value) as ByteArray
                result.outputBitmap = byteArrayToBitmap(rawOutput)
            }
        }
        
        return result
    }

    private fun byteArrayToBitmap(data: ByteArray): Bitmap {
        return BitmapFactory.decodeByteArray(data, 0, data.size)
    }
}