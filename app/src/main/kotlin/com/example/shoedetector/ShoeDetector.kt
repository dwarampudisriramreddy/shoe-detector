package com.example.shoedetector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections

class ShoeDetector(private val context: Context) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession
    private val inputName: String
    private val outputName: String
    private val modelInputSize: Int

    private val labels = listOf(
        "Boots", "Clogs", "Dress_Shoes", "Flats",
        "Flip Flops", "Heels", "Mules", "Sandals", "Sneakers"
    )

    data class Detection(
        val label: String,
        val confidence: Float,
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float
    )

    init {
        val modelBytes = context.assets.open("shoe.onnx").readBytes()
        ortSession = ortEnv.createSession(modelBytes)
        inputName = ortSession.inputNames.iterator().next()
        outputName = ortSession.outputNames.iterator().next()
        
        // Auto-detect input size from model metadata
        val inputShape = ortSession.inputInfo[inputName]?.info?.asArray()?.get(2) ?: 640L
        modelInputSize = inputShape.toInt()
        
        Log.d("ShoeDetector", "Session created. Input: $inputName ($modelInputSize x $modelInputSize), Output: $outputName")
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val w = bitmap.width
        val h = bitmap.height
        val scale = modelInputSize.toFloat() / maxOf(w, h)
        val nw = (w * scale).toInt()
        val nh = (h * scale).toInt()
        
        // Letterboxing: Resize while maintaining aspect ratio, then pad to square
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, nw, nh, true)
        val letterboxedBitmap = Bitmap.createBitmap(modelInputSize, modelInputSize, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(letterboxedBitmap)
        canvas.drawColor(android.graphics.Color.BLACK)
        val left = (modelInputSize - nw) / 2f
        val top = (modelInputSize - nh) / 2f
        canvas.drawBitmap(resizedBitmap, left, top, null)
        
        // Pre-process: Bitmap to FloatBuffer (NCHW)
        val floatBuffer = ByteBuffer.allocateDirect(1 * 3 * modelInputSize * modelInputSize * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        
        val pixels = IntArray(modelInputSize * modelInputSize)
        letterboxedBitmap.getPixels(pixels, 0, modelInputSize, 0, 0, modelInputSize, modelInputSize)

        // R channel
        for (pixel in pixels) floatBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        // G channel
        for (pixel in pixels) floatBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        // B channel
        for (pixel in pixels) floatBuffer.put((pixel and 0xFF) / 255.0f)
        
        floatBuffer.rewind()

        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, longArrayOf(1, 3, modelInputSize.toLong(), modelInputSize.toLong()))
        
        val results = ortSession.run(Collections.singletonMap(inputName, inputTensor))
        val outputTensor = results[0]
        val output = outputTensor.value as Array<*> 
        
        val data = if (output[0] is Array<*>) {
            output[0] as Array<FloatArray>
        } else {
            output[0] as Array<FloatArray>
        }

        return postProcess(data, nw, nh, left, top)
    }

    private fun postProcess(
        data: Array<FloatArray>, 
        newWidth: Int,
        newHeight: Int,
        padLeft: Float,
        padTop: Float
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        val isTransposed = data.size > data[0].size
        val numAnchors = if (isTransposed) data.size else data[0].size
        val numClasses = if (isTransposed) data[0].size - 4 else data.size - 4
        val confidenceThreshold = 0.05f 

        var highestScoreFound = 0f
        for (i in 0 until numAnchors) {
            var maxClassScore = 0f
            var classId = -1

            for (c in 0 until numClasses) {
                val score = if (isTransposed) data[i][c + 4] else data[c + 4][i]
                if (score > maxClassScore) {
                    maxClassScore = score
                    classId = c
                }
            }
            if (maxClassScore > highestScoreFound) highestScoreFound = maxClassScore

            if (maxClassScore > confidenceThreshold) {
                val cx = if (isTransposed) data[i][0] else data[0][i]
                val cy = if (isTransposed) data[i][1] else data[1][i]
                val w = if (isTransposed) data[i][2] else data[2][i]
                val h = if (isTransposed) data[i][3] else data[3][i]

                // Final normalized coordinates relative to original image
                val normX1 = (cx - w / 2 - padLeft) / newWidth
                val normY1 = (cy - h / 2 - padTop) / newHeight
                val normX2 = (cx + w / 2 - padLeft) / newWidth
                val normY2 = (cy + h / 2 - padTop) / newHeight

                detections.add(
                    Detection(
                        if (classId in labels.indices) labels[classId] else "Unknown",
                        maxClassScore,
                        normX1, normY1, normX2, normY2
                    )
                )
            }
        }

        Log.d("ShoeDetector", "Detections: ${detections.size}, max score: $highestScoreFound")
        return nms(detections)
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val sortedDetections = detections.sortedByDescending { it.confidence }
        val selectedDetections = mutableListOf<Detection>()
        val iouThreshold = 0.45f

        for (detection in sortedDetections) {
            var keep = true
            for (selected in selectedDetections) {
                if (calculateIou(detection, selected) > iouThreshold) {
                    keep = false
                    break
                }
            }
            if (keep) selectedDetections.add(detection)
        }
        return selectedDetections
    }

    private fun calculateIou(d1: Detection, d2: Detection): Float {
        val x1 = maxOf(d1.x1, d2.x1)
        val y1 = maxOf(d1.y1, d2.y1)
        val x2 = minOf(d1.x2, d2.x2)
        val y2 = minOf(d1.y2, d2.y2)

        val intersectionArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = (d1.x2 - d1.x1) * (d1.y2 - d1.y1)
        val area2 = (d2.x2 - d2.x1) * (d2.y2 - d2.y1)

        return intersectionArea / (area1 + area2 - intersectionArea)
    }

    fun close() {
        ortSession.close()
        ortEnv.close()
    }
}
