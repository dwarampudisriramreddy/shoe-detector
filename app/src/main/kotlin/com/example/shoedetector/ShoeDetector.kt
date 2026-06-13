package com.example.shoedetector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.Collections


class ShoeDetector(private val context: Context) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession
    private val inputName: String
    private val outputName: String

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
        Log.d("ShoeDetector", "Session created. Input: $inputName, Output: $outputName")
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val imgSize = 640
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imgSize, imgSize, true)
        
        // Pre-process: Bitmap to FloatBuffer (NCHW: 1, 3, 640, 640)
        val floatBuffer = ByteBuffer.allocateDirect(1 * 3 * imgSize * imgSize * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        
        val pixels = IntArray(imgSize * imgSize)
        resizedBitmap.getPixels(pixels, 0, imgSize, 0, 0, imgSize, imgSize)

        // R channel
        for (pixel in pixels) floatBuffer.put(((pixel shr 16) and 0xFF) / 255.0f)
        // G channel
        for (pixel in pixels) floatBuffer.put(((pixel shr 8) and 0xFF) / 255.0f)
        // B channel
        for (pixel in pixels) floatBuffer.put((pixel and 0xFF) / 255.0f)
        
        floatBuffer.rewind()

        val inputTensor = OnnxTensor.createTensor(ortEnv, floatBuffer, longArrayOf(1, 3, imgSize.toLong(), imgSize.toLong()))
        
        val results = ortSession.run(Collections.singletonMap(inputName, inputTensor))
        val outputTensor = results[0]
        Log.d("ShoeDetector", "Output info: ${outputTensor.info}")
        val output = outputTensor.value as Array<*> // [1][13][8400]
        
        // Robust handling of different output shapes
        val data = if (output[0] is Array<*>) {
            output[0] as Array<FloatArray> // [13][8400]
        } else {
            // If it's a primitive float[][], we might need to handle it differently in Kotlin
            // but usually ONNX Runtime returns Array<FloatArray> for 2D
            output[0] as Array<FloatArray>
        }

        return postProcess(data, bitmap.width, bitmap.height)
    }

    private fun postProcess(data: Array<FloatArray>, imgWidth: Int, imgHeight: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        // Auto-detect dimensions: YOLOv8 is usually [4+N, 8400]
        // But some exports might be [8400, 4+N]
        val isTransposed = data.size > data[0].size
        val numAnchors = if (isTransposed) data.size else data[0].size
        val numClasses = if (isTransposed) data[0].size - 4 else data.size - 4
        val confidenceThreshold = 0.15f // Lowered threshold further

        Log.d("ShoeDetector", "Processing: anchors=$numAnchors, classes=$numClasses, transposed=$isTransposed")

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

                val x1 = (cx - w / 2) * imgWidth / 640f
                val y1 = (cy - h / 2) * imgHeight / 640f
                val x2 = (cx + w / 2) * imgWidth / 640f
                val y2 = (cy + h / 2) * imgHeight / 640f

                detections.add(
                    Detection(
                        if (classId in labels.indices) labels[classId] else "Unknown",
                        maxClassScore,
                        x1, y1, x2, y2
                    )
                )
            }
        }

        Log.d("ShoeDetector", "Detections found: ${detections.size}, highest score: $highestScoreFound")
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
