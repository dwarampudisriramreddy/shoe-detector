package com.example.shoedetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections

class ShoeDetector(private val context: Context) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession

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
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        val imgSize = 640
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, imgSize, imgSize, true)
        
        // Pre-process: Bitmap to FloatBuffer (NCHW: 1, 3, 640, 640)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * imgSize * imgSize)
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
        
        val results = ortSession.run(Collections.singletonMap("images", inputTensor))
        val output = results[0].value as Array<*> // [1][13][8400]
        val data = output[0] as Array<FloatArray> // [13][8400]

        return postProcess(data, bitmap.width, bitmap.height)
    }

    private fun postProcess(data: Array<FloatArray>, imgWidth: Int, imgHeight: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numAnchors = 8400
        val numClasses = 9
        val confidenceThreshold = 0.45f

        for (i in 0 until numAnchors) {
            var maxClassScore = 0f
            var classId = -1

            // Find max class score (classes start at index 4)
            for (c in 0 until numClasses) {
                val score = data[c + 4][i]
                if (score > maxClassScore) {
                    maxClassScore = score
                    classId = c
                }
            }

            if (maxClassScore > confidenceThreshold) {
                // YOLOv8 output is [cx, cy, w, h] in pixels of 640x640
                val cx = data[0][i]
                val cy = data[1][i]
                val w = data[2][i]
                val h = data[3][i]

                val x1 = (cx - w / 2) * imgWidth / 640f
                val y1 = (cy - h / 2) * imgHeight / 640f
                val x2 = (cx + w / 2) * imgWidth / 640f
                val y2 = (cy + h / 2) * imgHeight / 640f

                detections.add(
                    Detection(
                        labels[classId],
                        maxClassScore,
                        x1, y1, x2, y2
                    )
                )
            }
        }

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
