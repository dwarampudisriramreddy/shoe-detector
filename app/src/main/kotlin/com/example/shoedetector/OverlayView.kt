package com.example.shoedetector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    private var detections: List<ShoeDetector.Detection> = emptyList()
    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        style = Paint.Style.FILL
    }
    private val textBackgroundPaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.FILL
        alpha = 160
    }

    fun setDetections(results: List<ShoeDetector.Detection>) {
        detections = results
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (detection in detections) {
            val x1 = detection.x1 * width
            val y1 = detection.y1 * height
            val x2 = detection.x2 * width
            val y2 = detection.y2 * height

            val rect = RectF(x1, y1, x2, y2)
            canvas.drawRect(rect, boxPaint)
            
            val text = "${detection.label} ${(detection.confidence * 100).toInt()}%"
            val textWidth = textPaint.measureText(text)
            canvas.drawRect(
                x1,
                y1 - 50f,
                x1 + textWidth + 10f,
                y1,
                textBackgroundPaint
            )
            canvas.drawText(text, x1 + 5f, y1 - 10f, textPaint)
        }
    }
}
