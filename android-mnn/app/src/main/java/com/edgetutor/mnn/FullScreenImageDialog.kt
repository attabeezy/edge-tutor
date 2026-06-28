package com.edgetutor.mnn

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.net.Uri
import android.view.ViewGroup
import android.widget.ImageView
import java.io.File

object FullScreenImageDialog {
    fun show(context: Context, path: String) {
        val image = ImageView(context).apply {
            setBackgroundColor(Color.BLACK)
            setImageURI(Uri.fromFile(File(path)))
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen).apply {
            setContentView(image, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            image.setOnClickListener { dismiss() }
            show()
        }
    }
}
