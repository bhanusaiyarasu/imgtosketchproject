package com.example.ribhance;

import android.graphics.Bitmap;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class ImageAdapter extends RecyclerView.Adapter<ImageAdapter.VH> {

    private Bitmap original, processed;

    public void setBitmaps(Bitmap orig, Bitmap proc) {
        this.original = orig;
        this.processed = proc;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public VH onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LinearLayout container = new LinearLayout(parent.getContext());
        container.setOrientation(LinearLayout.HORIZONTAL);
        container.setLayoutParams(new ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        ImageView iv1 = new ImageView(parent.getContext());
        ImageView iv2 = new ImageView(parent.getContext());

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f);
        iv1.setLayoutParams(lp);
        iv2.setLayoutParams(lp);

        iv1.setScaleType(ImageView.ScaleType.FIT_CENTER);
        iv2.setScaleType(ImageView.ScaleType.FIT_CENTER);

        container.addView(iv1);
        container.addView(iv2);

        return new VH(container, iv1, iv2);
    }

    @Override
    public void onBindViewHolder(@NonNull VH holder, int position) {
        holder.left.setImageBitmap(original);
        holder.right.setImageBitmap(processed);
    }

    @Override
    public int getItemCount() {
        // single page showing both images side-by-side
        return 1;
    }

    static class VH extends RecyclerView.ViewHolder {
        ImageView left, right;
        VH(@NonNull View itemView, ImageView l, ImageView r) {
            super(itemView);
            left = l;
            right = r;
        }
    }
}
