#!/bin/bash
source ~/DROID-SLAM/.venv/bin/activate

input_path_host=/home/mbo/bigfoot-FoMo/ijrr

process_trajectory() {
    local trajectory=$1
    local output_path_host="/home/mbo/droidslam-offline/droidslam-${trajectory}"

    for date_dir in "${input_path_host}"/*/; do
        date=$(basename "${date_dir}")
        for dataset_dir in "${date_dir}${trajectory}_"*/; do
            [ -d "${dataset_dir}" ] || continue
            dataset=$(basename "${dataset_dir}")
            echo $dataset
            mkdir -p "${output_path_host}/${date}/${dataset}"
            echo "Processing dataset ${date}/${dataset}"

            log_file="${output_path_host}/${date}/${dataset}/output.log"
            echo "START_TIME: $(date +%s)" > "$log_file"

            python demo.py \
                --imagedir "${dataset_dir}" \
                --stereo \
                --disable_vis \
                --reconstruction_path "${output_path_host}/${date}/${dataset}" \
                --buffer 6000 \
                --t0 0 \
                --stride 3 \
                --beta 0.3 \
                --filter_thresh 2.0 \
                --warmup 4 \
                --keyframe_thresh 4.0 \
                --frontend_thresh 16.0 \
                --frontend_window 50 \
                --frontend_radius 2 \
                --frontend_nms 1 \
                --backend_thresh 22.0 \
                --backend_radius 2 \
                --backend_nms 3 \
                2>&1 | tee -a "$log_file"
                
            echo "END_TIME: $(date +%s)" >> "$log_file"
        done
    done
}

process_trajectory "red"
process_trajectory "orange"
process_trajectory "yellow"
process_trajectory "blue"
process_trajectory "green"
process_trajectory "magenta"


