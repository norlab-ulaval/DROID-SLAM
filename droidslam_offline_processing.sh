#!/bin/bash
source ~/DROID-SLAM/.venv/bin/activate

input_path_host=/home/mbo/bigfoot-FoMo/ijrr

process_trajectory() {
    local trajectory=$1
    local output_path_host="/home/mbo/output/droidslam-offline-final/droidslam-${trajectory}"

    for date_dir in "${input_path_host}"/*/; do
        date=$(basename "${date_dir}")
        for dataset_dir in "${date_dir}${trajectory}_"*/; do
            [ -d "${dataset_dir}" ] || continue
            dataset=$(basename "${dataset_dir}")
            if [[ $date != "2025-01-29" ]]; then
                continue
            fi
            echo $date
            mkdir -p "${output_path_host}/${date}/${dataset}"
            echo "Processing dataset ${date}/${dataset}"

            log_file="${output_path_host}/${date}/${dataset}/output.log"
            echo "START_TIME: $(date +%s)" > "$log_file"

            python -u demo.py \
                --imagedir "${dataset_dir}" \
                --stereo \
                --disable_vis \
                --trajectory_path "${output_path_host}/${date}/${dataset}/${dataset}_${dataset}.txt" \
                --buffer 10000 \
                --t0 0 \
                --stride 3 \
                --beta 0.3 \
                --filter_thresh 2.0 \
                --weights /home/mbo/legs_ws/src/droid_slam_ros/droid.pth \
                --warmup 4 \
                --pgo \
                --keyframe_thresh 4.0 \
                --frontend_thresh 16.0 \
                --frontend_window 50 \
                --frontend_radius 2 \
                --frontend_nms 1 \
                --backend_thresh 22.0 \
                --backend_radius 2 \
                --backend_nms 5 \
                2>&1 | tee -a "$log_file"

            echo "END_TIME: $(date +%s)" >> "$log_file"
        done
    done
}

process_trajectory "orange"
# process_trajectory "yellow"
# process_trajectory "blue"
# process_trajectory "green"
# process_trajectory "magenta"
# process_trajectory "red"
