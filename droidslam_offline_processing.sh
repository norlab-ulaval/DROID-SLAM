#!/bin/bash
source $HOME/DROID-SLAM/.venv/bin/activate

input_path_host=$HOME/bigfoot-FoMo/ijrr

process_trajectory() {
    local trajectory=$1
    local output_path_host="$HOME/output/droidslam-offline-timestamps-from-online/droidslam-${trajectory}"

    for date_dir in "${input_path_host}"/*/; do
        date=$(basename "${date_dir}")
        for dataset_dir in "${date_dir}${trajectory}_"*/; do
            [ -d "${dataset_dir}" ] || continue
            dataset=$(basename "${dataset_dir}")
            if [[ $date != "2024-11-21" || $date == "2024-11-28" || $date == "2025-01-29" || $date == "2025-03-10" ]]; then
                continue
            fi
            # if [[ $date == "2024-11-21" || $date == "2024-11-28" || $date == "2025-01-29" || $date == "2025-03-10" ]]; then
                echo $date
                mkdir -p "${output_path_host}/${date}/${dataset}"
                echo "Processing dataset ${date}/${dataset}"

                log_file="${output_path_host}/${date}/${dataset}/output.log"
                echo "START_TIME: $(date +%s)" > "$log_file"

                python -u demo.py \
                    --imagedir "${dataset_dir}" \
                    --stereo \
                    --timestamps_path "$PWD/timestamps.txt" \
                    --disable_vis \
                    --trajectory_path "${output_path_host}/${date}/${dataset}/${dataset}_${dataset}.txt" \
                    --buffer 10000 \
                    --t0 0 \
                    --stride 4 \
                    --beta 0.3 \
                    --filter_thresh 2.0 \
                    --warmup 4 \
                    --keyframe_thresh 6.0 \
                    --frontend_thresh 16.0 \
                    --frontend_window 50 \
                    --frontend_radius 2 \
                    --frontend_nms 1 \
                    --backend_thresh 22.0 \
                    --backend_radius 2 \
                    --backend_nms 15 \
                    --weights $HOME/DROID-SLAM/droid.pth \
                    2>&1 | tee -a "$log_file"

                    # --pgo \
                echo "END_TIME: $(date +%s)" >> "$log_file"
            # fi
        done
    done
}

# process_trajectory "yellow"
# process_trajectory "blue"
# process_trajectory "green"
# process_trajectory "magenta"
# process_trajectory "orange"
process_trajectory "red"
