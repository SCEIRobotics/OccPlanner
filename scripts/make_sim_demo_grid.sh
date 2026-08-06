#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:-assets/sim/video}"
output_video="${2:-assets/sim/web/occplanner_sim_results_web.mp4}"
output_poster="${3:-assets/sim/poster/occplanner_sim_results_poster.jpg}"

names=(
  easy_1 hard_1 hard_2 hard_3
  commercial_1 commercial_2 commercial_3 home_1
  home_2 home_3 home_4 home_5
)

inputs=()
filter=""
for index in "${!names[@]}"; do
  inputs+=( -i "${input_dir}/${names[$index]}.mp4" )
  filter+="[${index}:v]setpts=PTS-STARTPTS,fps=10,"
  filter+="scale=400:470:force_original_aspect_ratio=decrease,"
  filter+="pad=400:470:(ow-iw)/2:(oh-ih)/2:color=white,"
  filter+="tpad=stop_mode=clone:stop_duration=12,trim=duration=11.3[v${index}];"
done

filter+="[v0][v1][v2][v3][v4][v5][v6][v7][v8][v9][v10][v11]"
filter+="xstack=inputs=12:layout="
filter+="0_0|400_0|800_0|1200_0|"
filter+="0_470|400_470|800_470|1200_470|"
filter+="0_940|400_940|800_940|1200_940:fill=white[grid]"

mkdir -p "$(dirname "$output_video")" "$(dirname "$output_poster")"

ffmpeg -y -v warning \
  "${inputs[@]}" \
  -filter_complex "$filter" \
  -map "[grid]" \
  -an \
  -c:v libx264 \
  -preset medium \
  -crf 21 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  -t 11.3 \
  "$output_video"

ffmpeg -y -v warning \
  -ss 0.5 \
  -i "$output_video" \
  -frames:v 1 \
  -update 1 \
  -q:v 3 \
  "$output_poster"

printf '%s\n%s\n' "$output_video" "$output_poster"
