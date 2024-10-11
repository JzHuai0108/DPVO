bagnames=(
"mocap_dark_fast"
"mocap_difficult"
"mocap_dark"
"mocap_easy"
"gym"
"indoor_floor"
"mocap_medium"
"outdoor_campus"
"outdoor_street")  # Use parentheses and quotes for array initialization

for bag in "${bagnames[@]}"; do  # Correct array referencing
    for i in $(seq 1 5); do  # Use seq to generate a sequence (1 to 4 inclusive)
        cmd="python3 demo.py --imagedir="data/rrxio/irs_rtvi_datasets_2021/$bag.bag" \
            --calib="calib/rrxio_thermal.yaml" \
            --stride=2 --save_trajectory --name=$bag$i"
        echo $cmd
        $cmd
    done
done
