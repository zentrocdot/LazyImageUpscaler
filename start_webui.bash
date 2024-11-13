#!/usr/bin/bash
#
# Start Script
# Version 0.0.0.1

# Get script name.
FN=$0

# Check if script is executed as sudo.
function check_sudo {
    if [ "$EUID" -eq 0 ]; then
        # Print error message.
        echo -e "Run script NOT as sudo!"
        # Exit the script.
        exit 1
    fi
}
check_sudo

# -------------------------------
# Function make script executable
# -------------------------------
function make_executable {
    # Assign the function argument to the local variable.
    scriptname=$1
    # Make the script executable.
    if [[ "$(stat -c '%A' $0)" == *'x'* ]] ; then
        echo -e "Script is executable!\n"
    else
        echo -e "Script is NOT executable yet!\n"
        chmod u+x "${scriptname}"
    fi
    # Return the error code 0.
    return 0
}
make_executable "${FN}"

# Start the Lazy Image Upscaler.
python3 ./scripts/lazy_image_upscaler.py

# Exit script.
exit 0
