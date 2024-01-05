# Radar-Infineon
This code is designed for data collection using the Infineon-BGT60TR13C Radar. While raw data can be obtained using the provided radar GUI, this data collection program is intended for users who want to process radar data using custom code.

# Steps for Raw Data Collection
1. Ensure that you have Python installed on your system.
2. Download the Radar Fusion GUI from [this link](https://softwaretools.infineon.com/tools/com.ifx.tb.tool.radarfusiongui).
3. Install the `ifxAvian` library from the folder where Infineon is installed, for example: `C:\Infineon\Tools\Radar Development Kit\3.4.0.202304250920\assets\software\radar_sdk\radar_sdk\sdk\py\wrapper_avian\src`. You can find the `ifxAvian` folder and install it using the following command: `pip install ifxAvian`.
4. Note: Make sure to use the firmware version: `RadarBaseboardMCU7_v2.5.12` and do not change the firmware.
5. Run the program `data-collect.py`.
