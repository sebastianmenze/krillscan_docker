Scans 'source_folder' for .raw files, processes and analyses them and stores them in 10min echograms in 'target_folder'. The settings .ini file in target_folder contains information on the echosounder frequency, calibration and data compression/email settings. In the current settings emails are send with 4h worth of of echograms, ca 1-2 Mb per mail.

To start the program the docker images need to be connected to the source and target volumes like this:

docker run -d --name krillscan -v YOUR SOURCE DIR:/source_folder -v YOUR TARGET DIR:/target_folder krillscan

To see if the program is running properly check if echogram files are appearing in target_folder. An overview of the already process files and pending files is given in: list_of_rawfiles.csv
