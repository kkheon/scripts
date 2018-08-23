#!/usr/bin/perl
#use strict
#use warnings

@INPUT= (
  "Campfire_Party"
, "Fountains"
, "Runners"
, "Rush_Hour"                                                                                             
, "Traffic_Flow"
);

$DIR_INPUT = '/hdd2T/kkheon/test_images/SJTU_4K_chopped';
$DIR_OUTPUT = '/hdd2T/kkheon/test_images/SJTU_4K_chopped_matlab_downsampled';
foreach $INPUT (@INPUT) {
    print("==== INPUT : $INPUT ====\n");
    system "matlab -nodesktop -nosplash -r \"yuv2yuv('$DIR_INPUT/$INPUT.yuv', 3840, 2160, '420', '$DIR_OUTPUT/bicubic_downsampled_$INPUT.yuv', 1920, 1080, '420'); exit;\"";
}

