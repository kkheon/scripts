#!/usr/bin/perl
#use strict
#use warnings

@INPUT= (
# "Fountains"
  "Campfire_Party"
, "Runners"
, "Rush_Hour"                                                                                             
, "Traffic_Flow"
);

foreach $INPUT (@INPUT) {
    print("==== INPUT : $INPUT ====\n");
    system "matlab -nodesktop -nosplash -r \"yuv2yuv('/backup/kkheon/SJTU_4K/$INPUT.yuv', 3840, 2160, '420', '/backup/kkheon/SJTU_4K/matlab_downsampled/bicubic_downsampled_$INPUT.yuv', 1920, 1080, '420'); exit;\"";
}

