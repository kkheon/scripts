#!/usr/bin/perl
#use strict
#use warnings

@QP= (
  22 
, 27 
, 32 
, 37
, 42 
, 47
);

@INPUT= (
  "Campfire_Party"
, "Fountains"
, "Runners"
, "Rush_Hour"                                                                                             
, "Traffic_Flow"
);

#system "mkdir -p /backup/kkheon/test_sr/result_lanczos_up";
#system "mkdir -p /backup/kkheon/test_sr/result_vdsr_lanczos_up";
system "mkdir -p /backup/kkheon/test_sr/result_lanczos_up_lowdelay_main";
foreach $INPUT (@INPUT) {
    print("==== INPUT : $INPUT ====\n");
    foreach $QP (@QP) {
        print("==== QP$QP ====\n");
        system "mkdir -p /backup/kkheon/test_sr/result_lanczos_up_lowdelay_main/QP$QP";
        system "matlab -nodesktop -nosplash -r \"yuv2yuv_lanczos('/backup/kkheon/test_sr/result_TAppEncoderStatic_16.9-4K-matlab_bicubic_downsampled_lowdelay_main/QP$QP/rec_bicubic_downsampled_$INPUT.yuv', 1920, 1080, '420', '/backup/kkheon/test_sr/result_lanczos_up_lowdelay_main/QP$QP/lanczos_upsampled_bicubic_downsampled_$INPUT.yuv', 3840, 2160, '420'); exit;\"";
    }
}

