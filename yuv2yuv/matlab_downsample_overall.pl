#!/usr/bin/perl
#use strict
#use warnings

@DIRS=(
  "/data/kkheon/test_images/BVI"
);

#$YUV2YUV_FUNC = "yuv2yuv";
$YUV2YUV_FUNC = "yuv2yuv_lanczos";
$FILTER_NAME = "lanczos";

foreach $DIR (@DIRS){   

    opendir D, $DIR or die "Could not open dir: $!\n";
    my @INPUT = grep(/[_0-9a-z]+.yuv/i, readdir D);

    foreach $INPUT (@INPUT) {
        print("==== INPUT : $INPUT ====\n");
        
        $W = 1920;
        $H = 1080;
        print("matlab -nodesktop -nosplash -r \"yuv2yuv('$DIR/$INPUT', 3840, 2160, '420', '$DIR/$FILTER_NAME\_$INPUT\_${W}x${H}', $W, $H, '420'); exit;\"");
        system "matlab -nodesktop -nosplash -r \"yuv2yuv('$DIR/$INPUT', 3840, 2160, '420', '$DIR/$FILTER_NAME\_$INPUT\_${W}x${H}', $W, $H, '420'); exit;\"";
    }
}

