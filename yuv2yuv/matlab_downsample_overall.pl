#!/usr/bin/perl
#use strict
#use warnings

@DIRS=(
  #"/data/kkheon/test_images/BVI"
  #"/backup/kkheon/test_images/BVI"
  "/backup/kkheon/test_images/derf"
);

#$YUV2YUV_FUNC = "yuv2yuv";
#$FILTER_NAME = "bicubic";

$YUV2YUV_FUNC = "yuv2yuv_lanczos";
#$YUV2YUV_FUNC = "yuv2yuv_10bit_lanczos";
$FILTER_NAME = "lanczos";

#$W = 1920;
#$H = 1080;

##$W = 3840;
$W = 4096;
$H = 2160;

foreach $DIR (@DIRS){   

    opendir D, $DIR or die "Could not open dir: $!\n";
    my @INPUT = grep(/[_0-9a-z]+.yuv/i, readdir D);

    foreach $INPUT (@INPUT) {
        print("==== INPUT : $INPUT ====\n");

        my @INPUT_STRING= split /\./, $INPUT;
        my $IMAGE_NAME = $INPUT_STRING[0];
        
        $W_SCALED = 1920;
        $H_SCALED = 1080;
        $OUT_DIR = "$DIR/$FILTER_NAME\_$H_SCALED";
        system "mkdir -p $OUT_DIR";
        print("matlab -nodesktop -nosplash -r \"$YUV2YUV_FUNC('$DIR/$INPUT', $W, $H, '420', '$OUT_DIR/$IMAGE_NAME\_${W_SCALED}x${H_SCALED}.yuv', $W_SCALED, $H_SCALED, '420'); exit;\"");
        system "matlab -nodesktop -nosplash -r \"$YUV2YUV_FUNC('$DIR/$INPUT', $W, $H, '420', '$OUT_DIR/$IMAGE_NAME\_${W_SCALED}x${H_SCALED}.yuv', $W_SCALED, $H_SCALED, '420'); exit;\"";

        $W_SCALED = 1280;
        $H_SCALED = 720;
        $OUT_DIR = "$DIR/$FILTER_NAME\_$H_SCALED";
        system "mkdir -p $OUT_DIR";
        print("matlab -nodesktop -nosplash -r \"$YUV2YUV_FUNC('$DIR/$INPUT', $W, $H, '420', '$OUT_DIR/$IMAGE_NAME\_${W_SCALED}x${H_SCALED}.yuv', $W_SCALED, $H_SCALED, '420'); exit;\"");
        system "matlab -nodesktop -nosplash -r \"$YUV2YUV_FUNC('$DIR/$INPUT', $W, $H, '420', '$OUT_DIR/$IMAGE_NAME\_${W_SCALED}x${H_SCALED}.yuv', $W_SCALED, $H_SCALED, '420'); exit;\"";

        $W_SCALED = 960;
        $H_SCALED = 544;
        $OUT_DIR = "$DIR/$FILTER_NAME\_$H_SCALED";
        system "mkdir -p $OUT_DIR";
        print("matlab -nodesktop -nosplash -r \"$YUV2YUV_FUNC('$DIR/$INPUT', $W, $H, '420', '$OUT_DIR/$IMAGE_NAME\_${W_SCALED}x${H_SCALED}.yuv', $W_SCALED, $H_SCALED, '420'); exit;\"");
        system "matlab -nodesktop -nosplash -r \"$YUV2YUV_FUNC('$DIR/$INPUT', $W, $H, '420', '$OUT_DIR/$IMAGE_NAME\_${W_SCALED}x${H_SCALED}.yuv', $W_SCALED, $H_SCALED, '420'); exit;\"";
    }
}

