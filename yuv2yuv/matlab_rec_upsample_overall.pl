#!/usr/bin/perl
#use strict
#use warnings

@DIRS=(
  #"/data/kkheon/test_images/BVI"
  #"/backup/kkheon/test_images/BVI"
  #"/backup/kkheon/test_images/derf"
  #"/home/kkheon/exp_vdsr/exp_03"
  #"/home/kkheon/exp_vdsr/exp_02"

# "/data/kkheon/dataset/myanmar_v1/downsampled_lanczos_720/train_hm"
# "/data/kkheon/dataset/myanmar_v1/downsampled_lanczos_544/train_hm"

#  "/data/kkheon/dataset/myanmar_v1/downsampled_lanczos_2160_to_720/train_hm"
#  "/data/kkheon/dataset/myanmar_v1/downsampled_lanczos_2160_to_1080/train_hm"
  "/data/kkheon/dataset/myanmar_v1/downsampled_lanczos_2160_to_544/train_hm"

#  "/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_720_hm"
#  "/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_1080_hm"
#  "/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_544_hm"
);

@QP=(
##  17,
#  22,
#  27,
  32,
  37,
  42,
  47,
);

#$YUV2YUV_FUNC = "yuv2yuv";
#$FILTER_NAME = "bicubic";

$YUV2YUV_FUNC = "yuv2yuv_lanczos";
#$YUV2YUV_FUNC = "yuv2yuv_10bit_lanczos";
$FILTER_NAME = "lanczos";

#$W = 1920;
#$H = 1080;

#$W = 1280;
#$H = 720;

$W = 960;
$H = 544;

#=== target resolution ===#
# 4K
$W_SCALED = 3840;
$H_SCALED = 2160;

## FHD
#$W_SCALED = 1920;
#$H_SCALED = 1080;

foreach $DIR (@DIRS){   
    $OUT_DIR = "$DIR/$FILTER_NAME\_${H}_to\_${H_SCALED}";
    system "mkdir -p $OUT_DIR";

    foreach $QP (@QP) {
        $DIR_QP = "$DIR/QP$QP";
        $OUT_DIR_QP = "$OUT_DIR/QP$QP";
        system "mkdir -p $OUT_DIR_QP";

        opendir D, $DIR_QP or die "Could not open dir: $!\n";
        my @INPUT = grep(/[_0-9a-z]+.yuv/i, readdir D);

        foreach $INPUT (@INPUT) {
            print("==== INPUT : $INPUT ====\n");

            my @INPUT_STRING= split /\./, $INPUT;
            my $IMAGE_NAME = $INPUT_STRING[0];
            
            print("matlab -nodesktop -nosplash -r \"$YUV2YUV_FUNC('$DIR_QP/$INPUT', $W, $H, '420', '$OUT_DIR_QP/$IMAGE_NAME\_${W_SCALED}x${H_SCALED}.yuv', $W_SCALED, $H_SCALED, '420'); exit;\"");
            system "matlab -nodesktop -nosplash -r \"$YUV2YUV_FUNC('$DIR_QP/$INPUT', $W, $H, '420', '$OUT_DIR_QP/$IMAGE_NAME\_${W_SCALED}x${H_SCALED}.yuv', $W_SCALED, $H_SCALED, '420'); exit;\"";

        }
    }
}

