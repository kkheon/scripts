#!/usr/bin/perl
#use strict
#use warnings

#$DIR_PREFIX = "/hdd2T/kkheon/gen_dataset/output";
$DIR_PREFIX = "/hdd2T/kkheon/gen_dataset/output_bugfixed";
@DIRS=(
  "train"
, "val"
);

foreach $DIR_NAME (@DIRS){   
    $DIR = "$DIR_PREFIX/orig/scenes_yuv/$DIR_NAME";

    opendir D, $DIR or die "Could not open dir: $!\n";
    my @INPUT = grep(/[_0-9a-z]+.yuv/i, readdir D);

    ### out_dir
    $DIR_OUTPUT = "$DIR_PREFIX/downsampled/$DIR_NAME";
    system "mkdir -p $DIR_OUTPUT";

    foreach $INPUT (@INPUT) {
        print("==== INPUT : $INPUT ====\n");
        print("matlab -nodesktop -nosplash -r \"yuv2yuv('$DIR/$INPUT', 3840, 2160, '420', '$DIR_OUTPUT/$INPUT', 1920, 1080, '420'); exit;\"");
        system "matlab -nodesktop -nosplash -r \"yuv2yuv('$DIR/$INPUT', 3840, 2160, '420', '$DIR_OUTPUT/$INPUT', 1920, 1080, '420'); exit;\"";
    }
}

