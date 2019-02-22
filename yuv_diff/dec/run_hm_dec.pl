#!/usr/bin/perl
#use strict
#use warnings

# ===== Settings ===== #
$DIR = "/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_mf1/result_QP32/result_mf_vcnn_down_3_hm/QP32";
@EXE= ( 
  "HM_120"
);
$OUT_DIR = "$DIR\_dec";

opendir D, $DIR or die "Could not open dir: $!\n";
my @INPUT = grep(/[_0-9a-z]+.bin/i, readdir D);

foreach $EXE (@EXE) {
    printf("coding : $EXE\n");
    system "mkdir -p $OUT_DIR";

    foreach $INPUT (@INPUT) {
        print("==== INPUT : $INPUT ====\n");

        #my @input_filename = split /\/([^\/]+)$/, $INPUT;
        #print("filename : $input_filename[0]\n");
        #my @input_string = split /\./, $input_filename[0];

        my @input_string = split /\./, $INPUT;
        my $image_name = $input_string[0];
        print("image name : $image_name\n");
        
        # './HM_120 -b /home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_mf1/result_QP32/result_mf_vcnn_down_3_hm/QP32/str_mf_vcnn_down_scene_53.bin'
        print "./$EXE -b $DIR/$INPUT\n";
        system "./$EXE -b $DIR/$INPUT | tee result.txt";

        # result file list
        # - decoder_bit_lcu.txt
        # - decoder_bit_scu.txt
        # - decoder_cupu.txt
        # - decoder_intra.txt
        # - decoder_merge.txt
        # - decoder_mv.txt
        # - decoder_pred.txt
        # - decoder_sps.txt
        # - decoder_tile.txt
        # - decoder_tu.txt
        # - encoder_me.txt

        system "cat decoder_bit_lcu.txt      > $OUT_DIR/decoder_bit_lcu_$image_name.txt";
    }
}
