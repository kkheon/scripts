#!/usr/bin/perl
#use strict
#use warnings

# ===== Settings ===== #
#$DIR = "/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_mf1/result_QP32/result_mf_vcnn_down_3_hm/QP32";
#$DIR = "/data/kkheon/data_vsr_bak/val/val_bicubic_down_up/result_val_ld_org_bugfixed_x2/QP32";
#$DIR = "/data/kkheon/data_vsr_bak/val/val_t1_mf1_down_2x1/result_QP32/result_mf_vcnn_down_3_hm/QP32";
$DIR = "/data/kkheon/data_vsr_bak/val/val_t1_mf1_down_1x2/result_QP32/result_mf_vcnn_down_3_hm/QP32";
#$DIR = "/home/kkheon/dataset/myanmar_v1/orig_hm/val/QP32";
#$DIR = "/home/kkheon/VSR-Tensorflow-exp-mf1/data_vsr/val_t1_mf1_vcnn_fixed_ipppp/result_QP32/result_mf_vcnn_down_3_hm/QP32";
@EXE= ( 
#  "HM_120"
  "HM_120_v2"
);
#$OUT_DIR = "$DIR\_dec";
$OUT_DIR = "$DIR\_dec_v2";

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
        system "cat decoder_overall.txt      > $OUT_DIR/decoder_overall_$image_name.txt";
    }
}
