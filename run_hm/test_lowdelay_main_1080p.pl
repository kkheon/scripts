#!/usr/bin/perl
#use strict
#use warnings

# ===== Settings ===== #
$WIDTH = "1920";
$HEIGHT = "1080";
$N_FRAME = "60";

@QP = (
#  "12",
#  "17",
  "22",
  "27",
  "32",
#  "37",
#  "42",
#  "47",
);

#$DIR = "/data/kkheon/dataset/SJTU_4K_test/label";
$DIR = "/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_1080";
$EXE = "./TAppEncoderStatic_16.9";
$CFG = "encoder_lowdelay_main.cfg";
$OUT_DIR = "$DIR\_hm";

opendir D, $DIR or die "Could not open dir: $!\n";
my @INPUT = grep(/[_0-9a-z]+.yuv/i, readdir D);

printf("coding : $EXE\n");
system "mkdir -p $OUT_DIR";

system "perl -pi -e 's/SourceWidth.+[0-9]+/SourceWidth                   :   $WIDTH/g' $CFG";
system "perl -pi -e 's/SourceHeight.+[0-9]+/SourceHeight                  :   $HEIGHT/g' $CFG";
system "perl -pi -e 's/FramesToBeEncoded.+[0-9]+/FramesToBeEncoded             :  $N_FRAME/g' $CFG";
system "perl -pi -e 's{InputFile.+}{InputFile                     : $DIR/0_down2_r000.yuv}g' $CFG";

foreach $QP (@QP) {
    foreach $INPUT (@INPUT) {
        print("==== INPUT : $INPUT ====\n");
        system "perl -pi -e 's{$DIR/[0-9a-zA-Z_]+.yuv}{$DIR/$INPUT}g' $CFG";
        
        #####     FrameRate     #####
        my @input_string= split /\./, $INPUT;

        # === parsing image name from filename === #
        my $image_name = $input_string[0];
        $image_name =~ s/_[0-9]+x[0-9]+//g; # remove resolution info
        #my $image_name = $INPUT;
        print("image name : $image_name\n");

        # === parsing FPS from filename === #
        #my $frame_rate = $input_string[2]; 
        my $frame_rate = 30; 

        system "perl -pi -e 's/FrameRate.+[0-9]+/FrameRate                     : $frame_rate/g' $CFG";

        print("==== QP$QP ====\n");
        system "mkdir -p $OUT_DIR/QP$QP";
        system "perl -pi -e 's/^QP.+[ ]+[0-9]+/QP                            : $QP/g' $CFG";

        print "$EXE -c $CFG\n";
        system "$EXE -c $CFG | tee result.txt";

        system "cat result.txt      > $OUT_DIR/QP$QP/result_$image_name.txt";
        system "cat str.bin         > $OUT_DIR/QP$QP/str_$image_name.bin";
        system "cat rec.yuv         > $OUT_DIR/QP$QP/rec_$image_name.yuv";
    }
}

#===== to telegram =====#
#=== get host name ===#
use Sys::Hostname;
my $host = hostname;

#=== get curr path ===#
use Cwd qw();
my $PATH = Cwd::abs_path();

$TOKENID="598336934:AAGKtE6tL9D8Ky30v0Fx1ZKbOqB9u1KEb5o";
$ID="55913643";
$msg="[$host] $PATH\/$0 is done.";
system "curl --data chat_id=$ID --data-urlencode \"text=$msg\" \"https://api.telegram.org/bot$TOKENID/sendMessage\" &> /dev/null";
