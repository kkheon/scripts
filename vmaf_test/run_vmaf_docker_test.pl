#!/usr/bin/perl
#use strict
#use warnings

# ===== Settings ===== #
$WIDTH  = "3840";
$HEIGHT = "2160";

#@QP = (
#  "27",
##  "32",
#  "37",
#  "42",
#  "47",
#);

# Target Path
$DIR_ROOT = "/data/kkheon/dataset/SJTU_4K_test";

# Docker File Mount 
$OPTION = "-v $DIR_ROOT:$DIR_ROOT"; 
$OPTION = $OPTION . " -v /data/kkheon/vmaf_test/dataset_replace:/data/kkheon/vmaf_test/dataset_replace";
$OPTION = $OPTION . " -v /data/kkheon/test_images/SJTU_4K_chopped:/data/kkheon/test_images/SJTU_4K_chopped";
$CMD = "docker run --rm $OPTION vmaf run_vmaf yuv420p $WIDTH $HEIGHT"; 


$DIR = "/data/kkheon/vmaf_test/dataset_replace";
# ===== Out Dir ===== #
my @dir_string = split /\//, $DIR;
my $DIR_NAME = $dir_string[-1];
$OUT_DIR = "./$DIR_NAME\_vmaf";
system "mkdir -p $OUT_DIR";

# ===== Loop ===== #
#$DIR = "$DIR_ROOT/label";

opendir D, $DIR or die "Could not open dir: $!\n";
my @INPUT = grep(/[_0-9a-z]+.yuv/i, readdir D);

#foreach $QP (@QP) 
{
    #print("==== QP$QP ====\n");
    #system "mkdir -p $OUT_DIR/QP$QP";

    foreach $INPUT (@INPUT) {
        print("==== INPUT : $INPUT ====\n");

        my @input_string = split /_frm/, $INPUT;
        my $INPUT_NAME = $input_string[0];

        $CFG = "$DIR_ROOT/label/$INPUT_NAME.yuv $DIR/$INPUT --out-fmt json";
        print "$CMD $CFG\n";
        system "$CMD $CFG | tee result.json";

        # ===== save result ===== #
        my @input_string= split /\./, $INPUT;
        my $IMAGE_NAME = $input_string[0];
        #system "cat result.json      > $OUT_DIR/QP$QP/result_$IMAGE_NAME.json";
        system "cat result.json      > $OUT_DIR/result_$IMAGE_NAME.json";
    }
}

#===== To Telegram =====#
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
