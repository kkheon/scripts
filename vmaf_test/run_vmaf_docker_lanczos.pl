#!/usr/bin/perl
#use strict
#use warnings

# ===== Settings ===== #
$WIDTH  = "3840";
$HEIGHT = "2160";

#@QP = (
#  "22",
#  "27",
#  "32",
#  "37",
#  "42",
#  "47",
#);
@QP = (
#  "12",
  "13",
  "14",
  "15",
  "16",
  "17",
  "18",
  "19",
  "20",
  "21",
  "22",
  "23",
  "24",
  "25",
  "26",
  "27",
  "28",
  "29",
  "30",
  "31",
  "32",
  "33",
  "34",
  "35",
  "36",
  "37",
  "38",
  "39",
  "40",
  "41",
  "42",
  "43",
  "44",
  "45",
  "46",
  "47",
);
# Lable Path
#$DIR_LABEL = "/data/kkheon/dataset/SJTU_4K_test/label";
#$DIR_LABEL = "/data/kkheon/dataset/ultra_video_group/org_chopped_from_0_n_60";
$DIR_LABEL = "/data/kkheon/dataset/ultra_video_group/org_chopped_from_0_n_5";

# Target Path 
@DIR_TARGET = (
  #"/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_1080_hm/lanczos_1080_to_2160",
  #"/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_720_hm/lanczos_720_to_2160",
  #"/data/kkheon/dataset/SJTU_4K_test/lanczos_2160_to_544_hm/lanczos_544_to_2160",
  #"/data/kkheon/dataset/ultra_video_group/label_hm",
  #"/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_1080_hm_lanczos_1080_to_2160",
  #"/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_1440_hm_lanczos_1440_to_2160",
  #"/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_720_hm_lanczos_720_to_2160",
  #"/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_544_hm_lanczos_544_to_2160",
  #"/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_360_hm_lanczos_360_to_2160",

  #"/data/kkheon/dataset/ultra_video_group/label_hm_5frm",
  "/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_1080_hm_5frm_lanczos_1080_to_2160",
  "/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_720_hm_5frm_lanczos_720_to_2160",
  "/data/kkheon/dataset/ultra_video_group/lanczos_2160_to_544_hm_5frm_lanczos_544_to_2160",
);

foreach $DIR_TARGET (@DIR_TARGET) {
  print("==== DIR_TARGET : $DIR_TARGET ====\n");

  # Docker File Mount 
  $OPTION = "-v $DIR_LABEL:$DIR_LABEL"; 
  $OPTION = $OPTION . " -v $DIR_TARGET:$DIR_TARGET";

  $CMD = "docker run --rm $OPTION vmaf run_vmaf yuv420p $WIDTH $HEIGHT"; 
  
  # ===== Out Dir ===== #
  my @dir_string = split /\//, $DIR_TARGET;
  my $DIR_NAME = $dir_string[-1];
  $OUT_DIR = "./$DIR_NAME\_vmaf";
  system "mkdir -p $OUT_DIR";
  
 
  foreach $QP (@QP) {
      # ===== Loop ===== #
      $DIR = "$DIR_TARGET/QP$QP";
      
      opendir D, $DIR or die "Could not open dir: $!\n";
      my @INPUT = grep(/[_0-9a-z]+.yuv/i, readdir D);
 
      print("==== QP$QP ====\n");
      system "mkdir -p $OUT_DIR/QP$QP";
  
      foreach $INPUT (@INPUT) {
          print("==== INPUT : $INPUT ====\n");
          
          # filename example : 
          #   rec_Campfire_Party_1280x720_3840x2160.yuv
          #   rec_Campfire_Party_1920x1080_3840x2160.yuv
          $INPUT_LABEL = $INPUT;
          $INPUT_LABEL =~ s/_[0-9]+x[0-9]+//g;
          $INPUT_LABEL =~ s/rec_//;
  
          $CFG = "$DIR_LABEL/$INPUT_LABEL $DIR/$INPUT --out-fmt json";
          print "$CMD $CFG\n";
          system "$CMD $CFG | tee result.json";
  
          # ===== save result ===== #
          my @input_string= split /\./, $INPUT;
          my $IMAGE_NAME = $input_string[0];
          system "cat result.json      > $OUT_DIR/QP$QP/result_$IMAGE_NAME.json";
      }
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
