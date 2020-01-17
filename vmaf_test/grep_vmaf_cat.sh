#!/bin/sh
DIR="./lanczos_2160_to_720_vmaf";
#DIR="./lanczos_2160_to_544_vmaf";
echo $DIR
cat $DIR/QP*/result_mf_vcnn_up_rec_C* | grep VMAF_score
cat $DIR/QP*/result_mf_vcnn_up_rec_F* | grep VMAF_score
cat $DIR/QP*/result_mf_vcnn_up_rec_Run* | grep VMAF_score
cat $DIR/QP*/result_mf_vcnn_up_rec_Rus* | grep VMAF_score
cat $DIR/QP*/result_mf_vcnn_up_rec_T* | grep VMAF_score
