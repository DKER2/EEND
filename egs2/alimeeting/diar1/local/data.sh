# !/usr/bin/env bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
fs=8k
textgrid_dir=downloads/Eval_Ali/Eval_Ali_far/textgrid_dir
wav_dir=downloads/Eval_Ali/Eval_Ali_far/audio_dir
work_dir=downloads/Eval_Ali/Eval_Ali_far
output_dir=data/val

 . utils/parse_options.sh || exit 1;

if [ -z "${ALIMEETING}" ]; then
    log "Fill the value of 'LIBRIMIX' of db.sh"
    exit 1
fi
mkdir -p ${ALIMEETING}

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] ; then
    # download data
    wget -c --tries=0 --read-timeout=20 https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz -P $ALIMEETING
    tar -xf $ALIMEETING/Eval_Ali.tar.gz
    rm -rf $ALIMEETING/Eval_Ali.tar.gz
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] ; then
mkdir -p $output_dir
# Prepare the AliMeeting data
echo "Prepare Alimeeting data"
find $wav_dir -name "*\.wav" > $output_dir/wavlist
sort  $output_dir/wavlist > $output_dir/tmp
cp $output_dir/tmp $output_dir/wavlist
awk -F '/' '{print $NF}' $output_dir/wavlist | awk -F '.' '{print $1}' > $output_dir/uttid
paste $output_dir/uttid $output_dir/wavlist > $output_dir/wav.scp
paste $output_dir/uttid $output_dir/uttid > $output_dir/utt2spk
cp $output_dir/utt2spk $output_dir/spk2utt
cp $output_dir/uttid $output_dir/text

echo "Process textgrid to obtain rttm label"
find -L $textgrid_dir -iname "*.TextGrid" >  $output_dir/textgrid.flist
sort  $output_dir/textgrid.flist  > $output_dir/tmp
cp $output_dir/tmp $output_dir/textgrid.flist 
paste $output_dir/uttid $output_dir/textgrid.flist > $output_dir/uttid_textgrid.flist
while read text_file
do  
    text_grid=`echo $text_file | awk '{print $1}'`
    echo $text_grid
    text_grid_path=`echo $text_file | awk '{print $2}'`
    python local/make_textgrid_rttm.py --input_textgrid_file $text_grid_path \
                                        --uttid $text_grid \
                                        --output_rttm_file $output_dir/${text_grid}.rttm
done < $output_dir/uttid_textgrid.flist
cat $output_dir/*.rttm > $output_dir/rttm
fi