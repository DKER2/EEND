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
output_dir=data
output_task_dir=

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
    tar -xf $ALIMEETING/Eval_Ali.tar.gz -C $ALIMEETING/
    rm -rf $ALIMEETING/Eval_Ali.tar.gz -C $ALIMEETING/

    wget -c --tries=0 --read-timeout=20 https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Train_Ali_far.tar.gz -P $ALIMEETING
    tar -xf $ALIMEETING/Train_Ali_far.tar.gz
    rm -rf $ALIMEETING/Train_Ali_far.tar.gz
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] ; then
    for name in train eval; do
        if [[ "$name" == "train" ]]; then
            textgrid_dir=downloads/Train_Ali_far/textgrid_dir
            wav_dir=downloads/Train_Ali_far/audio_dir
        fi
        output_task_dir="$output_dir/$name"
        mkdir -p $output_task_dir
        # Prepare the AliMeeting data
        echo "Prepare Kaldi Type Alimeeting data"
        find $wav_dir -name "*\.wav" > $output_task_dir/wavlist
        sort  $output_task_dir/wavlist > $output_task_dir/tmp
        cp $output_task_dir/tmp $output_task_dir/wavlist
        awk -F '/' '{print $NF}' $output_task_dir/wavlist | awk -F '.' '{print $1}' > $output_task_dir/uttid
        paste $output_task_dir/uttid $output_task_dir/wavlist > $output_task_dir/wav.scp
        paste $output_task_dir/uttid $output_task_dir/uttid > $output_task_dir/utt2spk
        cp $output_task_dir/utt2spk $output_task_dir/spk2utt
        cp $output_task_dir/uttid $output_task_dir/text

        echo "Process textgrid to obtain rttm label"
        find -L $textgrid_dir -iname "*.TextGrid" >  $output_task_dir/textgrid.flist
        sort  $output_task_dir/textgrid.flist  > $output_task_dir/tmp
        cp $output_task_dir/tmp $output_task_dir/textgrid.flist 
        paste $output_task_dir/uttid $output_task_dir/textgrid.flist > $output_task_dir/uttid_textgrid.flist
        while read text_file
        do  
            text_grid=`echo $text_file | awk '{print $1}'`
            echo $text_grid
            text_grid_path=`echo $text_file | awk '{print $2}'`
            python local/make_textgrid_rttm.py --input_textgrid_file $text_grid_path \
                                                --uttid $text_grid \
                                                --output_rttm_file $output_task_dir/${text_grid}.rttm
        done < $output_task_dir/uttid_textgrid.flist
        cat $output_task_dir/*.rttm > $output_task_dir/rttm
    done
fi
