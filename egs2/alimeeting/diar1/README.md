From ESPNET Respository
```

cd /content/espnet/tools

./setup_anaconda.sh anaconda espnet 3.7

```
Continue to run
```
make TH_VERSION=1.7.0 CUDA_VERSION=10.1
```
To Prepare AliMeeting
```
cd egs2/alimeeting/
bash ./diar.sh --stage 1 --stop_stage 2
``` 
Processed Data will appear in dump folder

#Run Train
``` 
bash ./diar.sh --stage 3 --stop-stage 4
```