LOGS='./logs.txt'
echo '=== Starting batch run ===' >> $LOGS





echo 'Start creating directory structure ...' >> $LOGS
[ -d './results5.4' ] || mkdir './results5.4'
[ -d './results5.4/data.windows' ] || mkdir './results5.4/data.windows'
echo '... Finished creating directory structure' >> $LOGS





RPATH='./results5.4/data.windows/rh'
echo "Starting $RPATH ..." >> $LOGS
[ -d $RPATH ] || mkdir $RPATH
python3 train_tucana_v5.4.py --modelfile './tucana_v5.4.json' --databasepath './' --fileformat 'batch_{0:05d}.npy' --resultpath "$RPATH/" --batchsize 128 --epochs 150 --initialepoch 0 --verbose 2 --histogramfreq 0 --maxfilecount 1024 --nocache 1 --checkpoint 1 --userectangle 1 --usebartlett 0 --usehanning 1 --usemeyer 0 --usepoisson 0 --mindist 0.0 >>$RPATH/logs.txt 2>&1
echo "... done $RPATH" >> $LOGS





echo '=== Finished batch run ===' >> $LOGS