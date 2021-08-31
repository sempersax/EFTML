python codeStart.py
python reweight_card.py
cp reweight_card.dat EFTML/Cards/reweight_card.dat
./bin/mg5_aMC run_EFT
rm -r EFTML/rw_me
rm -r EFTML/rw_me_2
cd EFTML/Events/
cd "$(\ls -1dt ./*/ | head -n 1)"
python ../../../preshower2.py
cd ../../../
python textMe.py
