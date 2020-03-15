#!/bin/bash
echo "Starting container as development environment!!!"
docker container run -i --name "v2d" scopeinfinity/video2description:deploy python parser.py server -s -m ~/v2d/models/ResNet_D512L512_G128G64_D1024D0.20BN_BDLSTM1024_D0.2L1024DVS_model.dat_4987_loss_2.203_Cider0.342_Blue0.353_Rouge0.572_Meteor0.256 &
docker container exec -i "v2d" python app.py &
echo "Tried turning up the processes! For now, please confirm yourself"

trap "kill $(jobs -p | tr '\n' ' ')" EXIT
for pid in $(jobs -p);do
    wait $pid;
done;