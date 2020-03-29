( while true;do echo "Starting Framework!";unbuffer python framework.py;test $? -gt 128 && break; done ) | tee -a logs.txt
