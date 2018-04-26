ls result*.txt | xargs -I {} sh -c '[ ! -f eval_{} ] && python ../../cococaption/cocoeval.py {} | tail -n 1 > eval_{}'
