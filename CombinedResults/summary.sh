ls result_* | xargs -I {} sh -c 'echo {}; [ -f eval_{} ] && cat eval_{} &&cat eval_{} | python calculate_total_score_json.py;python find_total_sentences_unique.py {};echo ""'
