ls eval_* | xargs -I {} sh -c 'echo {}; cat {} | python calculate_total_score_json.py'
