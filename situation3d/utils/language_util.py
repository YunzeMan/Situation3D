

def sqa3d_question_type(record):
    first_word = record.split()[0].lower()  # split the sentence into words and take the first one
    # change first word to lower case
    if first_word in ['what']:
        return 0
    elif first_word in ['is', 'are']:
        return 1
    elif first_word in ['how']:
        return 2
    elif first_word in ['can']:
        return 3
    elif first_word in ['which']:
        return 4
    elif first_word in ['if']:
        return 5
    elif first_word in ['where']:
        return 6
    elif first_word in ['am']:
        return 7
    else:
        return 8
