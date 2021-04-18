def summary_to_string(summary):
    """
    Converts summary from sumy type (list of Sentences) to one string
    """

    if len(summary) <= 0:
        return ""

    summary_str = str(summary[0])
    i = 1

    while i < len(summary):
        summary_str += ' ' + str(summary[i])
        i += 1

    return summary_str
