
def summary_to_string(summary):
    if len(summary) <= 0:
        return ""

    summary_str = str(summary[0])
    i = 1

    while i < len(summary):
        summary_str += ' ' + str(summary[i])
        i += 1

    return summary_str
