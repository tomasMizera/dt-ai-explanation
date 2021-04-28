import re
from IPython.core.display import display, HTML


def summary_to_string(summary, delim=' '):
    """
    Converts summary from sumy type (list of Sentences) to one string
    """

    if len(summary) <= 0:
        return ""

    summary_str = str(summary[0])
    i = 1

    while i < len(summary):
        summary_str += delim + str(summary[i])
        i += 1

    return summary_str


def highlight_summary(summary, class_names=None, summary_name=None, decision=None, minimal_word_weight=0.001):
    """
    Highlights important words from LIME explanation
    Display content immediately and do not return anything
    """

    # Normalize values to custom bounds https://stackoverflow.com/a/48109733/7875594
    def normalize(values, bounds):
        if bounds['actual']['upper'] == bounds['actual']['lower']:
            return values
        return [bounds['desired']['lower'] + (abs(x) - bounds['actual']['lower']) * (
                bounds['desired']['upper'] - bounds['desired']['lower']) /
                (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]

    colors = {0: '16,171,232', 1: '199,175,16'}

    start_highlight_tag = lambda col, a: f'<span style="background-color:rgba({colors[col]},{a});">'
    end_highlight_tag = '</span>'

    raw_text = summary[0]
    important_words_weights = summary[1]
    important_words = list(map(lambda x: x[0], important_words_weights))
    maxv = round(abs(important_words_weights[0][1]), 5)
    minv = round(abs(important_words_weights[-1][1]), 5)  # Here take abs for alpha calculations

    weights = list(map(lambda x: x[1], important_words_weights))
    upper_bound = 1
    lower_bound = 0.2
    # normalize weights to <1, 0.2> range to be usable as alpha color channel
    norm_weights = normalize(weights, {'actual': {'lower': minv, 'upper': maxv},
                                       'desired': {'lower': lower_bound, 'upper': upper_bound}})

    title = "<h2>Summary</h2>"
    if summary_name is not None:
        title = f'<h2>Summary: {summary_name}</h2>'

    legend = ""
    decision_html = ""
    if class_names is not None:
        legend = "<h4>Legend</h4>"
        legend += f'<span style="color:rgb({colors[0]});font-weight:bold">' + class_names[0] + "</span><br>"
        legend += f'Most significant <canvas width="200" height="10" style="border:1px solid #000000; ' \
                  f'background-image: linear-gradient(to left, ' \
                  f'rgba({colors[0]},{lower_bound}), rgba({colors[0]},{upper_bound}));"></canvas> Least significant'
        legend += "<br>"
        legend += f'<span style="color:rgb({colors[1]});font-weight:bold">' + class_names[1] + "</span><br>"
        legend += f'Most significant <canvas width="200" height="10" style="border:1px solid #000000; ' \
                  f'background-image: linear-gradient(to left, ' \
                  f'rgba({colors[1]},{lower_bound}), rgba({colors[1]},{upper_bound}));"></canvas> Least significant'
        legend += ""
        if decision is not None:
            dec = f'Model thinks this instance is <b>{class_names[decision]}</b>'
            decision_html = "<h3>Model decision</h3>" + dec

    for ix, word in enumerate(important_words):
        wx = important_words_weights[ix][1]

        if abs(wx) < minimal_word_weight:
            continue

        col = 1 if wx >= 0 else 0
        alpha = norm_weights[ix]

        # https://regex101.com/r/nNu7Rs/1
        pattern = r'(?<![><(=")\/])\b(' + word + r')\b(?!(:rgba)|(="back))'

        if word.isnumeric():
            # https://regex101.com/r/ZP4VV1/1
            pattern = r'(?<!\(|,)\b' + word + r'\b(?!\)|,)'

        raw_text = re.sub(pattern, start_highlight_tag(col, alpha) + word + end_highlight_tag, raw_text, flags=re.I)

    result = title + legend + decision_html + '<h3>Text</h3>' + raw_text + '<p style="margin-bottom:1cm;"></p>'

    # ims = list(filter(lambda x: abs(x[1] < minimal_word_weight), important_words_weights))
    # print(ims)
    print(raw_text)
    return display(HTML(result))
