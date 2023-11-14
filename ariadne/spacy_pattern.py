import os

from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from spacy.tokens import Token, Span
from spacy.tokens import Doc

import pandas as pd

class InsultoPattern:

    def __init__(self, vocab, nlp):
        df_insulto = pd.read_csv(os.getcwd() +'/data/VIL_1/all_insultos.tsv')
        patterns_insulto = [nlp(insulto) for insulto in df_insulto['span'].to_list()]
        # Register a new token extension to flag bad HTML
        Span.set_extension("insultos_pat", default=False)
        Doc.set_extension("insultos_pattern", default=False)
        self.matcher = PhraseMatcher(vocab)
        self.matcher.add("Insultos_PATTERN", patterns_insulto)

    def __call__(self, doc):
        # This method is invoked when the component is called on a Doc
        matches = self.matcher(doc)
        # spans_month = [{'start': doc[start:end].start_char, 'end':doc[start:end].end_char} for match_id, start, end in matches(doc)]
        spans = []  # Collect the matched spans here
        spans_insulto = []
        for match_id, start, end in matches:
            span = doc[start:end]
            spans.append(span)
            spans_insulto.append({'id': span.start, 'start': span.start_char, 'end':span.end_char})
        doc._.insultos_pattern = spans_insulto
        with doc.retokenize() as retokenizer:
            for span in spans:
                span._.insultos_pat = True
                # retokenizer.merge(span)
                # for token in span:
                #     token._.insultos_pat = True  # Mark token as bad HTML
        return doc