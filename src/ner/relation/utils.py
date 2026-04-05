import json
import logging
import sys
import functools
import random
import os

from shared.data_structures import Dataset

logger = logging.getLogger('root')

def decode_sample_id(sample_id):
    def _parse_span(span_text):
        span_text = span_text.strip()
        if not (span_text.startswith('(') and span_text.endswith(')')):
            raise ValueError("Invalid span format: %s" % span_text)
        vals = span_text[1:-1].split(',')
        if len(vals) != 2:
            raise ValueError("Invalid span coordinates: %s" % span_text)
        return (int(vals[0]), int(vals[1]))

    try:
        doc_sent, pair = sample_id.split('::', 1)
    except ValueError:
        raise ValueError("Invalid sample id (missing '::'): %s" % sample_id)

    # Use the explicit boundary between subject/object spans to safely handle
    # negative indices such as (-1,-1).
    if ')-(' in pair:
        left, right = pair.split(')-(', 1)
        sub = _parse_span(left + ')')
        obj = _parse_span('(' + right)
    else:
        # Backward-compatible fallback for unexpected legacy formatting.
        pair_parts = pair.split('-', 1)
        if len(pair_parts) != 2:
            raise ValueError("Invalid sample id (missing span separator '-'): %s" % sample_id)
        sub = _parse_span(pair_parts[0])
        obj = _parse_span(pair_parts[1])

    return doc_sent, sub, obj

def generate_relation_data(entity_data, use_gold=False, context_window=0):
    """
    Prepare data for the relation model
    If training: set use_gold = True
    """
    logger.info('Generate relation data from %s'%(entity_data))
    data = Dataset(entity_data)

    nner, nrel = 0, 0
    max_sentsample = 0
    samples = []
    for doc in data:
        for i, sent in enumerate(doc):
            sent_samples = []

            nner += len(sent.ner)
            nrel += len(sent.relations)
            if use_gold:
                sent_ner = sent.ner
            else:
                sent_ner = sent.predicted_ner
            
            gold_ner = {}
            for ner in sent.ner:
                gold_ner[ner.span] = ner.label
            
            gold_rel = {}
            for rel in sent.relations:
                gold_rel[rel.pair] = rel.label
            
            sent_start = 0
            sent_end = len(sent.text)
            tokens = sent.text

            if context_window > 0:
                add_left = (context_window-len(sent.text)) // 2
                add_right = (context_window-len(sent.text)) - add_left

                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    tokens = context_to_add + tokens
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    tokens = tokens + context_to_add
                    add_right -= len(context_to_add)
                    j += 1
            
            for x in range(len(sent_ner)):
                for y in range(len(sent_ner)):
                    if x == y:
                        continue
                    sub = sent_ner[x]
                    obj = sent_ner[y]
                    label = gold_rel.get((sub.span, obj.span), 'no_relation')
                    sample = {}
                    sample['docid'] = doc._doc_key
                    sample['id'] = '%s@%d::(%d,%d)-(%d,%d)'%(doc._doc_key, sent.sentence_ix, sub.span.start_doc, sub.span.end_doc, obj.span.start_doc, obj.span.end_doc)
                    sample['relation'] = label
                    sample['subj_start'] = sub.span.start_sent + sent_start
                    sample['subj_end'] = sub.span.end_sent + sent_start
                    sample['subj_type'] = sub.label
                    sample['obj_start'] = obj.span.start_sent + sent_start
                    sample['obj_end'] = obj.span.end_sent + sent_start
                    sample['obj_type'] = obj.label
                    sample['token'] = tokens
                    sample['sent_start'] = sent_start
                    sample['sent_end'] = sent_end
                    sent_samples.append(sample)

            max_sentsample = max(max_sentsample, len(sent_samples))
            samples += sent_samples
    
    tot = len(samples)
    logger.info('#samples: %d, max #sent.samples: %d'%(tot, max_sentsample))

    return data, samples, nrel
