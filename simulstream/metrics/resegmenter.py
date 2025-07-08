import string
import sys
from itertools import zip_longest
from typing import List, Tuple

import numpy as np


def levenshtein_align_hypothesis_to_reference(
        hypothesis: List[str], reference: List[str]) -> List[str]:
    """
    Runs the Levenshtein algorithm to get the minimal set of edit operations to convert the full
    list of hypothesis words into the full list of reference words. The edit operations implicitly
    define an alignment between hypothesis and reference words. Using this alignment, the
    hypotheses are re-segmented to match the reference segmentation.

    This implementation is taken from https://github.com/apptek/SubER.
    """
    try:
        import Levenshtein
    except ImportError:
        sys.exit(
            "`Levenshtein` is required for evaluation and it is not installed. Install it with "
            "`pip install python_Levenshtein`.")
    remove_punctuation_table = str.maketrans('', '', string.punctuation)

    def normalize_word(word):
        """
        Lower-cases and removes punctuation as this increases the alignment accuracy.
        """
        word = word.lower()
        word_without_punctuation = word.translate(remove_punctuation_table)

        if not word_without_punctuation:
            return word  # keep tokens that are purely punctuation

        return word_without_punctuation

    all_reference_words = [
        normalize_word(word) for segment in reference for word in segment.split(" ")]
    all_hypothesis_words = [
        normalize_word(word) for segment in hypothesis for word in segment.split(" ")]

    reference_string, hypothesis_string = _map_words_to_characters(
        all_reference_words, all_hypothesis_words)

    opcodes = Levenshtein.opcodes(reference_string, hypothesis_string)

    reference_segment_lengths = [len(segment) for segment in reference]
    reference_segment_boundary_indices = np.cumsum(reference_segment_lengths)
    current_segment_index = 0
    aligned_hypothesis_word_lists = [[] for _ in reference]

    for opcode_tuple in opcodes:
        edit_operation = opcode_tuple[0]
        hypothesis_position_range = range(opcode_tuple[3], opcode_tuple[4])
        reference_position_range = range(opcode_tuple[1], opcode_tuple[2])

        if edit_operation in ("equal", "replace"):
            assert len(hypothesis_position_range) == len(hypothesis_position_range)
        elif edit_operation == "insert":
            assert len(reference_position_range) == 0
        elif edit_operation == "delete":
            assert len(hypothesis_position_range) == 0
        else:
            assert False, f"Invalid edit operation '{edit_operation}'."

        # 'zip_longest' is a "clever" way to unify the different cases:
        # for 'equal'/'replace' we iterate through hypothesis and reference position in parallel,
        # for 'insert'/'delete' either 'hypothesis_position' or 'reference_position' will be None
        # in the loop.
        for hypothesis_position, reference_position in zip_longest(
                hypothesis_position_range, reference_position_range):
            # Update current segment index depending on current reference position.
            current_reference_segment_boundary = reference_segment_boundary_indices[
                current_segment_index]
            if (reference_position is not None
                    and reference_position >= current_reference_segment_boundary):

                assert reference_position == current_reference_segment_boundary, \
                    "Bug: missing reference position in edit operations."
                current_segment_index += 1

                # If there are empty segments in the reference, we get double entries in
                # 'reference_segment_boundary_indices' (because the empty segment ends at the same
                # word index as the previous segment).
                # Skip these empty segments, we don't want to assign any hypothesis words to them.
                while (current_segment_index < len(reference_segment_boundary_indices)
                       and reference_segment_boundary_indices[current_segment_index]
                       == reference_segment_boundary_indices[current_segment_index - 1]):
                    current_segment_index += 1

            # Add hypothesis word to current segment in case of 'equal', 'replace' or 'insert'.
            if hypothesis_position is not None:
                word = all_hypothesis_words[hypothesis_position]
                aligned_hypothesis_word_lists[current_segment_index].append(word)

    aligned_hypothesis = [" ".join(word_list) for word_list in aligned_hypothesis_word_lists]

    return aligned_hypothesis


def _map_words_to_characters(
        reference_words: List[str], hypothesis_words: List[str]) -> Tuple[str, str]:
    """
    The Levenshtein module operates on characters, not list of strings. Therefore we map words to
    characters here.
    Inspired by https://github.com/jitsi/jiwer/blob/master/jiwer/measures.py.
    """
    unique_words = set(reference_words + hypothesis_words)
    vocabulary = dict(zip(unique_words, range(len(unique_words))))

    reference_string = "".join(chr(vocabulary[word] + 32) for word in reference_words)
    hypothesis_string = "".join(chr(vocabulary[word] + 32) for word in hypothesis_words)

    return reference_string, hypothesis_string
